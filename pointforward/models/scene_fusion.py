"""Incremental cross-window and optional temporal scene attention."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class FourierPositionEncoding(nn.Module):
    def __init__(self, dim: int, bands: int = 6, scale: float = 0.1) -> None:
        super().__init__()
        self.scale = scale
        self.register_buffer("freqs", (2.0 ** torch.arange(bands)) * math.pi, persistent=False)
        self.proj = nn.Linear(3 * 2 * bands, dim)

    def forward(self, xyz: Tensor) -> Tensor:
        phase = (xyz * self.scale).unsqueeze(-1) * self.freqs
        return self.proj(torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2))


class _CrossAttentionLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_mult: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.norm_q = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.SiLU(inplace=True), nn.Linear(dim * ffn_mult, dim)
        )
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def _heads(self, value: Tensor) -> Tensor:
        return value.view(value.shape[0], self.heads, self.head_dim).transpose(0, 1)

    def forward(self, value: Tensor, query_position: Tensor, keys: Tensor) -> Tensor:
        query = self._heads(self.to_q(self.norm_q(value) + query_position))
        key = self._heads(self.to_k(keys))
        attended = F.scaled_dot_product_attention(query, key, self._heads(self.to_v(keys)))
        value = value + self.to_out(attended.transpose(0, 1).reshape(value.shape[0], -1))
        return value + self.ffn(self.norm_ffn(value))


class SceneFusion(nn.Module):
    def __init__(
        self, dim: int, heads: int = 8, layers: int = 2, ffn_mult: int = 2,
        pe_bands: int = 6, pe_scale: float = 0.1, voxel_size: float = 2.0,
        max_anchors: int = 8192, chunk_size: int = 32_768,
        grad_checkpoint: bool = True, deterministic_cache: bool = False,
    ) -> None:
        super().__init__()
        self.voxel_size = voxel_size
        self.max_anchors = max_anchors
        self.chunk_size = chunk_size
        self.grad_checkpoint = grad_checkpoint
        self.deterministic_cache = deterministic_cache
        self.pos = FourierPositionEncoding(dim, pe_bands, pe_scale)
        self.layers = nn.ModuleList(_CrossAttentionLayer(dim, heads, ffn_mult) for _ in range(layers))

    @torch.no_grad()
    def compact_anchors(
        self, features: Tensor, xyz: Tensor, is_static: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if is_static is not None:
            features, xyz = features[is_static], xyz[is_static]
        if features.shape[0] == 0:
            return features.new_zeros((0, features.shape[1])), xyz.new_zeros((0, 3))
        voxels = torch.floor(xyz / self.voxel_size).long()
        _, inverse = torch.unique(voxels, dim=0, return_inverse=True)
        count = int(inverse.max()) + 1
        denominator = torch.zeros(count, device=xyz.device).index_add_(
            0, inverse, torch.ones_like(inverse, dtype=torch.float32)
        ).clamp_min(1).unsqueeze(1)
        pooled_features = torch.zeros(
            count, features.shape[1], device=features.device, dtype=torch.float32
        ).index_add_(0, inverse, features.float()) / denominator
        pooled_xyz = xyz.new_zeros((count, 3)).index_add_(0, inverse, xyz) / denominator
        if count > self.max_anchors:
            selected = (
                torch.linspace(0, count - 1, self.max_anchors, device=xyz.device).long()
                if self.deterministic_cache else torch.randperm(count, device=xyz.device)[:self.max_anchors]
            )
            pooled_features, pooled_xyz = pooled_features[selected], pooled_xyz[selected]
        return pooled_features.to(features.dtype), pooled_xyz

    @torch.no_grad()
    def extend_cache(
        self, cache: tuple[Tensor, Tensor] | None, features: Tensor, xyz: Tensor,
        is_static: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        features, xyz = self.compact_anchors(features, xyz, is_static)
        if cache is not None and cache[0].shape[0]:
            features, xyz = torch.cat((cache[0], features)), torch.cat((cache[1], xyz))
            if self.deterministic_cache:
                return self.compact_anchors(features, xyz)
        if features.shape[0] > self.max_anchors:
            selected = torch.randperm(features.shape[0], device=features.device)[:self.max_anchors]
            features, xyz = features[selected], xyz[selected]
        return features, xyz

    def _fuse(self, value: Tensor, position: Tensor, keys: Tensor) -> Tensor:
        for layer in self.layers:
            value = layer(value, position, keys)
        return value

    def forward(self, query: Tensor, query_xyz: Tensor, anchors: Tensor, anchor_xyz: Tensor) -> Tensor:
        if anchors.shape[0] == 0:
            return query
        keys = anchors + self.pos(anchor_xyz)
        chunk = self.chunk_size or query.shape[0]
        outputs = []
        for start in range(0, query.shape[0], chunk):
            value = query[start:start + chunk]
            position = self.pos(query_xyz[start:start + chunk])
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                value = checkpoint(self._fuse, value, position, keys, use_reentrant=False)
            else:
                value = self._fuse(value, position, keys)
            outputs.append(value)
        return torch.cat(outputs)


@dataclass
class SceneTokenMemory:
    layer_inputs: tuple[Tensor, ...]
    next_position: int


class SceneTemporalAttention(nn.Module):
    def __init__(
        self, feat_dim: int, dim: int, heads: int = 8, layers: int = 2,
        max_memory_frames: int = 64,
        causal_inference: bool = True,
    ) -> None:
        super().__init__()
        if layers < 1 or max_memory_frames < 1 or dim % heads:
            raise ValueError("positive layers/memory and head-divisible dim required")
        self.max_memory_frames = max_memory_frames
        self.causal_inference = causal_inference
        self.view_proj = nn.Linear(feat_dim, dim)
        self.camera_pos = FourierPositionEncoding(dim)
        self.view_weight = nn.Linear(dim, 1)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                dim, heads, dim * 2, dropout=0.0, activation="gelu",
                batch_first=True, norm_first=True,
            ) for _ in range(layers)
        )
        self.register_buffer(
            "time_frequencies", torch.exp(-math.log(10000.0) * torch.arange(0, dim, 2) / dim),
            persistent=False,
        )

    def aggregate_views(
        self, features: Tensor, view_times: Tensor, camera_xyz: Tensor
    ) -> tuple[Tensor, Tensor]:
        views = self.view_proj(features.mean(dim=(-2, -1))) + self.camera_pos(camera_xyz)
        times, inverse = torch.unique(view_times, sorted=True, return_inverse=True)
        tokens = []
        for index in range(times.numel()):
            selected = views[inverse == index]
            tokens.append((selected * self.view_weight(selected).softmax(dim=0)).sum(dim=0))
        return torch.stack(tokens), inverse

    def forward(
        self, tokens: Tensor, memory: SceneTokenMemory | None = None
    ) -> tuple[Tensor, SceneTokenMemory]:
        if tokens.ndim != 2 or tokens.shape[0] == 0:
            raise ValueError("expected a nonempty (frames, dim) scene-token sequence")
        if memory is not None and len(memory.layer_inputs) != len(self.layers):
            raise ValueError("scene memory does not match temporal layer count")
        start = 0 if memory is None else memory.next_position
        positions = torch.arange(start, start + tokens.shape[0], device=tokens.device)
        phase = positions[:, None] * self.time_frequencies[None]
        current = tokens + torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(1)[:, :tokens.shape[1]]
        cached = []
        for index, layer in enumerate(self.layers):
            history = None if memory is None else memory.layer_inputs[index]
            combined = current if history is None else torch.cat((history, current))
            cached.append(combined[-self.max_memory_frames:].detach())
            offset = combined.shape[0] - current.shape[0]
            query_index = torch.arange(current.shape[0], device=tokens.device) + offset
            key_index = torch.arange(combined.shape[0], device=tokens.device)
            blocked = key_index[None] < query_index[:, None] - self.max_memory_frames
            if not self.training and self.causal_inference:
                blocked |= key_index[None] > query_index[:, None]
            normalized = layer.norm1(combined).unsqueeze(0)
            attended = layer.self_attn(
                normalized[:, offset:], normalized, normalized, attn_mask=blocked, need_weights=False
            )[0].squeeze(0)
            current = current + layer.dropout1(attended)
            current = current + layer.dropout2(
                layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(current)))))
            )
        return current, SceneTokenMemory(tuple(cached), start + tokens.shape[0])