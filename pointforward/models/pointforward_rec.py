"""Feature-color PointForward with incremental SceneFusion reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor, nn

from .pointforward import Gaussians, PointForward, SceneInput, TargetViews
from .scene_fusion import SceneFusion, SceneTemporalAttention, SceneTokenMemory


@dataclass
class ReconstructionCache:
    token_features: Tensor
    token_xyz: Tensor
    static_gaussians: Gaussians | None = None
    scene_memory: SceneTokenMemory | None = None


class PointForwardRec(PointForward):
    """Feature-color model with cross-window attention and optional temporal attention."""

    def __init__(
        self, *args, scene_fusion: dict | None = None, gaussian_map: dict | None = None,
        temporal_attention: dict | None = None, **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.scene_fusion = SceneFusion(dim=self.fusion.out_dim, **(scene_fusion or {}))
        gaussian_map = gaussian_map or {}
        self.gaussian_voxel_size = float(gaussian_map.get("voxel_size", 0.5))
        self.max_cached_gaussians = int(gaussian_map.get("max_gaussians", 100_000))
        self.temporal_attention = None
        if temporal_attention is not None:
            self.fusion.spatial_only = True
            self.freeze_backbone_bn = True
            self.temporal_attention = SceneTemporalAttention(
                feat_dim=self.fusion.feat_dim, dim=self.fusion.out_dim, **temporal_attention
            )
            self.temporal_to_query = nn.Sequential(
                nn.LayerNorm(self.fusion.out_dim), nn.Linear(self.fusion.out_dim, self.fusion.out_dim)
            )
            nn.init.zeros_(self.temporal_to_query[-1].weight)
            nn.init.zeros_(self.temporal_to_query[-1].bias)

    def encode_window_tokens(
        self, scene: SceneInput, depth: Tensor, sampled: dict[str, Tensor],
        memory: SceneTokenMemory | None = None,
    ) -> tuple[Tensor, Tensor, SceneTokenMemory | None]:
        if self.temporal_attention is None:
            fused, sigma = super().encode_tokens(scene, depth, sampled)
            return fused, sigma, None
        features = self.extract_features(scene.images)
        tokens, view_to_time = self.temporal_attention.aggregate_views(
            features, scene.times, scene.cam_to_world[:, :3, 3]
        )
        temporal, memory = self.temporal_attention(tokens, memory)
        fused, sigma = super().encode_tokens(scene, depth, sampled, features=features)
        context = self.temporal_to_query(temporal)[view_to_time[sampled["view_id"]]]
        return fused + context, sigma, memory

    @staticmethod
    def _subset(value: Gaussians, index: Tensor) -> Gaussians:
        data = {
            field.name: getattr(value, field.name)[index]
            for field in fields(Gaussians) if field.name != "extras"
        }
        data["extras"] = {
            key: item[index] if item.ndim and item.shape[0] == value.count else item
            for key, item in value.extras.items()
        }
        return Gaussians(**data)

    @staticmethod
    def _concat(first: Gaussians, second: Gaussians) -> Gaussians:
        data = {
            field.name: torch.cat((getattr(first, field.name), getattr(second, field.name)))
            for field in fields(Gaussians) if field.name != "extras"
        }
        data["extras"] = {}
        return Gaussians(**data)

    @staticmethod
    def _detach(value: Gaussians) -> Gaussians:
        data = {
            field.name: getattr(value, field.name).detach()
            for field in fields(Gaussians) if field.name != "extras"
        }
        data["extras"] = {key: item.detach() for key, item in value.extras.items()}
        return Gaussians(**data)

    def _update_gaussian_map(
        self, current: Gaussians, history: Gaussians | None,
    ) -> tuple[Gaussians | None, Gaussians | None]:
        current_index = torch.nonzero(current.instance_id < 0, as_tuple=False).flatten()
        if history is not None:
            history_index = torch.nonzero(history.instance_id < 0, as_tuple=False).flatten()
            history = self._subset(history, history_index) if history_index.numel() else None
        if current_index.numel() == 0:
            return history, history
        current_map = self._subset(current, current_index)
        combined = current_map if history is None else self._concat(current_map, history)
        voxels = torch.floor(combined.means / self.gaussian_voxel_size).long()
        _, inverse = torch.unique(voxels, dim=0, return_inverse=True)
        first = torch.full(
            (int(inverse.max()) + 1,), combined.count, dtype=torch.long, device=inverse.device
        )
        first.scatter_reduce_(0, inverse, torch.arange(combined.count, device=inverse.device), reduce="amin")
        historical = first[first >= current_map.count]
        render_history = self._detach(self._subset(combined, historical)) if historical.numel() else None
        if first.numel() > self.max_cached_gaussians:
            first = first[torch.linspace(0, first.numel() - 1, self.max_cached_gaussians, device=first.device).long()]
        updated = self._detach(self._subset(combined, first))
        updated.instance_id.fill_(-1)
        updated.times.zero_()
        updated.sigma.fill_(1_000_000.0)
        return render_history, updated

    def reconstruct_scene(
        self, scene: SceneInput, cache: ReconstructionCache | None = None,
    ) -> tuple[Tensor, Gaussians, Gaussians, ReconstructionCache]:
        depth = self.predict_depth(scene)
        sampled = self.build_queries(scene, depth)
        fused, sigma, memory = self.encode_window_tokens(
            scene, depth, sampled, None if cache is None else cache.scene_memory
        )
        if cache is not None and cache.token_features.shape[0]:
            fused = self.scene_fusion(fused, sampled["world_xyz"], cache.token_features, cache.token_xyz)
        gaussians = self.decode_gaussians(fused, sigma, sampled)
        history, static = self._update_gaussian_map(
            gaussians, None if cache is None else cache.static_gaussians
        )
        if self.temporal_attention is not None and cache is not None:
            history = cache.static_gaussians
            if history is not None:
                history = self._detach(history)
                history.times = torch.full_like(history.times, -1.0)
        render_gaussians = gaussians if history is None else self._concat(gaussians, history)
        token_cache = self.scene_fusion.extend_cache(
            None if cache is None else (cache.token_features, cache.token_xyz),
            fused.detach(), sampled["world_xyz"].detach(), sampled["instance_id"] < 0,
        )
        return depth, gaussians, render_gaussians, ReconstructionCache(
            token_cache[0], token_cache[1], static, memory
        )

    def pose_gaussians(self, gaussians, boxes, frame: int, target_time: Tensor | float):
        posed = super().pose_gaussians(gaussians, boxes, frame, target_time)
        if self.temporal_attention is None:
            return posed
        available = torch.ones_like(gaussians.times, dtype=torch.bool)
        if not self.training and self.temporal_attention.causal_inference:
            available = gaussians.times <= target_time
        history = gaussians.times < 0
        present_static = available & ~history & (gaussians.instance_id < 0)
        indices = torch.cat((
            torch.nonzero(present_static, as_tuple=False).flatten(),
            torch.nonzero(history, as_tuple=False).flatten(),
        ))
        if history.any() and indices.numel():
            voxels = torch.floor(gaussians.means[indices] / self.gaussian_voxel_size).long()
            _, inverse = torch.unique(voxels, dim=0, return_inverse=True)
            occupied = torch.zeros(indices.numel(), dtype=torch.long, device=indices.device)
            occupied.scatter_reduce_(
                0, inverse, (~history[indices]).long(), reduce="amax", include_self=True
            )
            available[indices] &= ~(history[indices] & occupied[inverse].bool())
        posed["opacities"] = posed["opacities"] * available.to(posed["opacities"].dtype)
        return posed

    def reconstruct_window(
        self, scene: SceneInput, targets: TargetViews, cache: ReconstructionCache | None = None,
        render_depth: bool = False,
    ) -> tuple[dict[str, Tensor], ReconstructionCache, Gaussians]:
        depth, gaussians, rendered, cache = self.reconstruct_scene(scene, cache)
        height, width = scene.image_size
        output = self.render(
            rendered, scene.boxes, targets, targets.height or height, targets.width or width,
            render_depth,
        )
        output["depth"] = depth
        return output, cache, gaussians

    @torch.no_grad()
    def reconstruct_incremental(
        self, windows: list[tuple[SceneInput, TargetViews]], render_depth: bool = False,
    ) -> list[dict[str, Tensor]]:
        cache = None
        outputs = []
        for scene, targets in windows:
            output, cache, _ = self.reconstruct_window(scene, targets, cache, render_depth)
            outputs.append(output)
        return outputs