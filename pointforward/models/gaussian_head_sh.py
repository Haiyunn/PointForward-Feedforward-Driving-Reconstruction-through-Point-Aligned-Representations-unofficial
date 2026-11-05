"""Spherical-harmonic Gaussian decoder used by SH model variants."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .gaussian_head import GaussianDecoder


class GaussianSHDecoder(GaussianDecoder):
    """Map fused query features to geometry and per-channel SH coefficients."""

    def __init__(
        self,
        in_dim: int = 128,
        sh_degree: int = 3,
        hidden: int = 256,
        max_offset: float = 1.0,
        scale_init: float = 0.05,
        max_scale: float = 2.0,
        chunk_size: int = 262_144,
        grad_checkpoint: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        if sh_degree < 0:
            raise ValueError("sh_degree must be >= 0")
        self.sh_degree = sh_degree
        self.num_sh = (sh_degree + 1) ** 2
        self.color_dim = self.num_sh * 3
        self.max_offset = max_offset
        self.max_scale = max_scale
        self.chunk_size = chunk_size
        self.grad_checkpoint = grad_checkpoint

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
        )
        self.head_sh = nn.Linear(hidden, self.num_sh * 3)
        self.head_offset = nn.Linear(hidden, 3)
        self.head_quat = nn.Linear(hidden, 4)
        self.head_scale = nn.Linear(hidden, 3)
        self.head_opacity = nn.Linear(hidden, 1)

        nn.init.zeros_(self.head_sh.weight)
        nn.init.zeros_(self.head_sh.bias)
        nn.init.zeros_(self.head_offset.weight)
        nn.init.zeros_(self.head_offset.bias)
        nn.init.zeros_(self.head_quat.weight)
        with torch.no_grad():
            self.head_quat.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        nn.init.zeros_(self.head_scale.weight)
        nn.init.constant_(self.head_scale.bias, float(torch.log(torch.tensor(scale_init))))
        nn.init.zeros_(self.head_opacity.weight)
        nn.init.zeros_(self.head_opacity.bias)

    def forward_chunk(self, features: Tensor, base_xyz: Tensor) -> dict[str, Tensor]:
        h = self.trunk(features)
        means = base_xyz + torch.tanh(self.head_offset(h)) * self.max_offset
        quats = F.normalize(self.head_quat(h), dim=-1)
        radius = self.head_scale(h).mean(dim=-1, keepdim=True).exp().clamp(max=0.20)
        return {
            "sh": self.head_sh(h).reshape(-1, self.num_sh, 3),
            "means": means,
            "quats": quats,
            "scales": radius.expand(-1, 3),
            "opacities": torch.sigmoid(self.head_opacity(h)).squeeze(-1),
        }