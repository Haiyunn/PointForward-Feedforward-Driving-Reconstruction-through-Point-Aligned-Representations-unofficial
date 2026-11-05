"""Gaussian decoding and temporal opacity (paper Sec. 3.4).

Each fused query q' is decoded into a 3D Gaussian

    g = (c, mu, r, s, o)

with a high-dimensional colour feature c in R^{N_c} (rather than raw RGB,
following UniSim / SplatAD), centre mu, rotation quaternion r, scale s and
opacity o.  Opacity is attenuated away from the query's own timestamp by

    o_{t'} = o * exp(-0.5 * ((t' - t) / (sigma + 1))^2)                     (7)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GaussianDecoder(nn.Module):
    """Map fused query features to Gaussian parameters.

    The centre is predicted as an offset from the query's own 3D coordinate,
    which keeps the point-aligned initialisation and lets the network refine it.

    Args:
        in_dim:        width of the fused query representation q'.
        color_dim:     N_c, width of the rendered colour feature.
        hidden:        MLP width.
        max_offset:    clamp on the predicted centre offset, in metres.
        scale_init:    initial (pre-activation) Gaussian scale, in metres.
        max_scale:     clamp on the decoded scale, in metres.
        chunk_size:    queries decoded per chunk; 0 disables chunking.
        grad_checkpoint: recompute each chunk in the backward pass.  With over a
                       million queries the trunk activations alone run to several
                       GB, so this is on by default.
    """

    def __init__(
        self,
        in_dim: int = 128,
        color_dim: int = 128,
        hidden: int = 256,
        max_offset: float = 1.0,
        scale_init: float = 0.05,
        max_scale: float = 2.0,
        chunk_size: int = 262_144,
        grad_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.color_dim = color_dim
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

        # Independent linear heads per Gaussian attribute.
        self.head_color = nn.Linear(hidden, color_dim)
        self.head_offset = nn.Linear(hidden, 3)
        self.head_quat = nn.Linear(hidden, 4)
        self.head_scale = nn.Linear(hidden, 3)
        self.head_opacity = nn.Linear(hidden, 1)

        # Start near identity rotation, small scales and mid opacity.
        nn.init.zeros_(self.head_offset.weight)
        nn.init.zeros_(self.head_offset.bias)
        nn.init.zeros_(self.head_quat.weight)
        with torch.no_grad():
            self.head_quat.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        nn.init.zeros_(self.head_scale.weight)
        nn.init.constant_(self.head_scale.bias, float(torch.log(torch.tensor(scale_init))))
        nn.init.zeros_(self.head_opacity.weight)
        nn.init.constant_(self.head_opacity.bias, 0.0)

    def forward(self, features: Tensor, base_xyz: Tensor) -> dict[str, Tensor]:
        """Decode Gaussians.

        Args:
            features: (N, in_dim) fused query representations q'.
            base_xyz: (N, 3) the query's own coordinate, used as the centre prior.
                      For dynamic queries this is the *canonical*-space position.

        Returns:
            dict with keys ``color`` (N, N_c), ``means`` (N, 3), ``quats`` (N, 4),
            ``scales`` (N, 3) and ``opacities`` (N,).
        """
        n = features.shape[0]
        chunk = self.chunk_size if self.chunk_size > 0 else n
        if n <= chunk:
            return self.forward_chunk(features, base_xyz)

        parts: list[dict[str, Tensor]] = []
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                out = checkpoint(
                    self.forward_chunk, features[lo:hi], base_xyz[lo:hi], use_reentrant=False
                )
            else:
                out = self.forward_chunk(features[lo:hi], base_xyz[lo:hi])
            parts.append(out)

        return {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}

    def forward_chunk(self, features: Tensor, base_xyz: Tensor) -> dict[str, Tensor]:
        """Decode one chunk of queries; see :meth:`forward`."""
        h = self.trunk(features)

        offset = torch.tanh(self.head_offset(h)) * self.max_offset
        means = base_xyz + offset

        quats = F.normalize(self.head_quat(h), dim=-1)
        scales = self.head_scale(h).exp().clamp(max=self.max_scale)
        opacities = torch.sigmoid(self.head_opacity(h)).squeeze(-1)

        return {
            "color": self.head_color(h),
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
        }


def temporal_opacity(opacity: Tensor, sigma: Tensor, query_time: Tensor, target_time: Tensor | float) -> Tensor:
    """Paper Eq. (7): Gaussian-in-time attenuation of opacity.

        o_{t'} = o * exp(-0.5 * ((t' - t) / (sigma + 1))^2)

    Args:
        opacity:     (N,) base opacity o.
        sigma:       (N,) lifespan parameter.
        query_time:  (N,) the query's own timestamp t.
        target_time: scalar or (N,) render timestamp t'.

    Returns:
        (N,) attenuated opacity.
    """
    if not torch.is_tensor(target_time):
        target_time = torch.as_tensor(target_time, device=opacity.device, dtype=opacity.dtype)
    delta = (target_time - query_time) / (sigma + 1.0)
    return opacity * torch.exp(-0.5 * delta.pow(2))
