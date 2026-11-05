"""Direct RGB rasterization for spherical-harmonic Gaussian variants."""

from __future__ import annotations

import torch
from torch import Tensor


def render_sh(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    sh_coeffs: Tensor,
    intrinsics: Tensor,
    world_to_cam: Tensor,
    height: int,
    width: int,
    sh_degree: int,
    near_plane: float = 0.1,
    far_plane: float = 500.0,
    render_depth: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Rasterize SH coefficients directly to RGB with gsplat."""
    from gsplat import rasterization

    with torch.autocast(device_type=means.device.type, enabled=False):
        rendered, alphas, _ = rasterization(
            means=means.float(),
            quats=quats.float(),
            scales=scales.float(),
            opacities=opacities.float(),
            colors=sh_coeffs.float(),
            viewmats=world_to_cam.float(),
            Ks=intrinsics.float(),
            width=width,
            height=height,
            near_plane=near_plane,
            far_plane=far_plane,
            sh_degree=sh_degree,
            render_mode="RGB+ED" if render_depth else "RGB",
            packed=True,
        )
    depth = None
    if render_depth:
        depth = rendered[..., -1:].permute(0, 3, 1, 2).contiguous()
        rendered = rendered[..., :-1]
    rgb = rendered.permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0)
    return rgb, alphas.permute(0, 3, 1, 2).contiguous(), depth