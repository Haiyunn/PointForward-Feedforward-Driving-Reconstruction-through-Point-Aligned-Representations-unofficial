"""Feature-map rendering and the RGB refinement UNet (paper Sec. 3.4).

Gaussians carry an N_c-dimensional colour feature, so rasterisation produces a
feature map rather than an image.  A lightweight multi-layer UNet then maps the
rendered feature map to three channels, yielding the final RGB image.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def render_features(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    intrinsics: Tensor,
    world_to_cam: Tensor,
    height: int,
    width: int,
    near_plane: float = 0.1,
    far_plane: float = 500.0,
    render_depth: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Rasterise feature-carrying Gaussians with gsplat.

    Args:
        means:        (N, 3) Gaussian centres in world space.
        quats:        (N, 4) rotation quaternions (w, x, y, z).
        scales:       (N, 3) per-axis scales.
        opacities:    (N,) opacities in [0, 1].
        colors:       (N, C) colour features.
        intrinsics:   (V, 3, 3)
        world_to_cam: (V, 4, 4)
        height, width: output resolution.
        render_depth: also return the expected-depth buffer, which the D-RMSE
            metric needs.  gsplat appends it as one extra channel.

    Returns:
        features (V, C, H, W) rendered feature maps.
        alpha    (V, 1, H, W) accumulated alpha.
        depth    (V, 1, H, W) expected depth, or None.
    """
    from gsplat import rasterization

    # Rasterisation runs in fp32 regardless of any surrounding autocast: the
    # projection and depth sorting are numerically sensitive, and gsplat's
    # kernels expect float32 inputs.  Gradients still flow back into whatever
    # dtype the caller used.
    device_type = means.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        rendered, alphas, _ = rasterization(
            means=means.float(),
            quats=quats.float(),
            scales=scales.float(),
            opacities=opacities.float(),
            colors=colors.float(),
            viewmats=world_to_cam.float(),
            Ks=intrinsics.float(),
            width=width,
            height=height,
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode="RGB+ED" if render_depth else "RGB",
            packed=True,
        )
    # gsplat returns (V, H, W, C) / (V, H, W, 1); with RGB+ED the final channel
    # of `rendered` is the expected depth.
    depth = None
    if render_depth:
        depth = rendered[..., -1:].permute(0, 3, 1, 2).contiguous()
        rendered = rendered[..., :-1]
    return (
        rendered.permute(0, 3, 1, 2).contiguous(),
        alphas.permute(0, 3, 1, 2).contiguous(),
        depth,
    )


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class FeatureToRGB(nn.Module):
    """Lightweight UNet mapping a rendered feature map to an RGB image.

    Args:
        in_channels: N_c, plus one extra channel for the accumulated alpha.
        base:        width of the first UNet level.
        levels:      number of downsampling stages.
    """

    def __init__(self, in_channels: int = 128, base: int = 64, levels: int = 2) -> None:
        super().__init__()
        self.levels = levels

        self.stem = _ConvBlock(in_channels + 1, base)

        downs: list[nn.Module] = []
        ups: list[nn.Module] = []
        ch = base
        for _ in range(levels):
            downs.append(_ConvBlock(ch, ch * 2))
            ch *= 2
        for _ in range(levels):
            ups.append(_ConvBlock(ch + ch // 2, ch // 2))
            ch //= 2

        self.downs = nn.ModuleList(downs)
        self.ups = nn.ModuleList(ups)
        self.head = nn.Conv2d(base, 3, kernel_size=3, padding=1)

    def forward(self, features: Tensor, alpha: Tensor) -> Tensor:
        """Args: features (B, C, H, W), alpha (B, 1, H, W). Returns (B, 3, H, W) in [0, 1]."""
        x = self.stem(torch.cat((features, alpha), dim=1))

        skips: list[Tensor] = []
        for down in self.downs:
            skips.append(x)
            x = down(F.avg_pool2d(x, kernel_size=2))

        for up, skip in zip(self.ups, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up(torch.cat((x, skip), dim=1))

        return torch.sigmoid(self.head(x))
