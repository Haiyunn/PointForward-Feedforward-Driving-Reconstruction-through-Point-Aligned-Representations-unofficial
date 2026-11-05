"""Image feature extractor (paper Sec. 3.1).

The paper uses a pre-trained EfficientNetV2 as the feature extractor and
fine-tunes it during reconstruction training (unlike the depth model, which
stays frozen).  We take multi-scale features from timm's ``tf_efficientnetv2_s``
and fuse them top-down into a single feature map F_i in R^{H' x W' x C} that
queries sample from by projection.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

DEFAULT_BACKBONE = "tf_efficientnetv2_s"


class FeatureExtractor(nn.Module):
    """EfficientNetV2-S trunk with a light top-down fusion neck.

    Args:
        out_channels: channel width C of the emitted feature map.
        backbone:     timm model name.
        pretrained:   load ImageNet weights.
        out_stride:   stride of the emitted feature map relative to the input.
                      Queries sample with ``grid_sample`` so a fractional stride
                      costs nothing; stride 4 keeps memory sane at 576x768.
        out_indices:  which timm feature stages to use.
    """

    def __init__(
        self,
        out_channels: int = 128,
        backbone: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        out_stride: int = 4,
        out_indices: tuple[int, ...] = (1, 2, 3, 4),
    ) -> None:
        super().__init__()
        import timm

        self.trunk = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        chs = self.trunk.feature_info.channels()
        self.out_stride = out_stride
        self.out_channels = out_channels

        # 1x1 lateral projections, then top-down addition (FPN-style).
        self.laterals = nn.ModuleList(nn.Conv2d(c, out_channels, kernel_size=1) for c in chs)
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    def forward(self, images: Tensor) -> Tensor:
        """Extract a fused feature map.

        Args:
            images: (B, 3, H, W) in [0, 1].

        Returns:
            (B, C, H // out_stride, W // out_stride)
        """
        x = (images - self.pixel_mean) / self.pixel_std
        feats = self.trunk(x)

        target = (
            images.shape[-2] // self.out_stride,
            images.shape[-1] // self.out_stride,
        )

        # Top-down: start from the coarsest stage and accumulate.
        fused: Tensor | None = None
        for lateral, feat in zip(reversed(self.laterals), reversed(feats)):
            proj = lateral(feat)
            if fused is None:
                fused = proj
            else:
                fused = proj + F.interpolate(
                    fused, size=proj.shape[-2:], mode="bilinear", align_corners=False
                )

        assert fused is not None
        if fused.shape[-2:] != target:
            fused = F.interpolate(fused, size=target, mode="bilinear", align_corners=False)
        return self.smooth(fused)


def sample_features(features: Tensor, uv: Tensor, height: int, width: int) -> Tensor:
    """Bilinearly sample a feature map at pixel coordinates.

    Args:
        features: (V, C, Hf, Wf) feature maps.
        uv:       (V, N, 2) pixel coordinates in the *input image* frame.
        height:   input image height H (not the feature map height).
        width:    input image width W.

    Returns:
        (V, N, C) sampled features.  Coordinates outside the image sample zero.
    """
    v, c, _, _ = features.shape
    # Normalise to [-1, 1] for grid_sample.
    norm_u = uv[..., 0] / (width - 1) * 2 - 1
    norm_v = uv[..., 1] / (height - 1) * 2 - 1
    grid = torch.stack((norm_u, norm_v), dim=-1).unsqueeze(2)  # (V, N, 1, 2)

    sampled = F.grid_sample(
        features,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (V, C, N, 1)
    return sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # (V, N, C)
