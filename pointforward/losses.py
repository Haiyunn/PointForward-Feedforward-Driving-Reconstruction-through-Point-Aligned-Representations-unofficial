"""Training objective and evaluation metrics.

The paper supervises rendering only, with

    L = L_l1 + 0.2 * L_LPIPS

There is no depth loss: the depth predictor is frozen, so photometric gradients
flow through the Gaussian parameters and the fusion / decoding MLPs only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

LPIPS_WEIGHT = 0.2


class LPIPSLoss(nn.Module):
    """Perceptual loss, kept frozen and in eval mode.

    Wraps the reference ``lpips`` package when available, and falls back to
    ``torchmetrics`` -- both expect inputs in [-1, 1].
    """

    def __init__(self, net: str = "vgg", chunk: int = 0) -> None:
        super().__init__()
        self.chunk = chunk
        self.backend: str
        try:
            import lpips  # type: ignore

            self.net = lpips.LPIPS(net=net)
            self.backend = "lpips"
        except ImportError:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

            self.net = LearnedPerceptualImagePatchSimilarity(net_type=net, normalize=False)
            self.backend = "torchmetrics"

        self.net.requires_grad_(False)
        self.net.eval()

    def train(self, mode: bool = True) -> "LPIPSLoss":
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Args: pred / target (B, 3, H, W) in [0, 1]. Returns a scalar."""
        pred = pred.clamp(0.0, 1.0) * 2.0 - 1.0
        target = target.clamp(0.0, 1.0) * 2.0 - 1.0
        c = self.chunk
        n = pred.shape[0]
        if c <= 0 or n <= c:
            return self.net(pred, target).mean()
        total = pred.new_zeros(())
        for i in range(0, n, c):
            total = total + self.net(pred[i : i + c], target[i : i + c]).sum()
        return total / n


class RenderLoss(nn.Module):
    """L = L_l1 + lpips_weight * L_LPIPS.

    Args:
        lpips_weight: 0.2 in the paper; set to 0 to skip the perceptual term
                      entirely (also skips constructing the VGG backbone).
    """

    def __init__(self, lpips_weight: float = LPIPS_WEIGHT, lpips_net: str = "vgg", lpips_chunk: int = 0) -> None:
        super().__init__()
        self.lpips_weight = lpips_weight
        self.lpips: LPIPSLoss | None = LPIPSLoss(lpips_net, chunk=lpips_chunk) if lpips_weight > 0 else None

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor | None = None) -> dict[str, Tensor]:
        """Args:
            pred / target: (B, 3, H, W) in [0, 1].
            mask:          (B, 1, H, W) optional; excludes pixels from L1 (e.g.
                           sky or unobserved regions).

        Returns a dict of scalars with keys ``loss``, ``l1`` and ``lpips``.
        """
        if mask is None:
            l1 = F.l1_loss(pred, target)
        else:
            m = mask.to(pred.dtype)
            l1 = ((pred - target).abs() * m).sum() / m.sum().clamp_min(1.0) / pred.shape[1]

        out = {"l1": l1, "loss": l1}
        if self.lpips is not None:
            perceptual = self.lpips(pred, target)
            out["lpips"] = perceptual
            out["loss"] = l1 + self.lpips_weight * perceptual
        return out


# --------------------------------------------------------------------- metrics


@torch.no_grad()
def psnr(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Peak signal-to-noise ratio over a batch of images in [0, 1].

    Args:
        mask: optional (B, 1, H, W) or (B, H, W); when given, restricts the
              MSE to the masked pixels (e.g. a dynamic-object mask) instead
              of averaging over the whole image.
    """
    diff = (pred.clamp(0, 1) - target.clamp(0, 1)) ** 2
    if mask is None:
        mse = diff.flatten(1).mean(dim=1).clamp_min(1e-10)
        return (-10.0 * torch.log10(mse)).mean()
    m = mask.to(diff.dtype)
    if m.dim() == diff.dim() - 1:
        m = m.unsqueeze(1)
    m = m.expand_as(diff)
    mse = (diff * m).sum().clamp_min(1e-10) / m.sum().clamp_min(1.0)
    return -10.0 * torch.log10(mse)


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.outer(g)


@torch.no_grad()
def ssim(
    pred: Tensor, target: Tensor, window_size: int = 11, sigma: float = 1.5, mask: Tensor | None = None
) -> Tensor:
    """Structural similarity, averaged over the batch and channels.

    Standard single-scale SSIM with a Gaussian window, matching the settings used
    by the 3DGS codebases the paper compares against.

    Args:
        mask: optional (B, 1, H, W) or (B, H, W); when given, the per-pixel
              similarity map is averaged only over the masked pixels (e.g. a
              dynamic-object mask) instead of the whole image.
    """
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    c = pred.shape[1]

    window = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    window = window.expand(c, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu_x = F.conv2d(pred, window, padding=pad, groups=c)
    mu_y = F.conv2d(target, window, padding=pad, groups=c)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_xx = F.conv2d(pred * pred, window, padding=pad, groups=c) - mu_xx
    sigma_yy = F.conv2d(target * target, window, padding=pad, groups=c) - mu_yy
    sigma_xy = F.conv2d(pred * target, window, padding=pad, groups=c) - mu_xy

    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)
    ssim_map = num / den

    if mask is None:
        return ssim_map.mean()
    m = mask.to(ssim_map.dtype)
    if m.dim() == ssim_map.dim() - 1:
        m = m.unsqueeze(1)
    m = m.expand_as(ssim_map)
    return (ssim_map * m).sum() / m.sum().clamp_min(1.0)


@torch.no_grad()
def depth_rmse(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """D-RMSE: root mean squared error of rendered depth against LiDAR."""
    diff = (pred - target) ** 2
    if mask is None:
        return diff.mean().sqrt()
    m = mask.to(diff.dtype)
    return ((diff * m).sum() / m.sum().clamp_min(1.0)).sqrt()
