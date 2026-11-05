"""Shared CVPR scoring harness: PSNR / SSIM / LPIPS over three regions.

A single reusable entry point, :func:`score_frames`, scores a batch of rendered
frames against ground truth and returns PSNR, SSIM and LPIPS for three regions:

* ``full``         -- the whole image.
* ``non_sky``      -- pixels where ``sky_mask < 0.5`` (i.e. everything that is
                      not sky).  ``None`` when no sky mask is supplied.
* ``dynamic_bbox`` -- pixels selected by the supplied dynamic-region mask.
                      ``None`` when no such mask is supplied.

PSNR and SSIM are reused verbatim from :mod:`pointforward.losses` -- they are
*not* reimplemented here.  LPIPS wraps ``losses.LPIPSLoss(net='vgg')`` in a
module-level cache and is lazy + CPU-fallback safe: if the ``lpips`` package or
its weights cannot be loaded, the perceptual network is disabled, every
``lpips`` field comes back ``None`` and a warning is logged once -- nothing
crashes.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

# Reuse the existing metric implementations; never reimplement PSNR/SSIM.
from pointforward.losses import psnr as _psnr
from pointforward.losses import ssim as _ssim

logger = logging.getLogger(__name__)

# Module-level LPIPS cache, keyed by device string.  Construction can download
# VGG weights, so we build at most once per device and remember a hard failure
# so we do not retry (and re-warn) on every clip.
_LPIPS_CACHE: dict[str, object] = {}
_LPIPS_DISABLED = False


def _get_lpips(device: torch.device):
    """Return a cached, frozen LPIPSLoss(net='vgg') on ``device`` or ``None``.

    Lazy and failure-tolerant: any import / weight-load problem disables LPIPS
    for the rest of the process rather than raising.
    """
    global _LPIPS_DISABLED
    if _LPIPS_DISABLED:
        return None
    key = str(device)
    cached = _LPIPS_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from pointforward.losses import LPIPSLoss

        net = LPIPSLoss(net="vgg").to(device)
        net.eval()
    except Exception as exc:  # noqa: BLE001 - any failure must be non-fatal
        logger.warning(
            "LPIPS unavailable (%s: %s); lpips metrics will be None",
            type(exc).__name__,
            exc,
        )
        _LPIPS_DISABLED = True
        return None
    _LPIPS_CACHE[key] = net
    return net


def _as_keep_mask(mask: Tensor, like: Tensor, device: torch.device) -> Optional[Tensor]:
    """Normalise a mask to a float (B,1,H,W) keep-mask aligned with ``like``.

    Accepts (B,1,H,W) or (B,H,W), bool or float.  Returns ``None`` if, after
    thresholding at 0.5, the mask selects no pixels at all (an empty region is
    treated the same as an absent one).
    """
    m = mask.to(device)
    if m.dim() == like.dim() - 1:
        m = m.unsqueeze(1)
    if m.dtype == torch.bool:
        m = m.float()
    else:
        m = (m > 0.5).float()
    if float(m.sum()) < 1.0:
        return None
    return m


def _lpips_score(net, pred: Tensor, gt: Tensor, keep: Optional[Tensor]) -> Optional[float]:
    """LPIPS over the whole image, or over a region by blanking its complement.

    For a region, the complement is zeroed *identically* in both images so the
    perceptual difference there is ~0 and the score reflects the kept region.
    """
    if net is None:
        return None
    if keep is None:
        return float(net(pred, gt))
    return float(net(pred * keep, gt * keep))


@torch.no_grad()
def score_frames(
    pred: Tensor,
    gt: Tensor,
    *,
    sky_mask: Optional[Tensor] = None,
    dyn_bbox_mask: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Score rendered frames against ground truth over three regions.

    Args:
        pred, gt:       (B, 3, H, W) tensors in [0, 1].
        sky_mask:       optional (B,1,H,W) or (B,H,W); the ``non_sky`` region is
                        ``sky_mask < 0.5``.  Absent -> ``non_sky`` is ``None``.
        dyn_bbox_mask:  optional (B,1,H,W) or (B,H,W) selecting dynamic pixels.
                        Absent -> ``dynamic_bbox`` is ``None``.
        device:         device to score on; defaults to ``pred.device``.

    Returns:
        ``{"full": {...}, "non_sky": {...} | None, "dynamic_bbox": {...} | None}``
        where each inner dict is ``{"psnr", "ssim", "lpips"}`` and ``lpips`` may
        be ``None`` if the perceptual backbone is unavailable.
    """
    if device is None:
        device = pred.device
    pred = pred.to(device).clamp(0.0, 1.0)
    gt = gt.to(device).clamp(0.0, 1.0)
    net = _get_lpips(device)

    def _region(keep: Optional[Tensor]) -> Optional[dict]:
        if keep is None:
            return None
        return {
            "psnr": float(_psnr(pred, gt, mask=keep)),
            "ssim": float(_ssim(pred, gt, mask=keep)),
            "lpips": _lpips_score(net, pred, gt, keep),
        }

    full = {
        "psnr": float(_psnr(pred, gt)),
        "ssim": float(_ssim(pred, gt)),
        "lpips": _lpips_score(net, pred, gt, None),
    }

    non_sky_keep = None
    if sky_mask is not None:
        non_sky_keep = _as_keep_mask((sky_mask.to(device) < 0.5), pred, device)

    dyn_keep = None
    if dyn_bbox_mask is not None:
        dyn_keep = _as_keep_mask(dyn_bbox_mask, pred, device)

    return {
        "full": full,
        "non_sky": _region(non_sky_keep),
        "dynamic_bbox": _region(dyn_keep),
    }
