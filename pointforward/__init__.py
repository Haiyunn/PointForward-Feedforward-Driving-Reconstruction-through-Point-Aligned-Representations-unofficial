"""PointForward -- feedforward driving-scene reconstruction with point-aligned
representations.

Unofficial reproduction of arXiv:2605.11594.  The paper releases no code, so the
implementation follows the text of Sections 3.1-3.4 and Table 1-4; the depth
prior is the public InfiniDepth release and the scene graph follows OmniRe.
"""

from .losses import RenderLoss, depth_rmse, psnr, ssim
from .models import (
    InstanceBoxes,
    PointForward,
    SceneInput,
    TargetViews,
)

__version__ = "0.1.0"

__all__ = [
    "InstanceBoxes",
    "PointForward",
    "SceneInput",
    "TargetViews",
    "RenderLoss",
    "depth_rmse",
    "psnr",
    "ssim",
    "__version__",
]
