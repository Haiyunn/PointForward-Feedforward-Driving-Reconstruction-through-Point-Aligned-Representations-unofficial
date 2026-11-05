"""Permutation-invariant fusion for arbitrary camera and view counts."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .fusion import SpatialTemporalFusion, _mlp


class VariableViewSpatialTemporalFusion(SpatialTemporalFusion):
    """Spatial-temporal fusion with fixed two-input masked mean/max ``phi_v``."""

    def __init__(self, *args, hidden: int = 128, **kwargs) -> None:
        kwargs["num_views"] = 0
        super().__init__(*args, hidden=hidden, **kwargs)
        self.phi_v = _mlp(2, max(hidden // 4, 8), 1)
        self._phi_v_views = -1

    def _sigma_head(self, num_views: int, device: torch.device, dtype: torch.dtype) -> nn.Module:
        return self.phi_v

    def _aggregate_sigma(self, per_view_scalar: Tensor, visible: Tensor) -> Tensor:
        valid = visible.to(per_view_scalar.dtype)
        mean_value = (per_view_scalar * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(1.0)
        minimum = torch.finfo(per_view_scalar.dtype).min
        max_value = torch.where(
            visible, per_view_scalar, torch.full_like(per_view_scalar, minimum)
        ).amax(dim=0)
        max_value = torch.where(visible.any(dim=0), max_value, mean_value)
        return self.phi_v(torch.stack((mean_value, max_value), dim=-1)).squeeze(-1)