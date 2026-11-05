"""Experimental learned topology for the feature-Gaussian base model."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .rotation import quaternion_multiply, quaternion_to_matrix

if TYPE_CHECKING:
    from .pointforward import Gaussians


ACTIONS = ("keep", "update", "split", "clone", "merge")
KEEP, UPDATE, SPLIT, CLONE, MERGE = range(len(ACTIONS))
ATTRIBUTES = ("means", "quats", "scales", "opacities", "colors", "sigma", "times", "instance_id")


class GaussianTopologyPolicy(nn.Module):
    """Predict actions and relative reliability from fused multi-view tokens."""

    def __init__(self, feature_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim + 4, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.action_head = nn.Linear(hidden, len(ACTIONS))
        self.quality_head = nn.Linear(hidden, 1)
        self.pair_head = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        with torch.no_grad():
            self.action_head.bias[KEEP] = 1.0
        nn.init.zeros_(self.quality_head.weight)
        nn.init.zeros_(self.quality_head.bias)
        nn.init.zeros_(self.pair_head[-1].weight)
        nn.init.zeros_(self.pair_head[-1].bias)

    def forward(self, features: Tensor, scales: Tensor, opacity: Tensor) -> dict[str, Tensor]:
        geometry = torch.cat((scales.clamp_min(1e-6).log(), opacity[:, None]), dim=-1)
        hidden = self.trunk(torch.cat((features, geometry), dim=-1))
        logits = self.action_head(hidden)
        return {
            "logits": logits,
            "probabilities": logits.softmax(dim=-1),
            "quality": self.quality_head(hidden).squeeze(-1),
        }

    def pair_logits(self, left: Tensor, right: Tensor, distance: Tensor) -> Tensor:
        evidence = torch.cat(((left - right).abs(), distance[:, None]), dim=-1)
        return self.pair_head(evidence).squeeze(-1)


def _take(gaussians: Gaussians, indices: Tensor) -> Gaussians:
    values = {name: getattr(gaussians, name)[indices] for name in ATTRIBUTES}
    return replace(gaussians, **values, extras={name: value[indices] for name, value in gaussians.extras.items()})


class GaussianTopology(nn.Module):
    """One topology pass in world/static or canonical/instance coordinates.

    Training renders a soft candidate mixture (up to 6.5 times the input count).
    Evaluation materialises hard actions. Pair proposals are detached, bounded,
    and conservative: one disjoint cross-view pair per instance/time/voxel.
    This is a differentiable surrogate, not an exact discrete-policy gradient.
    """

    def __init__(
        self, feature_dim: int, color_dim: int, hidden: int = 64,
        voxel_size: float = 0.05, merge_radius: float = 0.02,
        feature_similarity: float = 0.9, merge_threshold: float = 0.5,
        max_input_points: int = 32768, temperature: float = 1.0,
        max_update: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0 < merge_radius <= voxel_size:
            raise ValueError("require 0 < merge_radius <= voxel_size")
        if not -1 <= feature_similarity <= 1 or not 0 <= merge_threshold <= 1:
            raise ValueError("invalid similarity or merge threshold")
        if max_input_points <= 0 or temperature <= 0 or max_update <= 0:
            raise ValueError("point budget, temperature and max_update must be positive")
        self.voxel_size = voxel_size
        self.merge_radius = merge_radius
        self.feature_similarity = feature_similarity
        self.merge_threshold = merge_threshold
        self.max_input_points = max_input_points
        self.temperature = temperature
        self.max_update = max_update
        self.color_dim = color_dim
        self.policy = GaussianTopologyPolicy(feature_dim, hidden)
        self.residual = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.SiLU(), nn.Linear(hidden, 11 + color_dim),
        )
        self.clone_offset = nn.Linear(feature_dim, 3)
        self.split_offset = nn.Linear(feature_dim, 1)
        for head in (self.residual[-1], self.clone_offset, self.split_offset):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        with torch.no_grad():
            self.clone_offset.bias[0] = 0.25

    @torch.no_grad()
    def propose_pairs(self, gaussians: Gaussians, features: Tensor, view_id: Tensor) -> Tensor:
        """Propose local pairs without an N-by-N distance matrix.

        A voxel's first point is its anchor. The nearest compatible point from
        another view is its partner; other points wait for a future pass.
        Exact time matching avoids averaging unrelated temporal envelopes.
        """
        count = gaussians.count
        if count < 2:
            return view_id.new_empty((0, 2))
        _, time_id = torch.unique(gaussians.times, return_inverse=True)
        keys = torch.cat((
            (gaussians.means / self.voxel_size).floor().long(),
            gaussians.instance_id[:, None], time_id[:, None],
        ), dim=-1)
        groups, group_id = torch.unique(keys, dim=0, return_inverse=True)
        indices = torch.arange(count, device=features.device)
        anchors = indices.new_full((groups.shape[0],), count)
        anchors.scatter_reduce_(0, group_id, indices, reduce="amin", include_self=True)
        anchor = anchors[group_id]
        distance = (gaussians.means - gaussians.means[anchor]).norm(dim=-1)
        similarity = functional.cosine_similarity(features, features[anchor], dim=-1)
        rotation_similarity = (gaussians.quats * gaussians.quats[anchor]).sum(-1).abs()
        scale_ratio = (gaussians.scales / gaussians.scales[anchor].clamp_min(1e-6))
        compatible = (
            (view_id != view_id[anchor]) & (distance <= self.merge_radius)
            & (similarity >= self.feature_similarity) & (rotation_similarity >= 0.95)
            & (scale_ratio >= 0.5).all(-1) & (scale_ratio <= 2.0).all(-1)
            & ((gaussians.sigma - gaussians.sigma[anchor]).abs() <= 0.1)
        )
        distances = torch.where(compatible, distance, torch.full_like(distance, float("inf")))
        nearest = distances.new_full((groups.shape[0],), float("inf"))
        nearest.scatter_reduce_(0, group_id, distances, reduce="amin", include_self=True)
        candidates = torch.where(compatible & (distance == nearest[group_id]), indices, count)
        partners = indices.new_full((groups.shape[0],), count)
        partners.scatter_reduce_(0, group_id, candidates, reduce="amin", include_self=True)
        valid = partners < count
        return torch.stack((anchors[valid], partners[valid]), dim=-1)

    def _update(self, gaussians: Gaussians, features: Tensor) -> Gaussians:
        delta = self.residual(features).tanh()
        delta_quat = functional.normalize(torch.cat((torch.ones_like(delta[:, :1]), 0.1 * delta[:, 6:9]), -1), dim=-1)
        return replace(
            gaussians,
            means=gaussians.means + self.max_update * delta[:, :3],
            scales=(gaussians.scales * (0.25 * delta[:, 3:6]).exp()).clamp(1e-6, 2.0),
            quats=functional.normalize(quaternion_multiply(delta_quat, gaussians.quats), dim=-1),
            opacities=torch.sigmoid(torch.logit(gaussians.opacities.clamp(1e-6, 1 - 1e-6)) + 0.5 * delta[:, 9]),
            colors=gaussians.colors + 0.1 * delta[:, 10:10 + self.color_dim],
            sigma=(gaussians.sigma * (0.25 * delta[:, -1]).exp()).clamp_min(1e-6),
        )

    def _merge(self, gaussians: Gaussians, pairs: Tensor, quality: Tensor) -> Gaussians:
        """Reliability-weighted moments, diagonalised in an interpolated frame.

        Off-diagonal covariance is discarded in that frame. Duplicate-view
        opacity is averaged, not summed. Neither operation preserves rendering
        exactly; the image loss must supervise this approximation.
        """
        left, right = pairs.unbind(-1)
        weight = quality[pairs].softmax(-1)

        def average(values: Tensor) -> Tensor:
            shape = (*weight.shape, *((1,) * (values.ndim - 1)))
            return (values[pairs] * weight.reshape(shape)).sum(dim=1)

        means = average(gaussians.means)
        left_quat, right_quat = gaussians.quats[left], gaussians.quats[right]
        sign = torch.where((left_quat * right_quat).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        quats = functional.normalize(weight[:, :1] * left_quat + weight[:, 1:] * right_quat * sign, dim=-1)
        rotations = quaternion_to_matrix(gaussians.quats[pairs])
        covariance = (rotations * gaussians.scales[pairs].square().unsqueeze(-2)) @ rotations.transpose(-1, -2)
        displacement = gaussians.means[pairs] - means[:, None]
        covariance = (weight[:, :, None, None] * (
            covariance + displacement.unsqueeze(-1) * displacement.unsqueeze(-2)
        )).sum(dim=1)
        frame = quaternion_to_matrix(quats)
        variances = (frame.transpose(-1, -2) @ covariance @ frame).diagonal(dim1=-2, dim2=-1)
        return replace(
            _take(gaussians, left), means=means, quats=quats,
            scales=variances.clamp_min(1e-12).sqrt(), opacities=average(gaussians.opacities),
            colors=average(gaussians.colors), sigma=average(gaussians.sigma),
        )

    def forward(
        self, gaussians: Gaussians, features: Tensor, view_id: Tensor,
        hard: bool | None = None,
    ) -> Gaussians:
        if features.shape[0] != gaussians.count or view_id.shape != (gaussians.count,):
            raise ValueError("features and view_id must align with input Gaussians")
        if gaussians.count > self.max_input_points:
            raise ValueError("topology input budget exceeded; lower num_static/num_dynamic or explicitly raise max_input_points")
        if gaussians.colors.shape != (gaussians.count, self.color_dim):
            raise ValueError("topology supports feature colors, not SH coefficients")
        if any(value.ndim == 0 or value.shape[0] != gaussians.count for value in gaussians.extras.values()):
            raise ValueError("Gaussian extras must be aligned per input point")
        hard = not self.training if hard is None else hard
        with torch.autocast(device_type=features.device.type, enabled=False):
            values = {name: getattr(gaussians, name).float() for name in ATTRIBUTES if name != "instance_id"}
            return self._forward_float(replace(gaussians, **values), features.float(), view_id, hard)

    def _forward_float(self, gaussians: Gaussians, features: Tensor, view_id: Tensor, hard: bool) -> Gaussians:
        count = gaussians.count
        indices = torch.arange(count, device=features.device)
        if count == 0:
            return replace(gaussians, extras={**gaussians.extras, "topology_gate": features.new_empty(0),
                                           "topology_action": indices, "topology_parent": indices, "topology_partner": indices})
        prediction = self.policy(features, gaussians.scales, gaussians.opacities)
        probabilities = (prediction["logits"] / self.temperature).softmax(-1)
        pairs = self.propose_pairs(gaussians, features, view_id)
        left, right = pairs.unbind(-1)
        distance = (gaussians.means[left] - gaussians.means[right]).norm(dim=-1) / self.merge_radius
        agreement = self.policy.pair_logits(features[left], features[right], distance).sigmoid()
        if hard:
            probabilities = functional.one_hot(probabilities.argmax(-1), len(ACTIONS)).to(features.dtype)
            agreement = (agreement >= self.merge_threshold).to(features.dtype)
        pair_gate = probabilities[left, MERGE] * probabilities[right, MERGE] * agreement
        merged_gate = features.new_zeros(count).index_add(0, left, pair_gate).index_add(0, right, pair_gate)
        keep_gate = probabilities[:, KEEP] + probabilities[:, MERGE] - merged_gate
        updated = self._update(gaussians, features)
        major_axis = updated.scales.argmax(-1)
        axis_mask = functional.one_hot(major_axis, 3).to(features.dtype)
        fraction = 0.25 + 0.5 * self.split_offset(features).sigmoid()
        rotations = quaternion_to_matrix(updated.quats)
        offset = (rotations @ (axis_mask * updated.scales * fraction).unsqueeze(-1)).squeeze(-1)
        child_scales = updated.scales * (1 - axis_mask * fraction.square()).sqrt()
        child_opacity = -torch.expm1(0.5 * torch.log1p(-updated.opacities.clamp(max=1 - 1e-6)))
        clone_displacement = (rotations @ (updated.scales * self.clone_offset(features).tanh()).unsqueeze(-1)).squeeze(-1)
        parts = []

        def emit(source: Gaussians, gate: Tensor, action: int, parents: Tensor, partners: Tensor | None = None) -> None:
            selected = torch.nonzero(gate > 0).flatten() if hard else torch.arange(gate.numel(), device=gate.device)
            subset = _take(source, selected)
            parts.append(replace(subset, opacities=subset.opacities * gate[selected], extras={
                **subset.extras, "topology_gate": gate[selected],
                "topology_action": torch.full_like(selected, action),
                "topology_parent": parents[selected],
                "topology_partner": torch.full_like(selected, -1) if partners is None else partners[selected],
            }))

        emit(gaussians, keep_gate, KEEP, indices)
        emit(updated, probabilities[:, UPDATE], UPDATE, indices)
        for direction in (-1, 1):
            child = replace(updated, means=updated.means + direction * offset, scales=child_scales, opacities=child_opacity)
            emit(child, probabilities[:, SPLIT], SPLIT, indices)
        emit(replace(gaussians, opacities=-torch.expm1(0.5 * torch.log1p(-gaussians.opacities.clamp(max=1 - 1e-6)))),
             probabilities[:, CLONE], CLONE, indices)
        emit(replace(updated, means=updated.means + clone_displacement, opacities=child_opacity),
             probabilities[:, CLONE], CLONE, indices)
        emit(self._merge(gaussians, pairs, prediction["quality"]), pair_gate, MERGE, left, right)
        values = {name: torch.cat([getattr(part, name) for part in parts]) for name in ATTRIBUTES}
        extras = {name: torch.cat([part.extras[name] for part in parts]) for name in parts[0].extras}
        return replace(gaussians, **values, extras=extras)