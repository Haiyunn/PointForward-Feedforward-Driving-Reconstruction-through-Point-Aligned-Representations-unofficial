"""Scene-graph dynamic modelling (paper Sec. 3.2).

Organises queries into a static background node and one node per rigid dynamic
instance, following OmniRe-style scene graphs.  Each instance owns a temporally
ordered set of 3D boxes; the box at the earliest timestamp defines the canonical
frame for that instance.

Notation follows the paper: a box b_{i,j} for instance i at box-index j is
parameterised by centre c in R^3, dimensions m in R^3 and rotation R^B in SO(3).
Points are treated as ROW vectors, so a world point p is mapped into the box
local frame by (p - c) @ R^T.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# Index of the canonical box within an instance's temporally ordered box list.
# The paper designates b_{i,1} -- the earliest timestamp -- as the reference.
CANONICAL_BOX_INDEX = 0


@dataclass
class InstanceBoxes:
    """Temporally ordered 3D boxes for the dynamic instances of one scene.

    All tensors are padded to a common instance/box count so that a batch can be
    processed densely; ``valid`` marks the entries that correspond to a real box.

    Shapes, with O instances and B box slots per instance:
        centers  (O, B, 3)      box centres c_{i,j} in world space
        dims     (O, B, 3)      box extents m_{i,j} (full edge lengths, not half)
        rots     (O, B, 3, 3)   rotations R^B_{i,j}, world <- box-local
        times    (O, B)         normalised timestamp of each box
        valid    (O, B)         bool, whether the slot holds a real box
    """

    centers: Tensor
    dims: Tensor
    rots: Tensor
    times: Tensor
    valid: Tensor

    def __post_init__(self) -> None:
        o, b = self.valid.shape
        if self.centers.shape != (o, b, 3):
            raise ValueError(f"centers must be (O, B, 3), got {tuple(self.centers.shape)}")
        if self.dims.shape != (o, b, 3):
            raise ValueError(f"dims must be (O, B, 3), got {tuple(self.dims.shape)}")
        if self.rots.shape != (o, b, 3, 3):
            raise ValueError(f"rots must be (O, B, 3, 3), got {tuple(self.rots.shape)}")
        if self.times.shape != (o, b):
            raise ValueError(f"times must be (O, B), got {tuple(self.times.shape)}")

    @property
    def num_instances(self) -> int:
        return self.valid.shape[0]

    @property
    def device(self) -> torch.device:
        return self.centers.device

    def canonical_index(self) -> Tensor:
        """Slot of the canonical box b_{i,1} for each instance.

        Boxes are laid out on a common frame grid, so an instance that enters the
        scene late has invalid leading slots.  The canonical box is its *earliest
        valid* box, not simply slot 0.

        Returns:
            (O,) long tensor of slot indices.
        """
        # argmax over a bool tensor returns the first True; instances with no
        # valid box fall back to slot 0 and are ignored downstream.
        return torch.argmax(self.valid.long(), dim=1)

    def canonical(self) -> tuple[Tensor, Tensor]:
        """Return the canonical centre c_{i,1} and rotation R^B_{i,1} per instance.

        Returns:
            centers (O, 3), rots (O, 3, 3)
        """
        idx = self.canonical_index()
        rows = torch.arange(self.num_instances, device=self.device)
        return self.centers[rows, idx], self.rots[rows, idx]

    def index_for_time(self, times: Tensor) -> Tensor:
        """Nearest valid box slot for each requested timestamp.

        Args:
            times: (T,) normalised timestamps.

        Returns:
            (O, T) long tensor of slot indices per instance.
        """
        # Push invalid slots far away so they are never nearest.
        dist = (self.times.unsqueeze(-1) - times.view(1, 1, -1)).abs()
        dist = dist.masked_fill(~self.valid.unsqueeze(-1), float("inf"))
        return torch.argmin(dist, dim=1)


def to_box_local(points: Tensor, centers: Tensor, rots: Tensor) -> Tensor:
    """Map world points into box-local coordinates: (p - c) @ R^T.

    Args:
        points:  (..., 3) world-space points.
        centers: (..., 3) box centres, broadcastable against ``points``.
        rots:    (..., 3, 3) box rotations, broadcastable against ``points``.

    Returns:
        (..., 3) box-local coordinates.
    """
    delta = points - centers
    # (p - c) @ R^T  ==  einsum over the row-vector convention.
    return torch.einsum("...j,...ij->...i", delta, rots)


def from_box_local(local: Tensor, centers: Tensor, rots: Tensor) -> Tensor:
    """Inverse of :func:`to_box_local`: local @ R + c."""
    return torch.einsum("...j,...ji->...i", local, rots) + centers


def inside_box(points: Tensor, centers: Tensor, dims: Tensor, rots: Tensor) -> Tensor:
    """Paper Eq. (1): elementwise test |(p - c) R^T| <= m / 2.

    Args:
        points:  (..., 3) world-space points.
        centers: (..., 3) box centres.
        dims:    (..., 3) full box extents.
        rots:    (..., 3, 3) box rotations.

    Returns:
        (...) bool tensor, True where the point lies inside the box.
    """
    local = to_box_local(points, centers, rots)
    return (local.abs() <= dims * 0.5).all(dim=-1)


def canonicalise(points: Tensor, box_c: Tensor, box_r: Tensor, can_c: Tensor, can_r: Tensor) -> Tensor:
    """Paper Eq. (2): p_bar = (p - c_{i,j}) R^{B,T}_{i,j} R^B_{i,1} + c_{i,1}.

    Maps a world point observed under box b_{i,j} into the instance canonical
    space defined by the reference box b_{i,1}.

    Args:
        points: (..., 3) world points.
        box_c:  (..., 3) centre of the observing box c_{i,j}.
        box_r:  (..., 3, 3) rotation of the observing box R^B_{i,j}.
        can_c:  (..., 3) canonical centre c_{i,1}.
        can_r:  (..., 3, 3) canonical rotation R^B_{i,1}.

    Returns:
        (..., 3) canonical-space coordinates.
    """
    local = to_box_local(points, box_c, box_r)
    return from_box_local(local, can_c, can_r)


def decanonicalise(points: Tensor, box_c: Tensor, box_r: Tensor, can_c: Tensor, can_r: Tensor) -> Tensor:
    """Inverse of :func:`canonicalise`.

    Used at render time (Sec. 3.4) to push canonical-space Gaussians back into
    world space at the target timestamp, and during fusion (Sec. 3.3) to project
    dynamic queries into the source views.
    """
    local = to_box_local(points, can_c, can_r)
    return from_box_local(local, box_c, box_r)


def assign_instances(
    points: Tensor,
    boxes: InstanceBoxes,
    box_index: Tensor,
    chunk: int = 262_144,
) -> Tensor:
    """Assign each point to a dynamic instance, or to the static background.

    A query is dynamic if it falls inside *any* box of any instance at the box
    slot selected by ``box_index`` (i.e. the box contemporaneous with the query).
    Points inside no box are static.  When a point falls inside several boxes the
    lowest instance id wins, which keeps the assignment deterministic.

    The dense query set holds V*H*W points, so the (N, O, 3, 3) intermediates are
    evaluated in chunks to keep peak memory bounded.

    Args:
        points:    (N, 3) world-space query coordinates.
        boxes:     scene-graph boxes for this scene.
        box_index: (N,) long tensor, which box slot j to test for each point.
        chunk:     number of points processed at a time.

    Returns:
        (N,) long tensor; -1 for static, else the instance index in [0, O).
    """
    total = points.shape[0]
    if boxes.num_instances == 0:
        return torch.full((total,), -1, dtype=torch.long, device=points.device)
    if total > chunk:
        parts = [
            assign_instances(points[s : s + chunk], boxes, box_index[s : s + chunk], chunk=chunk)
            for s in range(0, total, chunk)
        ]
        return torch.cat(parts, dim=0)

    n = total
    o = boxes.num_instances
    idx = box_index.clamp_min(0).view(n, 1, 1).expand(n, o, 1)

    # Gather the box contemporaneous with each query, for every instance.
    centers = torch.gather(boxes.centers.unsqueeze(0).expand(n, o, -1, 3), 2, idx.unsqueeze(-1).expand(n, o, 1, 3))
    dims = torch.gather(boxes.dims.unsqueeze(0).expand(n, o, -1, 3), 2, idx.unsqueeze(-1).expand(n, o, 1, 3))
    rots = torch.gather(
        boxes.rots.unsqueeze(0).expand(n, o, -1, 3, 3),
        2,
        idx.unsqueeze(-1).unsqueeze(-1).expand(n, o, 1, 3, 3),
    )
    valid = torch.gather(boxes.valid.unsqueeze(0).expand(n, o, -1), 2, idx)

    centers = centers.squeeze(2)  # (N, O, 3)
    dims = dims.squeeze(2)
    rots = rots.squeeze(2)  # (N, O, 3, 3)
    valid = valid.squeeze(2)  # (N, O)

    hit = inside_box(points.unsqueeze(1).expand(n, o, 3), centers, dims, rots) & valid

    # First (lowest-id) instance that contains the point; -1 when none does.
    any_hit = hit.any(dim=1)
    first = torch.argmax(hit.long(), dim=1)
    return torch.where(any_hit, first, torch.full_like(first, -1))


def sample_queries(
    instance_id: Tensor,
    view_id: Tensor,
    num_static_per_view: int,
    num_dynamic_per_instance: int,
    num_views: int,
    num_instances: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sparsify the dense query set (paper Sec. 3.1 / 3.2).

    Draws ``N_s`` static queries per view and ``N_d`` dynamic queries per
    instance, giving N = V * N_s + O * N_d queries in total.  Sampling is without
    replacement where the pool is large enough and with replacement otherwise, so
    the returned count is exact and downstream shapes stay static.

    Args:
        instance_id: (M,) instance assignment from :func:`assign_instances`.
        view_id:     (M,) which input view each dense query came from.
        num_static_per_view:      N_s.
        num_dynamic_per_instance: N_d.
        num_views:     V.
        num_instances: O.
        generator: optional RNG for reproducible sampling.

    Returns:
        (V * N_s + O * N_d,) long tensor of indices into the dense query set.
    """
    device = instance_id.device
    picks: list[Tensor] = []

    def _draw(pool: Tensor, k: int) -> Tensor:
        if k <= 0:
            return pool.new_empty((0,), dtype=torch.long)
        if pool.numel() == 0:
            # Degenerate group (e.g. a fully occluded instance): fall back to
            # index 0 so shapes remain static.  These queries are masked out
            # downstream by their zero-valued features.
            return torch.zeros(k, dtype=torch.long, device=device)
        if pool.numel() >= k:
            perm = torch.randperm(pool.numel(), generator=generator, device=device)[:k]
        else:
            perm = torch.randint(pool.numel(), (k,), generator=generator, device=device)
        return pool[perm]

    static = instance_id < 0
    for v in range(num_views):
        pool = torch.nonzero(static & (view_id == v), as_tuple=False).flatten()
        picks.append(_draw(pool, num_static_per_view))

    for o in range(num_instances):
        pool = torch.nonzero(instance_id == o, as_tuple=False).flatten()
        picks.append(_draw(pool, num_dynamic_per_instance))

    return torch.cat(picks, dim=0) if picks else instance_id.new_empty((0,), dtype=torch.long)
