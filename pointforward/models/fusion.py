"""Spatial-temporal fusion (paper Sec. 3.3).

Each sparse query is projected into every input view, where it samples an image
feature and a depth-consistency cue.  The per-view evidence is then fused with
learned weights into a single feature, together with a lifespan parameter sigma
that controls how long the resulting Gaussian stays alive in time.

Implements paper Eqs. (3)-(6):

    f_{s,i} = phi([phi(f_i); phi(d_i)])                                    (3)
    f_{t,i} = [f_q * f_{s,i}; t == t_i; t],  w_i = Softmax(phi(f_{t,i}))   (4)
    sigma   = phi_v([phi_c(f_{t,1}), ..., phi_c(f_{t,V})]),  f = sum w_i f_{t,i}  (5)
    q'      = phi([phi(q); phi(f); phi(t); phi(sigma)])                    (6)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import sample_features
from .query import QUERY_DIM


def project_points(points: Tensor, intrinsics: Tensor, world_to_cam: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Project world points into every view.

    Args:
        points:       (V, N, 3) world-space points, one set per view (dynamic
                      queries differ per view because of instance motion).
        intrinsics:   (V, 3, 3)
        world_to_cam: (V, 4, 4)

    Returns:
        uv    (V, N, 2) pixel coordinates.
        depth (V, N)    z-buffer depth along the optical axis.
        valid (V, N)    bool, in front of the camera.
    """
    rot = world_to_cam[:, :3, :3]
    trans = world_to_cam[:, :3, 3]
    cam = torch.einsum("vij,vnj->vni", rot, points) + trans.unsqueeze(1)

    z = cam[..., 2]
    valid = z > 1e-3
    safe_z = z.clamp_min(1e-3)

    proj = torch.einsum("vij,vnj->vni", intrinsics, cam / safe_z.unsqueeze(-1))
    return proj[..., :2], z, valid


def box_dynamic_masks(
    boxes,
    box_index: Tensor,
    intrinsics: Tensor,
    world_to_cam: Tensor,
    height: int,
    width: int,
) -> Tensor:
    """Rasterise a per-view dynamic-region mask by projecting all 3D boxes.

    The paper projects every bounding box onto each image plane to mark dynamic
    regions.  We fill the axis-aligned image-space extent of each box's eight
    projected corners, which conservatively covers the instance.

    Vectorised over instances and free of ``.item()`` calls: the obvious
    per-instance loop costs a GPU sync per box per view, which at V=12 and O=24
    is 288 synchronisations in every forward pass.

    Args:
        boxes:        :class:`~.scene_graph.InstanceBoxes` for the scene.
        box_index:    (V,) which box slot is contemporaneous with each view.
        intrinsics:   (V, 3, 3)
        world_to_cam: (V, 4, 4)
        height, width: image size.

    Returns:
        (V, H, W) bool mask, True inside a projected box.
    """
    v = intrinsics.shape[0]
    device = intrinsics.device
    if boxes.num_instances == 0:
        return torch.zeros(v, height, width, dtype=torch.bool, device=device)

    o = boxes.num_instances
    slots = box_index.clamp(0, boxes.centers.shape[1] - 1)

    # Gather the box contemporaneous with each view: (V, O, ...).
    centers = boxes.centers[:, slots].permute(1, 0, 2)
    dims = boxes.dims[:, slots].permute(1, 0, 2)
    rots = boxes.rots[:, slots].permute(1, 0, 2, 3)
    box_valid = boxes.valid[:, slots].permute(1, 0)

    signs = torch.tensor(
        [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ],
        dtype=intrinsics.dtype,
        device=device,
    )  # (8, 3)

    # Box-local corners -> world.  ``rots`` maps world -> box-local, so the
    # inverse (its transpose) takes the local corners back out.
    local = signs.view(1, 1, 8, 3) * (dims * 0.5).unsqueeze(2)      # (V, O, 8, 3)
    corners = torch.einsum("voji,vokj->voki", rots, local) + centers.unsqueeze(2)

    uv, _, in_front = project_points(
        corners.reshape(v, o * 8, 3), intrinsics, world_to_cam
    )
    uv = uv.reshape(v, o, 8, 2)
    vis = in_front.reshape(v, o, 8) & box_valid.unsqueeze(-1)

    # Axis-aligned extent over the *visible* corners.  Boxes with none collapse
    # to lo=+inf, hi=-inf, which selects no pixels at all.
    big = torch.finfo(uv.dtype).max
    lo = torch.where(vis.unsqueeze(-1), uv, torch.full_like(uv, big)).amin(dim=2)
    hi = torch.where(vis.unsqueeze(-1), uv, torch.full_like(uv, -big)).amax(dim=2)

    us = torch.arange(width, device=device, dtype=uv.dtype)
    vs = torch.arange(height, device=device, dtype=uv.dtype)

    # One view at a time: the full (V, O, H, W) intermediate would be large, and
    # per-view is already sync-free.
    mask = torch.zeros(v, height, width, dtype=torch.bool, device=device)
    for i in range(v):
        u_in = (us.view(1, 1, width) >= lo[i, :, 0].view(o, 1, 1)) & (
            us.view(1, 1, width) <= hi[i, :, 0].view(o, 1, 1)
        )
        v_in = (vs.view(1, height, 1) >= lo[i, :, 1].view(o, 1, 1)) & (
            vs.view(1, height, 1) <= hi[i, :, 1].view(o, 1, 1)
        )
        mask[i] = (u_in & v_in).any(dim=0)

    return mask


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    """The paper's generic multi-layer MLP operator phi(.)."""
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(max(layers - 1, 0)):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU(inplace=True)]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class SpatialTemporalFusion(nn.Module):
    """Fuse multi-view, multi-timestamp evidence onto each sparse 3D query.

    Args:
        feat_dim:  channel width C of the image features.
        hidden:    hidden width of the phi MLPs.
        query_dim: dimensionality of the raw query vector (12 in the paper).
        use_weighted_fusion: if False, replace the learned softmax weights by a
            plain mean -- the paper's "w/o weighted fusion" ablation.
        use_depth_cue: if False, drop the depth-difference input -- the paper's
            "w/o depth-aware fusion" ablation.
        num_views: V, the number of context views.  phi_v (Eq. 5) is an MLP over
            the V per-view scalars, so its width depends on V.  Passing it here
            builds phi_v in __init__ -- which matters, because an optimiser is
            normally constructed from ``model.parameters()`` before the first
            forward pass, and a lazily-created module's parameters would never
            be registered with it and so would never train.  Leaving this at 0
            falls back to lazy construction, which is fine for inference.
        chunk_size: queries processed per chunk.  With the paper's settings
            N = V*Ns + O*Nd runs past a million, and the per-view intermediates
            are (V, N, C) -- tens of GB if materialised at once.  Every operation
            here is independent across queries (the softmax reduces over views),
            so chunking is exact, not an approximation.
        grad_checkpoint: recompute each chunk during the backward pass instead of
            storing its activations.  This is what makes the paper's query count
            fit in memory; it costs roughly one extra forward pass.
    """

    def __init__(
        self,
        feat_dim: int = 128,
        hidden: int = 128,
        query_dim: int = QUERY_DIM,
        num_views: int = 0,
        use_weighted_fusion: bool = True,
        use_depth_cue: bool = True,
        chunk_size: int = 65_536,
        grad_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.use_weighted_fusion = use_weighted_fusion
        self.use_depth_cue = use_depth_cue
        self.chunk_size = chunk_size
        self.grad_checkpoint = grad_checkpoint
        self.spatial_only = False

        # Eq. (3): phi(f_i), phi(d_i) then the joint projection.
        self.phi_feat = _mlp(feat_dim, hidden, hidden)
        self.phi_depth = _mlp(1, hidden // 4, hidden // 4)
        joint_in = hidden + (hidden // 4 if use_depth_cue else 0)
        self.phi_spatial = _mlp(joint_in, hidden, feat_dim)

        # Eq. (4): per-view weight from f_{t,i} (dim feat_dim + 2).
        temporal_dim = feat_dim + 2
        self.phi_weight = _mlp(temporal_dim, hidden, 1)

        # Eq. (5): phi_c reduces each view to a scalar, phi_v aggregates them.
        self.phi_c = _mlp(temporal_dim, hidden, 1)

        # Eq. (6): final query update.
        self.phi_query = _mlp(query_dim, hidden, hidden)
        self.phi_fused = _mlp(temporal_dim, hidden, hidden)
        self.phi_time = _mlp(1, hidden // 4, hidden // 4)
        self.phi_sigma = _mlp(1, hidden // 4, hidden // 4)
        self.phi_out = _mlp(hidden + hidden + hidden // 4 + hidden // 4, hidden, hidden)

        self.out_dim = hidden
        self._phi_v_views = 0
        if num_views > 0:
            self._build_sigma_head(num_views)

    def _build_sigma_head(self, num_views: int) -> nn.Module:
        """Create phi_v for a given view count and register it as a submodule."""
        head = _mlp(num_views, max(num_views * 2, 8), 1)
        # Registered only through add_module, never also as a plain attribute:
        # assigning a Module to an attribute would register it a second time and
        # duplicate it in state_dict.
        self.add_module("phi_v", head)
        self._phi_v_views = num_views
        return head

    def _sigma_head(self, num_views: int, device: torch.device, dtype: torch.dtype) -> nn.Module:
        """phi_v maps the V per-view scalars to a single lifespan sigma."""
        head = getattr(self, "phi_v", None)
        if head is None or self._phi_v_views != num_views:
            if head is not None:
                raise ValueError(
                    f"fusion was built for {self._phi_v_views} views but received "
                    f"{num_views}; phi_v (Eq. 5) is view-count specific, so rebuild "
                    "the model with the right num_views rather than swapping it in "
                    "mid-run (its parameters would not be in the optimiser)."
                )
            head = self._build_sigma_head(num_views).to(device=device, dtype=dtype)
        return head

    def _aggregate_sigma(self, per_view_scalar: Tensor, visible: Tensor) -> Tensor:
        """Apply the legacy fixed-view lifespan head without changing its keys."""
        num_views = per_view_scalar.shape[0]
        head = self._sigma_head(num_views, per_view_scalar.device, per_view_scalar.dtype)
        return head(per_view_scalar.permute(1, 0)).squeeze(-1)

    def forward(
        self,
        queries: Tensor,
        points_per_view: Tensor,
        features: Tensor,
        depth_maps: Tensor,
        intrinsics: Tensor,
        world_to_cam: Tensor,
        view_times: Tensor,
        query_times: Tensor,
        image_size: tuple[int, int],
        dynamic_masks: Tensor | None = None,
        is_static: Tensor | None = None,
        point_valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Fuse in chunks over the query dimension; see :meth:`forward_chunk`."""
        n = queries.shape[0]
        # phi_v is built lazily, so materialise it before any checkpointed call:
        # creating parameters inside a checkpointed region would lose their grads.
        self._sigma_head(points_per_view.shape[0], queries.device, queries.dtype)

        chunk = self.chunk_size if self.chunk_size > 0 else n
        if n <= chunk:
            bounds = [(0, n)]
        else:
            bounds = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]

        updated_parts: list[Tensor] = []
        sigma_parts: list[Tensor] = []

        for lo, hi in bounds:
            args = (
                queries[lo:hi],
                points_per_view[:, lo:hi],
                features,
                depth_maps,
                intrinsics,
                world_to_cam,
                view_times,
                query_times[lo:hi],
                image_size,
                dynamic_masks,
                None if is_static is None else is_static[lo:hi],
                None if point_valid is None else point_valid[:, lo:hi],
            )
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                out = checkpoint(self.forward_chunk, *args, use_reentrant=False)
            else:
                out = self.forward_chunk(*args)
            updated_parts.append(out[0])
            sigma_parts.append(out[1])

        if len(bounds) == 1:
            return updated_parts[0], sigma_parts[0]
        return torch.cat(updated_parts, dim=0), torch.cat(sigma_parts, dim=0)

    def forward_chunk(
        self,
        queries: Tensor,
        points_per_view: Tensor,
        features: Tensor,
        depth_maps: Tensor,
        intrinsics: Tensor,
        world_to_cam: Tensor,
        view_times: Tensor,
        query_times: Tensor,
        image_size: tuple[int, int],
        dynamic_masks: Tensor | None = None,
        is_static: Tensor | None = None,
        point_valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run fusion on one chunk of queries.

        Args:
            queries:         (N, query_dim) raw query vectors.
            points_per_view: (V, N, 3) the query's world position as seen by each
                             view (dynamic queries are de-canonicalised per view).
            features:        (V, C, Hf, Wf) image features.
            depth_maps:      (V, H, W) reference depth for the consistency cue.
            intrinsics:      (V, 3, 3)
            world_to_cam:    (V, 4, 4)
            view_times:      (V,) normalised timestamps t_i.
            query_times:     (N,) normalised timestamp t of each query.
            image_size:      (H, W) of the input images.
            dynamic_masks:   (V, H, W) bool; where each view is dynamic.
            is_static:       (N,) bool; True for background queries.
            point_valid:     (V, N) bool; False where the query has no meaningful
                             world position in that view -- e.g. a dynamic query
                             whose instance is absent from that timestamp.

        Returns:
            updated (N, out_dim) the fused query representation q'.
            sigma   (N,) the lifespan parameter.
        """
        height, width = image_size
        v, n = points_per_view.shape[0], points_per_view.shape[1]

        uv, proj_depth, in_front = project_points(points_per_view, intrinsics, world_to_cam)

        # --- sample image features -------------------------------------------------
        feats = sample_features(features, uv, height, width)  # (V, N, C)

        in_bounds = (
            (uv[..., 0] >= 0) & (uv[..., 0] <= width - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= height - 1)
        )
        visible = in_front & in_bounds
        if point_valid is not None:
            visible = visible & point_valid
        if self.spatial_only:
            visible = visible & (view_times[:, None] == query_times[None, :])

        # A static query landing in a dynamic region carries motion-contaminated
        # evidence, so its sampled feature is zeroed (paper Sec. 3.3).
        if dynamic_masks is not None and is_static is not None:
            mask_at_query = _sample_mask(dynamic_masks, uv, height, width)  # (V, N)
            contaminated = mask_at_query & is_static.view(1, n)
            visible = visible & ~contaminated

        feats = feats * visible.unsqueeze(-1).to(feats.dtype)

        # --- depth consistency cue -------------------------------------------------
        ref_depth = _sample_depth(depth_maps, uv, height, width)  # (V, N)
        depth_diff = (proj_depth - ref_depth).abs()
        depth_diff = torch.where(visible, depth_diff, torch.zeros_like(depth_diff))

        # Eq. (3)
        spatial_in = [self.phi_feat(feats)]
        if self.use_depth_cue:
            spatial_in.append(self.phi_depth(depth_diff.unsqueeze(-1)))
        f_s = self.phi_spatial(torch.cat(spatial_in, dim=-1))  # (V, N, C)

        # Eq. (4): f_q is the query's own-view feature, broadcast over views.
        f_q = _own_view_feature(feats, visible)  # (N, C)
        prod = f_q.view(1, n, -1) * f_s

        t_view = view_times.view(v, 1, 1).expand(v, n, 1)
        t_query = query_times.view(1, n, 1).expand(v, n, 1)
        same_time = (t_view == t_query).to(prod.dtype)
        if self.spatial_only:
            same_time = torch.ones_like(same_time)
            t_query = torch.zeros_like(t_query)
        f_t = torch.cat((prod, same_time, t_query), dim=-1)  # (V, N, C + 2)

        logits = self.phi_weight(f_t).squeeze(-1)  # (V, N)
        logits = logits.masked_fill(~visible, float("-inf"))
        # Queries seen by no view fall back to uniform weights.
        unseen = ~visible.any(dim=0)
        logits = torch.where(unseen.view(1, n).expand(v, n), torch.zeros_like(logits), logits)

        if self.use_weighted_fusion:
            weights = torch.softmax(logits, dim=0)
        else:
            valid = visible.to(f_t.dtype)
            weights = valid / valid.sum(dim=0, keepdim=True).clamp_min(1.0)
            weights = torch.where(
                unseen.view(1, n).expand(v, n), torch.full_like(weights, 1.0 / v), weights
            )
        weights = torch.nan_to_num(weights, nan=1.0 / v)

        # Eq. (5)
        fused = (weights.unsqueeze(-1) * f_t).sum(dim=0)  # (N, C + 2)
        per_view_scalar = self.phi_c(f_t).squeeze(-1)  # (V, N)
        if self.spatial_only:
            pooled_scalar = (per_view_scalar * weights).sum(dim=0, keepdim=True)
            per_view_scalar = pooled_scalar.expand(v, -1)
        sigma = F.softplus(self._aggregate_sigma(per_view_scalar, visible))  # (N,)

        # Eq. (6)
        updated = self.phi_out(
            torch.cat(
                (
                    self.phi_query(queries),
                    self.phi_fused(fused),
                    self.phi_time(
                        (torch.zeros_like(query_times) if self.spatial_only else query_times).unsqueeze(-1)
                    ),
                    self.phi_sigma(sigma.unsqueeze(-1)),
                ),
                dim=-1,
            )
        )
        return updated, sigma


def _sample_mask(masks: Tensor, uv: Tensor, height: int, width: int) -> Tensor:
    """Nearest-neighbour lookup of a bool mask at projected pixel coordinates."""
    u = uv[..., 0].round().long().clamp(0, width - 1)
    v_idx = uv[..., 1].round().long().clamp(0, height - 1)
    view_idx = torch.arange(masks.shape[0], device=masks.device).view(-1, 1).expand_as(u)
    return masks[view_idx, v_idx, u]


def _sample_depth(depth_maps: Tensor, uv: Tensor, height: int, width: int) -> Tensor:
    """Bilinearly sample per-view depth at projected pixel coordinates."""
    sampled = sample_features(depth_maps.unsqueeze(1), uv, height, width)  # (V, N, 1)
    return sampled.squeeze(-1)


def _own_view_feature(feats: Tensor, visible: Tensor) -> Tensor:
    """The query's initial associated feature f_q.

    Queries originate from a specific pixel of a specific view, but after
    canonicalisation and sparsification the cleanest definition available at this
    point is the mean over the views that actually observe the query.  This keeps
    f_q well defined for dynamic queries, whose source view may not see them at
    the query timestamp.
    """
    w = visible.unsqueeze(-1).to(feats.dtype)
    total = w.sum(dim=0).clamp_min(1.0)
    return (feats * w).sum(dim=0) / total
