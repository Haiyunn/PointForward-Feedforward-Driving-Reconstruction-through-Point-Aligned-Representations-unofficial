"""PointForward: feedforward driving-scene reconstruction (full model).

Wires together the five stages of Fig. 2:

    1. Sec. 3.1  frozen depth + camera geometry  ->  dense point-aligned queries
    2. Sec. 3.2  OmniRe-style scene graph        ->  static / per-instance nodes,
                 dynamic queries canonicalised, then the set is sparsified
    3. Sec. 3.3  spatial-temporal fusion         ->  fused query features + sigma
    4. Sec. 3.4  Gaussian decoding               ->  feature-carrying Gaussians
    5. Sec. 3.4  gsplat rasterisation + UNet     ->  RGB at any target view/time

The model operates on ONE scene per forward call.  Batching is handled by the
training loop, which accumulates gradients over scenes -- the per-scene query
count (V*N_s + O*N_d) is already large enough to saturate a GPU, and each scene
carries its own instance count, so a ragged batch dimension would buy nothing.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .backbone import DEFAULT_BACKBONE, FeatureExtractor
from .fusion import SpatialTemporalFusion, box_dynamic_masks
from .gaussian_head import GaussianDecoder, temporal_opacity
from .query import QUERY_DIM, build_dense_queries, query_xyz, set_query_xyz
from .renderer import FeatureToRGB, render_features
from .rotation import matrix_to_quaternion, quaternion_multiply
from .scene_graph import (
    InstanceBoxes,
    assign_instances,
    canonicalise,
    decanonicalise,
)


@dataclass
class SceneInput:
    """The V context views of one scene.

    Shapes, with V context views at resolution H x W:
        images       (V, 3, H, W)   RGB in [0, 1]
        intrinsics   (V, 3, 3)      pinhole K, in pixels of the given resolution
        cam_to_world (V, 4, 4)
        times        (V,)           timestamps normalised to [0, 1] over the clip
        frame_index  (V,)           long; which box slot is contemporaneous with
                                    each view (several views share a frame in a
                                    multi-camera rig)
        boxes        scene-graph boxes, or None for a fully static scene
        sparse_depth (V, 1, H, W)   optional LiDAR depth, zeros where missing
        sparse_mask  (V, 1, H, W)   optional validity mask for sparse_depth
    """

    images: Tensor
    intrinsics: Tensor
    cam_to_world: Tensor
    times: Tensor
    frame_index: Tensor
    boxes: InstanceBoxes | None = None
    sparse_depth: Tensor | None = None
    sparse_mask: Tensor | None = None

    @property
    def num_views(self) -> int:
        return self.images.shape[0]

    @property
    def image_size(self) -> tuple[int, int]:
        return int(self.images.shape[-2]), int(self.images.shape[-1])


@dataclass
class TargetViews:
    """The views to render.

    Shapes, with T target views:
        intrinsics   (T, 3, 3)
        cam_to_world (T, 4, 4)
        times        (T,)      normalised render timestamps t'
        frame_index  (T,)      long; box slot to pose dynamic instances at
        height/width           render resolution; defaults to the context size
    """

    intrinsics: Tensor
    cam_to_world: Tensor
    times: Tensor
    frame_index: Tensor
    height: int | None = None
    width: int | None = None

    @property
    def num_views(self) -> int:
        return self.intrinsics.shape[0]


@dataclass
class Gaussians:
    """Decoded Gaussians, still in their own reference frames.

    Static Gaussians live in world space; dynamic ones live in the canonical
    space of their instance and are posed at render time.
    """

    means: Tensor  # (N, 3)
    quats: Tensor  # (N, 4)
    scales: Tensor  # (N, 3)
    opacities: Tensor  # (N,)
    colors: Tensor  # (N, N_c)
    sigma: Tensor  # (N,)
    times: Tensor  # (N,) the query's own timestamp t
    instance_id: Tensor  # (N,) -1 for static
    extras: dict[str, Tensor] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return self.means.shape[0]


class PointForward(nn.Module):
    """The full PointForward model.

    Args:
        depth_model:  a frozen depth predictor (see :mod:`.depth`); must map
                      (V, 3, H, W) -> (V, H, W).
        backbone:     timm name of the image backbone.  The paper uses
                      EfficientNetV2-S.
        feat_dim:     channel width of the image features.
        color_dim:    N_c, the Gaussian colour-feature width (128 in the paper).
        hidden:       hidden width of the fusion MLPs.
        num_static:   N_s, static queries sampled per view.
        num_dynamic:  N_d, dynamic queries sampled per instance.
        num_views:    V, the number of context views (frames x cameras).  Needed
                      up front so the fusion module's per-view head exists before
                      the optimiser is built; 0 defers it, which is fine for
                      inference but would leave it untrained.
        depth_scale:  resolution multiplier for the depth field.  InfiniDepth is
                      resolution-free, so depth can be queried above the feature
                      resolution; 1.0 keeps it at the image resolution.
        use_weighted_fusion / use_depth_cue: the two fusion ablations of Table 4.
        depth_view_chunk: maximum views per depth call; 0 keeps the single call.
        backbone_view_chunk: maximum views per backbone call; 0 keeps the legacy
                  path. Positive values use non-reentrant checkpointing in
                  gradient-enabled training, independently of grad_checkpoint.
        stream_queries: build one view's dense queries at a time. Static budget
                  is per view; dynamic budget is per instance across ALL
                  views, using random-key reservoir sampling. Empty pools
                  produce no queries (also on the legacy path).
        freeze_backbone_bn: use stored BatchNorm statistics in the backbone,
                  without freezing affine parameters. Opt-in for long clips;
                  default False retains training BN (per chunk if chunked).
                  Checkpoint recomputation never accumulates BN stats twice.
    """

    def __init__(
        self,
        depth_model: nn.Module,
        backbone: str = DEFAULT_BACKBONE,
        feat_dim: int = 128,
        color_dim: int = 128,
        hidden: int = 128,
        num_static: int = 100_000,
        num_dynamic: int = 10_000,
        num_views: int = 0,
        depth_scale: float = 1.0,
        use_weighted_fusion: bool = True,
        use_depth_cue: bool = True,
        pretrained_backbone: bool = True,
        fusion_chunk: int = 65_536,
        decoder_chunk: int = 262_144,
        grad_checkpoint: bool = True,
        depth_view_chunk: int = 0,
        backbone_view_chunk: int = 0,
        stream_queries: bool = False,
        freeze_backbone_bn: bool = False,
        refiner_view_chunk: int = 0,
    ) -> None:
        super().__init__()
        if depth_view_chunk < 0 or backbone_view_chunk < 0:
            raise ValueError("view chunk sizes must be nonnegative (0 disables chunking)")
        self.num_static = num_static
        self.num_dynamic = num_dynamic
        self.depth_scale = depth_scale
        self.depth_view_chunk = depth_view_chunk
        self.backbone_view_chunk = backbone_view_chunk
        self.stream_queries = stream_queries
        self.freeze_backbone_bn = freeze_backbone_bn
        self.refiner_view_chunk = refiner_view_chunk

        self.depth_model = depth_model
        self.backbone = FeatureExtractor(
            out_channels=feat_dim, backbone=backbone, pretrained=pretrained_backbone
        )
        self.fusion = SpatialTemporalFusion(
            feat_dim=feat_dim,
            hidden=hidden,
            query_dim=QUERY_DIM,
            num_views=num_views,
            use_weighted_fusion=use_weighted_fusion,
            use_depth_cue=use_depth_cue,
            chunk_size=fusion_chunk,
            grad_checkpoint=grad_checkpoint,
        )
        self.decoder = GaussianDecoder(
            in_dim=self.fusion.out_dim,
            color_dim=color_dim,
            hidden=hidden * 2,
            chunk_size=decoder_chunk,
            grad_checkpoint=grad_checkpoint,
        )
        self.refiner = FeatureToRGB(in_channels=color_dim)

    # ------------------------------------------------------------------ stages

    @torch.no_grad()
    def predict_depth(self, scene: SceneInput) -> Tensor:
        """Stage 1a: dense depth from the frozen predictor. Returns (V, H, W)."""
        h, w = scene.image_size
        dh, dw = int(round(h * self.depth_scale)), int(round(w * self.depth_scale))
        chunk = self.depth_view_chunk or scene.num_views
        parts = []
        for start in range(0, scene.num_views, chunk):
            sl = slice(start, start + chunk)
            depth = self.depth_model(
                scene.images[sl],
                height=dh,
                width=dw,
                sparse_depth=None if scene.sparse_depth is None else scene.sparse_depth[sl],
                sparse_mask=None if scene.sparse_mask is None else scene.sparse_mask[sl],
            )
            if (dh, dw) != (h, w):
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
                ).squeeze(1)
            parts.append(depth)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    @contextmanager
    def _backbone_bn_context(self, recompute: bool = False):
        """Restore BN mode/buffers even when checkpoint early-stop raises.

        Recompute uses disposable buffer clones, NOT in-place rollback: autograd
        may have saved the original buffers. Batch statistics and affine gradients
        are unchanged, including for momentum=None; only duplicate running-stat
        updates are discarded. Frozen mode applies in both forward and recompute.
        """
        states = []
        try:
            for module in self.backbone.modules():
                if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                    continue
                buffers = (module.running_mean, module.running_var, module.num_batches_tracked)
                states.append((module, module.training, buffers))
                if self.freeze_backbone_bn:
                    module.training = False
                elif recompute:
                    for name, value in zip(
                        ("running_mean", "running_var", "num_batches_tracked"), buffers
                    ):
                        if value is not None:
                            setattr(module, name, value.clone())
            yield
        finally:
            for module, training, buffers in states:
                module.training = training
                module.running_mean, module.running_var, module.num_batches_tracked = buffers

    def extract_features(self, images: Tensor) -> Tensor:
        """Keep every view in order, retaining only chunk-local trunk activations.

        The full output feature tensor is still needed by the unchanged fusion.
        Checkpointing works even when the input images do not require gradients.
        """
        if self.backbone_view_chunk == 0:
            with self._backbone_bn_context():
                return self.backbone(images)

        def contexts():
            return self._backbone_bn_context(), self._backbone_bn_context(recompute=True)

        parts = []
        for part in images.split(self.backbone_view_chunk, dim=0):
            if self.training and torch.is_grad_enabled():
                features = checkpoint(
                    self.backbone, part, use_reentrant=False, context_fn=contexts
                )
            else:
                with self._backbone_bn_context():
                    features = self.backbone(part)
            parts.append(features)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    @staticmethod
    @torch.no_grad()
    def _draw_query_indices(pool: Tensor, count: int) -> Tensor:
        """Uniform draw; an empty group must never borrow another group's pixel."""
        if count <= 0 or pool.numel() == 0:
            return pool.new_empty((0,))
        if pool.numel() >= count:
            idx = torch.randperm(pool.numel(), device=pool.device)[:count]
        else:
            idx = torch.randint(pool.numel(), (count,), device=pool.device)
        return pool[idx]

    def build_queries(self, scene: SceneInput, depth: Tensor) -> dict[str, Tensor]:
        """Stages 1b + 2: lift pixels to queries, split by scene-graph node, sparsify.

        Returns a dict with the sampled query set:
            queries      (N, 12)  coordinate slice already canonicalised
            world_xyz    (N, 3)   world position at the query's own timestamp
            base_xyz     (N, 3)   frame the Gaussian centre is predicted in
            view_id      (N,)     source view
            instance_id  (N,)     -1 for static
            times        (N,)     query timestamp t
            box_index    (N,)     box slot contemporaneous with the query
        """
        if self.stream_queries:
            return self._build_queries_streaming(scene, depth)

        dense, view_id = build_dense_queries(
            scene.images, depth, scene.intrinsics, scene.cam_to_world, z_depth=True
        )
        box_index = scene.frame_index[view_id]

        boxes = scene.boxes
        if boxes is None or boxes.num_instances == 0:
            instance_id = torch.full_like(view_id, -1)
            num_instances = 0
        else:
            instance_id = assign_instances(query_xyz(dense), boxes, box_index)
            num_instances = boxes.num_instances

        # Same draw order/RNG as the original sampler for nonempty groups; only
        # the unsafe empty-pool fallback to dense index 0 is removed.
        picks = [
            self._draw_query_indices(
                torch.nonzero((instance_id < 0) & (view_id == v)).flatten(), self.num_static
            )
            for v in range(scene.num_views)
        ]
        picks.extend(
            self._draw_query_indices(
                torch.nonzero(instance_id == o).flatten(), self.num_dynamic
            )
            for o in range(num_instances)
        )
        idx = torch.cat(picks)

        return self._finish_queries(scene, dense[idx], view_id[idx], instance_id[idx])

    @torch.no_grad()
    def _build_queries_streaming(self, scene: SceneInput, depth: Tensor) -> dict[str, Tensor]:
        """One dense view + O(instances * num_dynamic) reservoir storage.

        Give EVERY dynamic pixel an independent torch random key. Keeping the
        largest k keys locally then globally is equivalent to global top-k, so
        views contribute in proportion to their pool sizes, not equal per-view
        quotas. Float64 keys make finite-precision ties negligible. Only selected
        12-D queries survive each view; the full dynamic pixel pool never does.
        Fixed torch RNG state reproduces this path (not the legacy sample order).
        """
        boxes = scene.boxes
        num_instances = 0 if boxes is None else boxes.num_instances
        reservoirs: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
        query_parts, view_parts, instance_parts = [], [], []

        for v in range(scene.num_views):
            dense, local_view = build_dense_queries(
                scene.images[v:v + 1], depth[v:v + 1], scene.intrinsics[v:v + 1],
                scene.cam_to_world[v:v + 1], z_depth=True,
            )
            slots = scene.frame_index[v].expand_as(local_view)
            ids = (
                assign_instances(query_xyz(dense), boxes, slots)
                if boxes is not None and num_instances else torch.full_like(local_view, -1)
            )
            static = self._draw_query_indices(torch.nonzero(ids < 0).flatten(), self.num_static)
            query_parts.append(dense[static])
            view_parts.append(torch.full_like(static, v))
            instance_parts.append(torch.full_like(static, -1))

            if self.num_dynamic > 0:
                for o in range(num_instances):
                    pool = torch.nonzero(ids == o).flatten()
                    if pool.numel() == 0:
                        continue
                    keys = torch.rand(pool.numel(), dtype=torch.float64, device=dense.device)
                    keys, selected = keys.topk(min(self.num_dynamic, pool.numel()))
                    candidates = dense[pool[selected]]
                    views = torch.full_like(selected, v)
                    if o in reservoirs:
                        old_keys, old_queries, old_views = reservoirs[o]
                        keys = torch.cat((old_keys, keys))
                        candidates = torch.cat((old_queries, candidates))
                        views = torch.cat((old_views, views))
                        keys, selected = keys.topk(min(self.num_dynamic, keys.numel()))
                        candidates, views = candidates[selected], views[selected]
                    reservoirs[o] = (keys, candidates, views)
            # Do not keep the previous dense view alive while building the next.
            del dense, local_view, slots, ids

        for o in sorted(reservoirs):
            keys, queries, views = reservoirs[o]
            if keys.numel() < self.num_dynamic:
                # The reservoir contains the ENTIRE pool in this case. Match the
                # legacy replacement semantics, without inventing any identity.
                idx = torch.randint(keys.numel(), (self.num_dynamic,), device=keys.device)
                queries, views = queries[idx], views[idx]
            query_parts.append(queries)
            view_parts.append(views)
            instance_parts.append(torch.full_like(views, o))

        return self._finish_queries(
            scene, torch.cat(query_parts), torch.cat(view_parts), torch.cat(instance_parts)
        )

    def _finish_queries(
        self, scene: SceneInput, queries: Tensor, view_id: Tensor, instance_id: Tensor
    ) -> dict[str, Tensor]:
        boxes = scene.boxes
        box_index = scene.frame_index[view_id]
        world_xyz = query_xyz(queries)
        times = scene.times[view_id]

        # Canonicalise dynamic queries (Eq. 2); static queries keep world coords.
        base_xyz = world_xyz
        if boxes is not None and boxes.num_instances > 0:
            base_xyz = self._canonicalise_queries(world_xyz, instance_id, box_index, boxes)

        return {
            "queries": set_query_xyz(queries, base_xyz),
            "world_xyz": world_xyz,
            "base_xyz": base_xyz,
            "view_id": view_id,
            "instance_id": instance_id,
            "times": times,
            "box_index": box_index,
        }

    def _canonicalise_queries(
        self, world_xyz: Tensor, instance_id: Tensor, box_index: Tensor, boxes: InstanceBoxes
    ) -> Tensor:
        """Map dynamic queries into their instance's canonical space."""
        dynamic = instance_id >= 0
        out = world_xyz.clone()
        if not bool(dynamic.any()):
            return out

        inst = instance_id[dynamic]
        slot = box_index[dynamic].clamp(0, boxes.centers.shape[1] - 1)
        can_c, can_r = boxes.canonical()

        out[dynamic] = canonicalise(
            world_xyz[dynamic],
            boxes.centers[inst, slot],
            boxes.rots[inst, slot],
            can_c[inst],
            can_r[inst],
        )
        return out

    def _pose_points(
        self,
        base_xyz: Tensor,
        instance_id: Tensor,
        boxes: InstanceBoxes | None,
        frame: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Place queries in world space at the box poses of the given frames.

        Args:
            base_xyz:    (N, 3) canonical (dynamic) / world (static) coordinates.
            instance_id: (N,)
            boxes:       scene-graph boxes, or None.
            frame:       (K,) long box slots, one per output set.

        Returns:
            points (K, N, 3) world positions.
            valid  (K, N)    False where the instance is absent from that slot.
        """
        k = frame.shape[0]
        n = base_xyz.shape[0]
        points = base_xyz.unsqueeze(0).expand(k, n, 3).clone()
        valid = torch.ones(k, n, dtype=torch.bool, device=base_xyz.device)

        if boxes is None or boxes.num_instances == 0:
            return points, valid

        dynamic = instance_id >= 0
        if not bool(dynamic.any()):
            return points, valid

        inst = instance_id[dynamic]                       # (D,)
        d = inst.shape[0]
        can_c, can_r = boxes.canonical()
        slots = frame.clamp(0, boxes.centers.shape[1] - 1)  # (K,)

        # Gather every (frame, instance) box pair at once.  Looping over frames
        # would cost a GPU sync each, and there is one loop per forward pass.
        inst_k = inst.view(1, d).expand(k, d)
        slot_k = slots.view(k, 1).expand(k, d)
        box_c = boxes.centers[inst_k, slot_k]             # (K, D, 3)
        box_r = boxes.rots[inst_k, slot_k]                # (K, D, 3, 3)

        posed = decanonicalise(
            base_xyz[dynamic].unsqueeze(0).expand(k, d, 3),
            box_c,
            box_r,
            can_c[inst].unsqueeze(0),
            can_r[inst].unsqueeze(0),
        )

        idx = torch.nonzero(dynamic, as_tuple=False).flatten()
        points[:, idx] = posed
        valid[:, idx] = boxes.valid[inst_k, slot_k]
        return points, valid

    def encode_tokens(
        self, scene: SceneInput, depth: Tensor, sampled: dict[str, Tensor],
        features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Stage 3: fuse multi-view evidence into scene tokens and lifespans."""
        h, w = scene.image_size
        if features is None:
            features = self.extract_features(scene.images)

        world_to_cam = torch.linalg.inv(scene.cam_to_world)

        points_per_view, point_valid = self._pose_points(
            sampled["base_xyz"], sampled["instance_id"], scene.boxes, scene.frame_index
        )

        dynamic_masks = None
        if scene.boxes is not None and scene.boxes.num_instances > 0:
            dynamic_masks = box_dynamic_masks(
                scene.boxes, scene.frame_index, scene.intrinsics, world_to_cam, h, w
            )

        return self.fusion(
            queries=sampled["queries"],
            points_per_view=points_per_view,
            features=features,
            depth_maps=depth,
            intrinsics=scene.intrinsics,
            world_to_cam=world_to_cam,
            view_times=scene.times,
            query_times=sampled["times"],
            image_size=(h, w),
            dynamic_masks=dynamic_masks,
            is_static=sampled["instance_id"] < 0,
            point_valid=point_valid,
        )

    def decode_gaussians(
        self, fused: Tensor, sigma: Tensor, sampled: dict[str, Tensor]
    ) -> Gaussians:
        """Stage 4: decode scene tokens into feature-carrying Gaussians."""
        decoded = self.decoder(fused, sampled["base_xyz"])
        return Gaussians(
            means=decoded["means"],
            quats=decoded["quats"],
            scales=decoded["scales"],
            opacities=decoded["opacities"],
            colors=decoded["color"],
            sigma=sigma,
            times=sampled["times"],
            instance_id=sampled["instance_id"],
        )

    def decode(self, scene: SceneInput, depth: Tensor, sampled: dict[str, Tensor]) -> Gaussians:
        """Stages 3 + 4: fuse multi-view evidence, then decode Gaussians."""
        fused, sigma = self.encode_tokens(scene, depth, sampled)
        return self.decode_gaussians(fused, sigma, sampled)

    def pose_gaussians(
        self,
        gaussians: Gaussians,
        boxes: InstanceBoxes | None,
        frame: int,
        target_time: Tensor | float,
    ) -> dict[str, Tensor]:
        """Push Gaussians into world space at one target frame / timestamp.

        Dynamic Gaussians are de-canonicalised, which also rotates their
        orientation; opacity is then attenuated by Eq. (7).
        """
        means = gaussians.means
        quats = gaussians.quats
        opacities = temporal_opacity(
            gaussians.opacities, gaussians.sigma, gaussians.times, target_time
        )

        if boxes is not None and boxes.num_instances > 0:
            dynamic = gaussians.instance_id >= 0
            if bool(dynamic.any()):
                inst = gaussians.instance_id[dynamic]
                slot = min(frame, boxes.centers.shape[1] - 1)
                box_c = boxes.centers[inst, slot]
                box_r = boxes.rots[inst, slot]
                can_c, can_r = boxes.canonical()

                means = means.clone()
                quats = quats.clone()
                opacities = opacities.clone()

                means[dynamic] = decanonicalise(
                    gaussians.means[dynamic], box_c, box_r, can_c[inst], can_r[inst]
                )
                # Column-vector convention: the linear part of the rigid map is
                # R_box^T R_can, and it composes on the left of the orientation.
                rel = torch.einsum("nji,njk->nik", box_r, can_r[inst])
                quats[dynamic] = quaternion_multiply(
                    matrix_to_quaternion(rel), gaussians.quats[dynamic]
                )
                # Instances absent from this frame contribute nothing.
                absent = ~boxes.valid[inst, slot]
                if bool(absent.any()):
                    dyn_idx = torch.nonzero(dynamic, as_tuple=False).flatten()
                    opacities[dyn_idx[absent]] = 0.0

        return {
            "means": means,
            "quats": quats,
            "scales": gaussians.scales,
            "opacities": opacities,
            "colors": gaussians.colors,
        }

    def render(
        self,
        gaussians: Gaussians,
        boxes: InstanceBoxes | None,
        targets: TargetViews,
        height: int,
        width: int,
        render_depth: bool = False,
    ) -> dict[str, Tensor]:
        """Stage 5: rasterise features per target view, then refine to RGB.

        Target views sharing a frame index are rasterised in a single gsplat
        call, since the dynamic Gaussian poses are identical for them.
        """
        t = targets.num_views
        world_to_cam = torch.linalg.inv(targets.cam_to_world)

        feats_out: list[Tensor] = []
        alpha_out: list[Tensor] = []
        depth_out: list[Tensor] = []
        order: list[Tensor] = []

        frames = targets.frame_index.tolist()
        for frame in sorted(set(frames)):
            sel = torch.tensor(
                [i for i, f in enumerate(frames) if f == frame],
                dtype=torch.long,
                device=gaussians.means.device,
            )
            # Views in a frame group share a timestamp by construction.
            posed = self.pose_gaussians(gaussians, boxes, int(frame), targets.times[sel[0]])
            feats, alpha, depth = render_features(
                means=posed["means"],
                quats=posed["quats"],
                scales=posed["scales"],
                opacities=posed["opacities"],
                colors=posed["colors"],
                intrinsics=targets.intrinsics[sel],
                world_to_cam=world_to_cam[sel],
                height=height,
                width=width,
                render_depth=render_depth,
            )
            feats_out.append(feats)
            alpha_out.append(alpha)
            if depth is not None:
                depth_out.append(depth)
            order.append(sel)

        features = torch.cat(feats_out, dim=0)
        alpha = torch.cat(alpha_out, dim=0)

        # Restore the caller's target ordering.  Advanced indexing
        # (features[perm]) triggers a "CUDA error: invalid configuration
        # argument" from PyTorch's generic indexing kernel once the tensor
        # is large enough (observed at (12, 128, 700, 1036), ~1.1e9
        # elements, with torch 2.8.0+cu128) -- index_select along dim 0 is
        # semantically identical for a 1-D index and does not hit this limit.
        perm = torch.argsort(torch.cat(order, dim=0))
        features = torch.index_select(features, 0, perm)
        alpha = torch.index_select(alpha, 0, perm)

        rgb = self._refiner_chunked(features, alpha)
        assert rgb.shape[0] == t
        out = {"rgb": rgb, "features": features, "alpha": alpha}
        if depth_out:
            depth_cat = torch.cat(depth_out, dim=0)
            out["render_depth"] = torch.index_select(depth_cat, 0, perm)
        return out

    def _refiner_chunked(self, features, alpha):
        """Run the refiner UNet per view-chunk. GroupNorm is per-image, so
        chunking over dim 0 is bit-identical; only trades speed for memory."""
        c = self.refiner_view_chunk
        if c <= 0 or features.shape[0] <= c:
            return self.refiner(features, alpha)
        outs = [
            self.refiner(features[i : i + c], alpha[i : i + c])
            for i in range(0, features.shape[0], c)
        ]
        return torch.cat(outs, dim=0)

    # ----------------------------------------------------------------- forward

    def forward(
        self,
        scene: SceneInput,
        targets: TargetViews,
        return_gaussians: bool = False,
        render_depth: bool = False,
    ) -> dict[str, Tensor]:
        """Reconstruct the scene and render the target views.

        Returns a dict with ``rgb`` (T, 3, H, W), ``alpha`` (T, 1, H, W) and
        ``depth`` (V, H, W) -- the *input* depth prior.  ``render_depth=True``
        additionally returns ``render_depth`` (T, 1, H, W), the rasterised
        expected depth that the D-RMSE metric is computed against.
        """
        depth = self.predict_depth(scene)
        sampled = self.build_queries(scene, depth)
        gaussians = self.decode(scene, depth, sampled)

        h, w = scene.image_size
        out = self.render(
            gaussians,
            scene.boxes,
            targets,
            height=targets.height or h,
            width=targets.width or w,
            render_depth=render_depth,
        )
        out["depth"] = depth
        if return_gaussians:
            out["gaussians"] = gaussians
        return out
