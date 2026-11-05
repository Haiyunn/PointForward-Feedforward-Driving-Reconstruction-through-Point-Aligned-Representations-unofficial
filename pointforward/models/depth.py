"""Frozen depth prior (paper Sec. 3.1, Fig. 2 "Depth Predictor").

PointForward obtains dense depth from a pre-trained InfiniDepth model that stays
*frozen* throughout training so its monocular prior is preserved.  LiDAR is an
optional input: the ``InfiniDepth_DepthSensor`` variant conditions on sparse
depth (``prompt_depth``), while the plain ``InfiniDepth`` variant runs RGB-only
and recovers metric scale via MoGe-2 -- which is exactly the paper's
"w/o LiDAR depth" ablation.

InfiniDepth represents depth as an implicit field queried at continuous 2D
coordinates.  Its ``query_coord`` convention is (y, x) ordering, normalised to
[-1, 1] with pixel centres, shaped (B, N, 2).
"""

from __future__ import annotations

import os
import sys
from typing import Literal

import torch
from torch import Tensor, nn

DepthVariant = Literal["InfiniDepth", "InfiniDepth_DepthSensor"]


def depth_to_disparity(depth: Tensor) -> Tensor:
    """Convert metric depth (metres) to disparity (1/depth), zero where invalid.

    InfiniDepth_DepthSensor's ``WarpMedian``/``_to_depth_disparity`` machinery
    (see ``InfiniDepth/model/model.py``) operates on disparity, not depth --
    confirmed against InfiniDepth's own reference usage in
    ``inference_depth.py``, which calls this exact conversion (there named
    ``depth_to_disparity`` in ``InfiniDepth/utils/io_utils.py``) on both
    ``gt_depth`` and ``prompt_depth`` before calling ``model.inference``.
    Passing raw metric depth instead silently inverts the output scale (a
    real driving-scene median of ~15 m came back as ~0.07 m).
    """

    disparity = depth.clone()
    valid = disparity > 0
    disparity[valid] = 1.0 / disparity[valid]
    return disparity


def dense_query_coord(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Dense (y, x) query grid in [-1, 1], pixel-centre convention.

    Mirrors InfiniDepth's ``make_coord_seq``: for n samples the centres sit at
    -1 + r + 2r*k with r = 1/n.

    Returns:
        (H * W, 2) tensor ordered (y, x), row-major over (H, W).
    """

    def seq(n: int) -> Tensor:
        r = 1.0 / n
        return -1.0 + r + 2.0 * r * torch.arange(n, device=device, dtype=dtype)

    yy, xx = torch.meshgrid(seq(height), seq(width), indexing="ij")
    return torch.stack((yy, xx), dim=-1).reshape(-1, 2)


class FrozenInfiniDepth(nn.Module):
    """Wrapper around a frozen InfiniDepth model.

    Args:
        repo_root:   path to the cloned InfiniDepth repository (added to sys.path).
        checkpoint:  path to the ``.ckpt`` file.
        variant:     which registry entry to build.
        encoder:     DINOv3 encoder name used by InfiniDepth.
        batch_query: chunk size for implicit-field queries.
    """

    def __init__(
        self,
        repo_root: str,
        checkpoint: str,
        variant: DepthVariant = "InfiniDepth_DepthSensor",
        encoder: str = "vitl16",
        batch_query: int = 100_000,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.batch_query = batch_query

        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"InfiniDepth checkpoint not found: {checkpoint}")

        from InfiniDepth.utils.model_utils import build_model

        # InfiniDepth's constructor already moves the model to CUDA and calls
        # eval(); we only need to make the freeze explicit.
        self.model = build_model(variant, model_path=checkpoint, encoder=encoder)
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "FrozenInfiniDepth":
        """Keep the depth model in eval mode regardless of the parent's mode."""
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        images: Tensor,
        height: int | None = None,
        width: int | None = None,
        sparse_depth: Tensor | None = None,
        sparse_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict a dense depth map per view.

        Args:
            images:       (V, 3, H, W) in [0, 1].
            height/width: output resolution; defaults to the input resolution.
                          Because depth is an implicit field, this may exceed the
                          input resolution at no architectural cost.
            sparse_depth: (V, 1, H, W) optional LiDAR depth, zeros where missing.
            sparse_mask:  (V, 1, H, W) optional validity mask for ``sparse_depth``.

        Returns:
            (V, H_out, W_out) dense depth.
        """
        v = images.shape[0]
        out_h = height or images.shape[-2]
        out_w = width or images.shape[-1]

        coord = dense_query_coord(out_h, out_w, images.device, images.dtype)
        coord = coord.unsqueeze(0).expand(v, -1, -1)

        kwargs: dict[str, Tensor] = {}
        if self.variant == "InfiniDepth_DepthSensor" and sparse_depth is not None:
            # InfiniDepth_DepthSensor._prepare_inference hard-requires gt_depth/
            # gt_depth_mask in addition to prompt_depth/prompt_mask -- it falls
            # back to a ground-truth median only when the prompt mask has <=5
            # valid pixels. InfiniDepth's own inference_utils.py aliases
            # gt_depth to the same sparse LiDAR tensor used as the prompt, so
            # we do the same rather than requiring a separate dense target.
            # Both must be disparity, not metric depth -- see depth_to_disparity.
            sparse_disparity = depth_to_disparity(sparse_depth)
            kwargs["prompt_depth"] = sparse_disparity
            kwargs["gt_depth"] = sparse_disparity
            if sparse_mask is not None:
                kwargs["prompt_mask"] = sparse_mask
                kwargs["gt_depth_mask"] = sparse_mask

        out = self.model.inference(images, coord, use_batch_infer=True, **kwargs)
        depth = out[0] if isinstance(out, (tuple, list)) else out
        return depth.reshape(v, out_h, out_w).float()


class ConstantDepth(nn.Module):
    """Stand-in depth source for shape/gradient smoke tests.

    Returns a fixed plane so the rest of the pipeline can be exercised without
    the InfiniDepth checkpoint or a CUDA-only DINOv3 build.
    """

    def __init__(self, value: float = 10.0) -> None:
        super().__init__()
        self.value = value

    @torch.no_grad()
    def forward(
        self,
        images: Tensor,
        height: int | None = None,
        width: int | None = None,
        sparse_depth: Tensor | None = None,
        sparse_mask: Tensor | None = None,
    ) -> Tensor:
        v = images.shape[0]
        out_h = height or images.shape[-2]
        out_w = width or images.shape[-1]
        base = images.new_full((v, out_h, out_w), self.value)
        if sparse_depth is not None and sparse_mask is not None:
            m = sparse_mask.squeeze(1) > 0
            base = torch.where(m, sparse_depth.squeeze(1), base)
        return base
