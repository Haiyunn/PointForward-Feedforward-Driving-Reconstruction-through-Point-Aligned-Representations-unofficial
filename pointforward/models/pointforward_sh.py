"""Spherical-harmonic direct-RGB PointForward variants."""

from __future__ import annotations

import torch
from torch import Tensor

from .fusion_variable import VariableViewSpatialTemporalFusion
from .gaussian_head_sh import GaussianSHDecoder
from .pointforward import Gaussians, PointForward, TargetViews
from .renderer_sh import render_sh
from .scene_graph import InstanceBoxes


class PointForwardSH(PointForward):
    """Fixed-view SH model compatible with checkpoints from PointForward_SH."""

    def __init__(self, *args, sh_degree: int = 3, hidden: int = 128, **kwargs) -> None:
        super().__init__(*args, hidden=hidden, **kwargs)
        self.sh_degree = sh_degree
        self.decoder = GaussianSHDecoder(
            in_dim=self.fusion.out_dim,
            sh_degree=sh_degree,
            hidden=hidden * 2,
            chunk_size=self.decoder.chunk_size,
            grad_checkpoint=self.decoder.grad_checkpoint,
        )
        self.refiner = None

    def decode_gaussians(
        self, fused: Tensor, sigma: Tensor, sampled: dict[str, Tensor]
    ) -> Gaussians:
        decoded = self.decoder(fused, sampled["base_xyz"])
        return Gaussians(
            means=decoded["means"],
            quats=decoded["quats"],
            scales=decoded["scales"],
            opacities=decoded["opacities"],
            colors=decoded["sh"],
            sigma=sigma,
            times=sampled["times"],
            instance_id=sampled["instance_id"],
        )

    def render(
        self,
        gaussians: Gaussians,
        boxes: InstanceBoxes | None,
        targets: TargetViews,
        height: int,
        width: int,
        render_depth: bool = False,
    ) -> dict[str, Tensor]:
        world_to_cam = torch.linalg.inv(targets.cam_to_world)
        rgb_out, alpha_out, depth_out, order = [], [], [], []
        frames = targets.frame_index.tolist()
        for frame in sorted(set(frames)):
            selected = torch.tensor(
                [index for index, value in enumerate(frames) if value == frame],
                dtype=torch.long,
                device=gaussians.means.device,
            )
            posed = self.pose_gaussians(gaussians, boxes, int(frame), targets.times[selected[0]])
            rgb, alpha, depth = render_sh(
                means=posed["means"], quats=posed["quats"], scales=posed["scales"],
                opacities=posed["opacities"], sh_coeffs=posed["colors"],
                intrinsics=targets.intrinsics[selected], world_to_cam=world_to_cam[selected],
                height=height, width=width, sh_degree=self.sh_degree,
                render_depth=render_depth,
            )
            rgb_out.append(rgb)
            alpha_out.append(alpha)
            if depth is not None:
                depth_out.append(depth)
            order.append(selected)
        permutation = torch.argsort(torch.cat(order))
        output = {
            "rgb": torch.index_select(torch.cat(rgb_out), 0, permutation),
            "alpha": torch.index_select(torch.cat(alpha_out), 0, permutation),
        }
        if depth_out:
            output["render_depth"] = torch.index_select(torch.cat(depth_out), 0, permutation)
        return output


class PointForwardVariableViewSH(PointForwardSH):
    """SH model whose lifespan fusion supports arbitrary view counts."""

    def __init__(self, *args, hidden: int = 128, **kwargs) -> None:
        super().__init__(*args, hidden=hidden, **kwargs)
        old = self.fusion
        self.fusion = VariableViewSpatialTemporalFusion(
            feat_dim=old.feat_dim,
            hidden=hidden,
            use_weighted_fusion=old.use_weighted_fusion,
            use_depth_cue=old.use_depth_cue,
            chunk_size=old.chunk_size,
            grad_checkpoint=old.grad_checkpoint,
        )