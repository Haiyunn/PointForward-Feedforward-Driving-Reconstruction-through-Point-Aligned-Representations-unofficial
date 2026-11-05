"""Visibility-masked rendering consistency for calibrated video sequences."""

import torch
import torch.nn.functional as F
from torch import nn


class TemporalRenderConsistencyLoss(nn.Module):
    def __init__(self, rgb_weight=0.05, depth_weight=0.01,
                 temporal_residual_weight=0.05, downsample=2,
                 alpha_threshold=0.5, occlusion_threshold=0.05):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.depth_weight = depth_weight
        self.temporal_residual_weight = temporal_residual_weight
        self.downsample = downsample
        self.alpha_threshold = alpha_threshold
        self.occlusion_threshold = occlusion_threshold

    def pack(self, output, targets, truth, dynamic, sky):
        height, width = truth.shape[-2:]
        size = (max(1, height // self.downsample), max(1, width // self.downsample))
        resize = lambda value: F.interpolate(value.float(), size=size, mode="nearest")
        intrinsic = targets.intrinsics.float().clone()
        intrinsic[:, 0] *= size[1] / width
        intrinsic[:, 1] *= size[0] / height
        return dict(rgb=resize(output["rgb"]), depth=resize(output["render_depth"]),
                    alpha=resize(output["alpha"]), truth=resize(truth),
                    static=1 - resize(torch.maximum(dynamic, sky)),
                    intrinsic=intrinsic, pose=targets.cam_to_world.float())

    @staticmethod
    def masked_mean(value, mask):
        return (value * mask).sum() / (mask.sum().clamp_min(1) * value.shape[1])

    def pair(self, source, target):
        depth = target["depth"]
        batch, _, height, width = depth.shape
        rows, columns = torch.meshgrid(
            torch.arange(height, device=depth.device),
            torch.arange(width, device=depth.device), indexing="ij",
        )
        pixels = torch.stack((columns, rows, torch.ones_like(rows))).float().flatten(1)
        points = torch.linalg.solve(target["intrinsic"], pixels.expand(batch, -1, -1))
        points = points * depth.flatten(2)
        transform = torch.linalg.solve(source["pose"], target["pose"])
        projected = transform[:, :3, :3] @ points + transform[:, :3, 3:4]
        expected_depth = projected[:, 2:3].reshape(batch, 1, height, width)
        homogeneous = source["intrinsic"] @ projected
        coordinates = homogeneous[:, :2] / homogeneous[:, 2:3].clamp_min(1e-6)
        horizontal = (coordinates[:, 0] + 0.5) * (2 / width) - 1
        vertical = (coordinates[:, 1] + 0.5) * (2 / height) - 1
        grid = torch.stack((horizontal, vertical), -1).reshape(batch, height, width, 2).detach()
        warp = lambda value: F.grid_sample(value, grid, align_corners=False)
        reference_depth = warp(source["depth"].detach())
        with torch.no_grad():
            valid = ((grid.abs() < 1).all(-1).unsqueeze(1)
                     & (expected_depth > 0.1) & (reference_depth > 0.1)
                     & (target["alpha"] > self.alpha_threshold)
                     & (warp(source["alpha"]) > self.alpha_threshold)
                     & (target["static"] > 0.99) & (warp(source["static"]) > 0.99))
            valid &= ((expected_depth - reference_depth).abs()
                      < self.occlusion_threshold * reference_depth.clamp_min(0.1))
        residual = target["rgb"] - target["truth"]
        warped_residual = warp(source["rgb"] - source["truth"])
        rgb = self.masked_mean((residual - warped_residual).abs(), valid)
        geometry = self.masked_mean(
            (expected_depth - reference_depth).abs() / reference_depth.clamp_min(0.1), valid)
        temporal = (residual - (source["rgb"] - source["truth"])).abs().mean()
        return rgb, geometry, temporal, valid.float().mean()

    def forward(self, current, cameras, previous=None):
        count = current["rgb"].shape[0]
        if previous is not None and previous["rgb"].shape[-2:] != current["rgb"].shape[-2:]:
            old_height, old_width = previous["rgb"].shape[-2:]
            height, width = current["rgb"].shape[-2:]
            previous = dict(previous)
            for key in ("rgb", "depth", "alpha", "truth", "static"):
                previous[key] = F.interpolate(previous[key], size=(height, width), mode="nearest")
            previous["intrinsic"] = previous["intrinsic"].clone()
            previous["intrinsic"][:, 0] *= width / old_width
            previous["intrinsic"][:, 1] *= height / old_height
        if count % cameras:
            raise ValueError("Target views must contain complete time-major camera groups")
        groups = [{key: value[start:start + cameras] for key, value in current.items()}
                  for start in range(0, count, cameras)]
        sequence = groups if previous is None else [previous, *groups]
        terms = []
        for source, target in zip(sequence, sequence[1:]):
            terms.append(torch.stack(self.pair(source, target)))
            terms.append(torch.stack(self.pair(target, source)))
        values = torch.stack(terms).mean(0) if terms else current["rgb"].sum() * 0 + torch.zeros(4, device=current["rgb"].device)
        rgb, depth, residual, coverage = values.unbind()
        loss = self.rgb_weight * rgb + self.depth_weight * depth + self.temporal_residual_weight * residual
        return dict(consistency_rgb=rgb, consistency_depth=depth,
                    consistency_residual=residual, consistency_coverage=coverage,
                    consistency_loss=loss), {key: value.detach() for key, value in groups[-1].items()}