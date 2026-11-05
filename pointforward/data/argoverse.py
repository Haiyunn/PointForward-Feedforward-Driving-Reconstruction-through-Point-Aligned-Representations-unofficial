"""Argoverse camera calibration, principal-point crop, and four-column LiDAR."""

import os

import numpy as np
import torch
from PIL import Image

from .drivestudio import WaymoDriveStudioClips


ARGOVERSE_RIGID_CLASSES = (
    "REGULAR_VEHICLE", "LARGE_VEHICLE", "TRUCK", "BUS", "VEHICULAR_TRAILER",
    "MOTORCYCLE", "BICYCLE", "BICYCLIST", "MOTORCYCLIST",
)


class ArgoverseDriveStudioClips(WaymoDriveStudioClips):
    def __init__(self, *args, rigid_classes=ARGOVERSE_RIGID_CLASSES, **kwargs):
        kwargs["apply_opencv2waymo"] = False
        super().__init__(*args, rigid_classes=rigid_classes, **kwargs)

    @staticmethod
    def read_lidar(path):
        if not os.path.exists(path):
            return None
        raw = np.fromfile(path, dtype=np.float32)
        if not raw.size or raw.size % 4:
            return None
        points = raw.reshape(-1, 4)[:, :3]
        return points.astype(np.float64) if np.isfinite(points).all() else None

    def _load_calibrated_image(self, metadata, relative_path, resolution, camera, gray=False):
        height, width = resolution
        intrinsic = metadata["intrinsics"][camera].copy()
        with Image.open(os.path.join(metadata["path"], relative_path)) as image:
            image = image.convert("L" if gray else "RGB")
            original_width, original_height = image.size
            if original_width / original_height > width / height:
                crop_height = original_height
                crop_width = min(original_width, int(round(crop_height * width / height)))
                crop_x = max(0, min(int(round(intrinsic[0, 2] - crop_width / 2)), original_width - crop_width))
                crop_y = 0
            else:
                crop_width = original_width
                crop_height = min(original_height, int(round(crop_width * height / width)))
                crop_x = 0
                crop_y = max(0, min(int(round(intrinsic[1, 2] - crop_height / 2)), original_height - crop_height))
            image = image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
            image = image.resize((width, height), Image.NEAREST if gray else Image.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        intrinsic[0, 2] -= crop_x
        intrinsic[1, 2] -= crop_y
        intrinsic[0, :] *= width / crop_width
        intrinsic[1, :] *= height / crop_height
        tensor = torch.from_numpy(array)
        return (tensor.unsqueeze(0) if gray else tensor.permute(2, 0, 1)), intrinsic

    def _collect(self, meta, frames, slot_of, norm_times, resolution, world_from, lidar, cameras=None):
        images, intrinsics, cam_to_world, times, frame_index = [], [], [], [], []
        dynamic_masks, sky_masks, depths, masks = [], [], [], []
        for frame in frames:
            for camera in cameras if cameras is not None else self.cameras:
                image, intrinsic = self._load_calibrated_image(
                    meta, f"images/{frame:03d}_{camera}.jpg", resolution, camera
                )
                images.append(image)
                intrinsics.append(torch.from_numpy(intrinsic).float())
                cam_to_world.append(torch.from_numpy(self._cam_to_world(meta, frame, camera, world_from)).float())
                times.append(norm_times[slot_of[frame]])
                frame_index.append(slot_of[frame])
                if meta["has_dynamic_mask"]:
                    mask, _ = self._load_calibrated_image(
                        meta, f"dynamic_masks/all/{frame:03d}_{camera}.png", resolution, camera, gray=True
                    )
                    dynamic_masks.append(mask)
                if meta["has_sky_mask"]:
                    mask, _ = self._load_calibrated_image(
                        meta, f"sky_masks/{frame:03d}_{camera}.png", resolution, camera, gray=True
                    )
                    sky_masks.append(mask)
                if lidar and meta["has_lidar"]:
                    depth, mask = self._lidar_depth(meta, frame, camera, intrinsic, resolution)
                    depths.append(depth)
                    masks.append(mask)
        output = {
            "images": torch.stack(images), "intrinsics": torch.stack(intrinsics),
            "cam_to_world": torch.stack(cam_to_world), "times": torch.stack(times),
            "frame_index": torch.tensor(frame_index, dtype=torch.long),
        }
        if dynamic_masks:
            output["dynamic_mask"] = torch.stack(dynamic_masks)
        if sky_masks:
            output["sky_mask"] = torch.stack(sky_masks)
        if depths:
            output["sparse_depth"] = torch.stack(depths)
            output["sparse_mask"] = torch.stack(masks)
        return output