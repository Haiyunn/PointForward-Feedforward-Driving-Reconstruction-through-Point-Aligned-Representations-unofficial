"""Waymo clip loader (DriveStudio / OmniRe preprocessing layout).

This is the layout that carries per-frame LiDAR, which is what the paper's main
setting needs -- the depth prior runs as ``InfiniDepth_DepthSensor`` with the
projected sweep as ``prompt_depth``.

Directory layout of one processed scene::

    <scene>/
        images/{frame:03d}_{cam}.jpg
        ego_pose/{frame:03d}.txt              4x4 ego -> world
        extrinsics/{cam}.txt                  4x4 camera -> ego
        intrinsics/{cam}.txt                  fx fy cx cy k1 k2 p1 p2 k3
        lidar/{frame:03d}.bin                 float32 (N, 14) or (N, 10), ego frame
        dynamic_masks/{all,vehicle,human}/{frame:03d}_{cam}.png
        sky_masks/{frame:03d}_{cam}.png
        instances/instances_info.json         per-instance world-space boxes
        instances/frame_instances.json        frame -> visible instance keys
        frame_info.json                       scene-level metadata

Unlike the StreetGaussian layout in :mod:`.waymo`, instance boxes are stored as
full 4x4 object-to-world matrices, so no ego-frame lift is needed.

Waymo world coordinates are UTM-like and run to ~1e4 m, which leaves float32
with only millimetre resolution.  Each clip is therefore re-expressed relative
to the ego pose of its first frame, following DriveStudio's own convention.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..models.scene_graph import InstanceBoxes

# Waymo runs at 10 FPS; the layout stores no timestamps, so they come from the
# frame index.
FRAME_RATE = 10.0

FRONT_CAMERAS = (0, 1, 2)
ALL_CAMERAS = (0, 1, 2, 3, 4)

# DriveStudio capitalises class names.  Rigid instances get scene-graph nodes;
# pedestrians deform and are left to the static branch (the paper models rigid
# instances, following OmniRe's rigid nodes).
RIGID_CLASSES = ("Vehicle", "Cyclist")

# The stored extrinsics are camera-to-ego in *Waymo* camera axes (x front, y left,
# z up).  Every projection here assumes the OpenCV convention (x right, y down,
# z front), so the extrinsics need this right-multiplication -- the same constant
# DriveStudio applies in its own loader.  Without it nothing projects in front of
# the camera, because depth would be read off the sideways axis.
OPENCV2WAYMO = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _read_matrix(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64).reshape(4, 4)


def _read_intrinsics(path: str) -> np.ndarray:
    vals = np.loadtxt(path, dtype=np.float64).reshape(-1)
    fx, fy, cx, cy = vals[:4]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


class WaymoDriveStudioClips(Dataset):
    """Clips of Waymo frames in the DriveStudio processed layout.

    Args:
        root:        directory holding one subdirectory per scene.
        scenes:      scene names to use; defaults to every usable scene found.
        cameras:     camera ids (0 FRONT, 1 FRONT_LEFT, 2 FRONT_RIGHT, ...).
        num_context: input frames per clip.
        num_target:  supervision frames drawn from the clip window.
        frame_stride: gap between context frames; 5 frames = 0.5 s at 10 FPS,
                     matching the paper's 0/0.5/1.0/1.5 s context schedule.
        clip_length: frames spanned by the window; the paper uses 20 (2 s).
        resolution:  (H, W) images are resized to.
        resolutions: optional pool of sizes, one drawn per clip.
        rigid_classes: classes promoted to scene-graph instance nodes.
        max_instances: cap on instances per clip, nearest kept.
        load_lidar:  project the LiDAR sweep into each context view to produce
                     the sparse depth prompt fed to the depth model.
        target_lidar: also project sweeps for the *target* views.  Training
                     never needs this, but D-RMSE scores rendered depth
                     against the target view's own sweep, so evaluation does.
        deterministic: derive the target choice from the index (for evaluation).
    """

    def __init__(
        self,
        root: str,
        scenes: list[str] | None = None,
        cameras: tuple[int, ...] = FRONT_CAMERAS,
        num_context: int = 4,
        num_target: int = 4,
        frame_stride: int = 5,
        clip_length: int = 20,
        resolution: tuple[int, int] = (576, 768),
        resolutions: tuple[tuple[int, int], ...] | None = None,
        rigid_classes: tuple[str, ...] = RIGID_CLASSES,
        max_instances: int = 24,
        load_lidar: bool = True,
        target_lidar: bool = False,
        require_lidar: bool = False,
        require_instances: bool = False,
        seed: int = 0,
        deterministic: bool = False,
        require_contiguous: bool = False,
        apply_opencv2waymo: bool = True,
        clip_stride: int = 1,
        variable_views: bool = False,
        min_cameras: int = 2,
        target_boundary_pairs: bool = False,
    ) -> None:
        self.root = root
        self.cameras = tuple(cameras)
        self.num_context = num_context
        self.num_target = num_target
        self.frame_stride = frame_stride
        self.clip_length = clip_length
        self.resolution = resolution
        self.resolutions = tuple(tuple(r) for r in resolutions) if resolutions else None
        self.rigid_classes = tuple(rigid_classes)
        self.max_instances = max_instances
        self.load_lidar = load_lidar
        self.target_lidar = target_lidar
        self.require_lidar = require_lidar
        self.require_instances = require_instances
        self.seed = seed
        self.deterministic = deterministic
        self.require_contiguous = require_contiguous
        self.apply_opencv2waymo = apply_opencv2waymo
        self.clip_stride = max(int(clip_stride), 1)
        self.variable_views = variable_views
        self.min_cameras = max(int(min_cameras), 1)
        self.target_boundary_pairs = target_boundary_pairs

        names = scenes if scenes is not None else sorted(os.listdir(root))
        self.scenes: list[dict] = []
        for name in names:
            meta = self._scan_scene(os.path.join(root, name))
            if meta is None:
                continue
            if self.require_lidar and not meta["has_lidar"]:
                continue
            if self.require_instances and not meta["instances"]:
                continue
            meta["name"] = name
            self.scenes.append(meta)
        if not self.scenes:
            raise RuntimeError(f"no usable scenes under {root}")

        span = max((num_context - 1) * frame_stride, clip_length - 1)
        self.index: list[tuple[int, int]] = []
        for si, meta in enumerate(self.scenes):
            for start in range(0, len(meta["frames"]) - span, self.clip_stride):
                if require_contiguous and meta["frames"][start + span] - meta["frames"][start] != span:
                    continue
                self.index.append((si, start))
        if not self.index:
            raise RuntimeError("no usable clip windows (check camera coverage and frame continuity)")

    # ------------------------------------------------------------------ scan

    def _scan_scene(self, path: str) -> dict | None:
        img_dir = os.path.join(path, "images")
        if not os.path.isdir(img_dir):
            return None

        by_frame: dict[int, set[int]] = {}
        for fname in os.listdir(img_dir):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in (".jpg", ".jpeg", ".png") or "_" not in stem:
                continue
            frame_s, cam_s = stem.rsplit("_", 1)
            try:
                by_frame.setdefault(int(frame_s), set()).add(int(cam_s))
            except ValueError:
                continue

        frames = sorted(f for f, cams in by_frame.items() if set(self.cameras) <= cams)
        span = max((self.num_context - 1) * self.frame_stride, self.clip_length - 1)
        if len(frames) < span + 1:
            return None

        try:
            intrinsics, extrinsics = {}, {}
            for cam in self.cameras:
                intrinsics[cam] = _read_intrinsics(os.path.join(path, "intrinsics", f"{cam}.txt"))
                extrinsic = _read_matrix(os.path.join(path, "extrinsics", f"{cam}.txt"))
                extrinsics[cam] = (
                    extrinsic @ OPENCV2WAYMO if self.apply_opencv2waymo else extrinsic
                )
        except (OSError, ValueError):
            return None

        instances = self._load_instances(path)

        return {
            "path": path,
            "frames": frames,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "instances": instances,
            "has_dynamic_mask": os.path.isdir(os.path.join(path, "dynamic_masks", "all")),
            "has_sky_mask": os.path.isdir(os.path.join(path, "sky_masks")),
            # A lidar directory is not enough: some scenes ship placeholder
            # files, so probe the first sweep before claiming lidar support.
            "has_lidar": self.read_lidar(os.path.join(path, "lidar", f"{frames[0]:03d}.bin"))
            is not None,
        }

    def _load_instances(self, path: str) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
        """Read instances_info.json into {instance -> {frame -> (obj_to_world, size)}}."""
        info_path = os.path.join(path, "instances", "instances_info.json")
        if not os.path.exists(info_path):
            return {}
        with open(info_path, "r") as fh:
            raw = json.load(fh)

        out: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        for key, entry in raw.items():
            if entry.get("class_name") not in self.rigid_classes:
                continue
            ann = entry.get("frame_annotations", {})
            idxs = ann.get("frame_idx", [])
            poses = ann.get("obj_to_world", [])
            sizes = ann.get("box_size", [])
            per_frame = {
                int(f): (np.asarray(p, dtype=np.float64).reshape(4, 4), np.asarray(s, dtype=np.float64))
                for f, p, s in zip(idxs, poses, sizes)
            }
            if per_frame:
                out[int(key)] = per_frame
        return out

    # --------------------------------------------------------------- loading

    def __len__(self) -> int:
        return len(self.index)

    def _ego_pose(self, meta: dict, frame: int) -> np.ndarray:
        return _read_matrix(os.path.join(meta["path"], "ego_pose", f"{frame:03d}.txt"))

    def _cam_to_world(self, meta: dict, frame: int, cam: int, world_from: np.ndarray) -> np.ndarray:
        return world_from @ self._ego_pose(meta, frame) @ meta["extrinsics"][cam]

    def _load_image(
        self, meta: dict, rel: str, resolution: tuple[int, int], gray: bool = False
    ) -> tuple[Tensor, float, float]:
        from PIL import Image

        path = os.path.join(meta["path"], rel)
        with Image.open(path) as im:
            im = im.convert("L" if gray else "RGB")
            orig_w, orig_h = im.size
            h, w = resolution
            im = im.resize((w, h), Image.NEAREST if gray else Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr)
        t = t.unsqueeze(0) if gray else t.permute(2, 0, 1)
        return t, w / orig_w, h / orig_h

    @staticmethod
    def read_lidar(path: str) -> np.ndarray | None:
        """Read one sweep and return its (N, 3) ego-frame points, or None.

        Three column layouts occur in the wild, all float32::

            Nx14  origins(3) points(3) flows(3) flow_class ground intensity
                  elongation laser_id
            Nx10  origins(3) points(3) ground intensity elongation laser_id
                  (the same thing without flow)
            Nx4   points(3) intensity (Argoverse)

        Some scenes ship a placeholder file of uniform noise instead of a real
        sweep; those are rejected by checking that laser_id is a small integer
        set and ground is binary, which noise cannot satisfy.
        """
        if not os.path.exists(path):
            return None
        raw = np.fromfile(path, dtype=np.float32)
        for cols in (14, 10):
            if raw.size % cols:
                continue
            a = raw.reshape(-1, cols)
            laser = np.unique(a[:, cols - 1])
            ground = np.unique(a[:, cols - 4])
            if len(laser) > 5 or not np.all(np.isin(laser, (0, 1, 2, 3, 4))):
                continue
            if not np.all(np.isin(ground, (0, 1))):
                continue
            return a[:, 3:6].astype(np.float64)
        if raw.size % 4 == 0:
            return raw.reshape(-1, 4)[:, :3].astype(np.float64)
        return None

    def _lidar_depth(
        self, meta: dict, frame: int, cam: int, k: np.ndarray, resolution: tuple[int, int]
    ) -> tuple[Tensor, Tensor]:
        """Project the frame's LiDAR sweep into one camera.

        Returns (depth (1, H, W), mask (1, H, W)); zeros where no point lands.
        Keeps the nearest point per pixel so occluded returns do not overwrite
        foreground ones.
        """
        h, w = resolution
        depth = torch.zeros(1, h, w)
        mask = torch.zeros(1, h, w)

        pts = self.read_lidar(os.path.join(meta["path"], "lidar", f"{frame:03d}.bin"))
        if pts is None or len(pts) == 0:
            return depth, mask

        # LiDAR shares the ego frame in Waymo; move it into the camera.
        cam_from_ego = np.linalg.inv(meta["extrinsics"][cam])
        cam_pts = pts @ cam_from_ego[:3, :3].T + cam_from_ego[:3, 3]

        z = cam_pts[:, 2]
        front = z > 1e-3
        if not front.any():
            return depth, mask
        cam_pts, z = cam_pts[front], z[front]

        uv = (cam_pts / z[:, None]) @ k.T
        u = np.round(uv[:, 0]).astype(np.int64)
        v = np.round(uv[:, 1]).astype(np.int64)
        keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not keep.any():
            return depth, mask
        u, v, z = u[keep], v[keep], z[keep]

        # Nearest-wins z-buffer: sort far-to-near so near values land last.
        order = np.argsort(-z)
        u, v, z = u[order], v[order], z[order]
        depth[0, v, u] = torch.from_numpy(z).float()
        mask[0, v, u] = 1.0
        return depth, mask

    def _build_boxes(
        self, meta: dict, frames: list[int], world_from: np.ndarray
    ) -> InstanceBoxes | None:
        """Assemble world-space InstanceBoxes for the instances in this window."""
        instances = meta["instances"]
        if not instances:
            return None

        present: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        for slot, frame in enumerate(frames):
            for inst, per_frame in instances.items():
                if frame in per_frame:
                    present.setdefault(inst, {})[slot] = per_frame[frame]
        if not present:
            return None

        # Distance to the ego vehicle, which after recentring sits at the origin.
        origin = self._ego_pose(meta, frames[0])[:3, 3]

        def closeness(item: tuple[int, dict[int, tuple[np.ndarray, np.ndarray]]]) -> float:
            return min(float(np.linalg.norm(p[:3, 3] - origin)) for p, _ in item[1].values())

        chosen = sorted(present.items(), key=closeness)[: self.max_instances]

        o, b = len(chosen), len(frames)
        centers = np.zeros((o, b, 3))
        dims = np.ones((o, b, 3))
        rots = np.tile(np.eye(3), (o, b, 1, 1))
        valid = np.zeros((o, b), dtype=bool)

        for i, (_, slots) in enumerate(chosen):
            for slot, (raw_pose, size) in slots.items():
                obj_to_world = world_from @ raw_pose
                centers[i, slot] = obj_to_world[:3, 3]
                # Stored rotations map world -> box-local (column convention),
                # i.e. the transpose of the object-to-world rotation.
                rots[i, slot] = obj_to_world[:3, :3].T
                dims[i, slot] = size  # (length, width, height) == local (x, y, z)
                valid[i, slot] = True

        times = self._normalised_times(frames).numpy()[None, :].repeat(o, axis=0)

        return InstanceBoxes(
            centers=torch.from_numpy(centers).float(),
            dims=torch.from_numpy(dims).float(),
            rots=torch.from_numpy(rots).float(),
            times=torch.from_numpy(times).float(),
            valid=torch.from_numpy(valid),
        )

    def _normalised_times(self, frames: list[int]) -> Tensor:
        raw = np.asarray(frames, dtype=np.float64) / FRAME_RATE
        span = raw[-1] - raw[0]
        norm = (raw - raw[0]) / span if span > 1e-9 else np.zeros_like(raw)
        return torch.from_numpy(norm).float()

    # ------------------------------------------------------------------ item

    def __getitem__(self, idx: int) -> dict:
        si, start = self.index[idx]
        meta = self.scenes[si]
        rng = np.random.default_rng(self.seed + idx if self.deterministic else None)
        return self._build_window(meta, start, rng)

    def _build_window(
        self, meta: dict, start: int, rng: np.random.Generator,
        world_from: np.ndarray | None = None,
    ) -> dict:
        all_frames = meta["frames"]
        if self.variable_views and not self.deterministic and len(self.cameras) > self.min_cameras:
            count = int(rng.integers(self.min_cameras, len(self.cameras) + 1))
            indices = sorted(int(i) for i in rng.choice(len(self.cameras), size=count, replace=False))
            cameras = tuple(self.cameras[i] for i in indices)
        else:
            cameras = self.cameras

        window = all_frames[start : start + self.clip_length]
        # Context at 0, 0.5, 1.0, 1.5 s -- clamped to the window we actually have.
        context_frames = [
            window[min(i * self.frame_stride, len(window) - 1)] for i in range(self.num_context)
        ]
        context_frames = sorted(set(context_frames))

        pool = [f for f in window if f not in context_frames] or list(context_frames)
        take = min(self.num_target, len(pool))
        target_frames = sorted(pool[i] for i in rng.choice(len(pool), size=take, replace=False))

        if self.target_boundary_pairs:
            boundaries = [frame for frame in context_frames if frame > window[0]]
            if self.num_target != 4 or not boundaries:
                raise ValueError("Boundary supervision requires four targets and multiple context times")
            boundary = int(rng.choice(boundaries))
            before = window[window.index(boundary) - 1]
            target_frames = sorted(set((window[0], before, boundary, window[-1])))

        frames = sorted(set(context_frames) | set(target_frames))
        slot_of = {f: i for i, f in enumerate(frames)}
        norm_times = self._normalised_times(frames)
        if self.target_boundary_pairs:
            norm_times = self._normalised_times(window)[[window.index(frame) for frame in frames]]

        resolution = (
            self.resolutions[int(rng.integers(len(self.resolutions)))]
            if self.resolutions
            else self.resolution
        )

        # Re-express the clip relative to its first ego pose so world coordinates
        # stay small enough for float32 to resolve.
        if world_from is None:
            world_from = np.linalg.inv(self._ego_pose(meta, frames[0]))

        ctx = self._collect(
            meta, context_frames, slot_of, norm_times, resolution, world_from,
            lidar=self.load_lidar, cameras=cameras,
        )
        tgt = self._collect(
            meta,
            target_frames,
            slot_of,
            norm_times,
            resolution,
            world_from,
            lidar=self.target_lidar,
            cameras=cameras,
        )

        return {
            "scene": meta["name"],
            "context": ctx,
            "target": tgt,
            "boxes": self._build_boxes(meta, frames, world_from),
            "frames": torch.tensor(frames, dtype=torch.long),
            "resolution": resolution,
        }

    def _collect(
        self,
        meta: dict,
        frames: list[int],
        slot_of: dict[int, int],
        norm_times: Tensor,
        resolution: tuple[int, int],
        world_from: np.ndarray,
        lidar: bool,
        cameras: tuple[int, ...] | None = None,
    ) -> dict:
        images, intrinsics, cam_to_world, times, frame_index = [], [], [], [], []
        dyn, sky, depths, masks = [], [], [], []

        for frame in frames:
            for cam in cameras if cameras is not None else self.cameras:
                img, sx, sy = self._load_image(meta, f"images/{frame:03d}_{cam}.jpg", resolution)
                images.append(img)

                k = meta["intrinsics"][cam].copy()
                k[0, :] *= sx
                k[1, :] *= sy
                intrinsics.append(torch.from_numpy(k).float())

                cam_to_world.append(
                    torch.from_numpy(self._cam_to_world(meta, frame, cam, world_from)).float()
                )
                times.append(norm_times[slot_of[frame]])
                frame_index.append(slot_of[frame])

                if meta["has_dynamic_mask"]:
                    m, _, _ = self._load_image(
                        meta, f"dynamic_masks/all/{frame:03d}_{cam}.png", resolution, gray=True
                    )
                    dyn.append(m)
                if meta["has_sky_mask"]:
                    m, _, _ = self._load_image(
                        meta, f"sky_masks/{frame:03d}_{cam}.png", resolution, gray=True
                    )
                    sky.append(m)
                if lidar and meta["has_lidar"]:
                    d, mk = self._lidar_depth(meta, frame, cam, k, resolution)
                    depths.append(d)
                    masks.append(mk)

        out = {
            "images": torch.stack(images),
            "intrinsics": torch.stack(intrinsics),
            "cam_to_world": torch.stack(cam_to_world),
            "times": torch.stack(times),
            "frame_index": torch.tensor(frame_index, dtype=torch.long),
        }
        if dyn:
            out["dynamic_mask"] = torch.stack(dyn)
        if sky:
            out["sky_mask"] = torch.stack(sky)
        if depths:
            out["sparse_depth"] = torch.stack(depths)
            out["sparse_mask"] = torch.stack(masks)
        return out


class WaymoDriveStudioWindows(WaymoDriveStudioClips):
    """Consecutive clips sharing the first window's world coordinate frame."""

    def __init__(self, *, num_windows: int = 2, **kwargs) -> None:
        self.num_windows = num_windows
        super().__init__(**kwargs)
        span = self.num_windows * self.clip_length - 1
        self.index = []
        for scene_index, meta in enumerate(self.scenes):
            frames = meta["frames"]
            for start in range(0, len(frames) - span, self.clip_stride):
                if self.require_contiguous and frames[start + span] - frames[start] != span:
                    continue
                self.index.append((scene_index, start))
        if not self.index:
            raise RuntimeError(
                f"no usable {num_windows}-window spans; need "
                f"{self.num_windows * self.clip_length} frames per scene"
            )

    def __getitem__(self, idx: int) -> dict:
        scene_index, start = self.index[idx]
        meta = self.scenes[scene_index]
        rng = np.random.default_rng(self.seed + idx if self.deterministic else None)
        world_from = np.linalg.inv(self._ego_pose(meta, meta["frames"][start]))
        windows = [
            self._build_window(meta, start + offset * self.clip_length, rng, world_from)
            for offset in range(self.num_windows)
        ]
        return {"scene": meta["name"], "windows": windows}


def collate_windows(batch: list[dict]) -> list[dict]:
    """Keep variable-size window sequences as an identity-collated list."""
    return batch
