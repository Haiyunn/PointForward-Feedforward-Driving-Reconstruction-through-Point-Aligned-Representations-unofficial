"""Waymo clip loader (StreetGaussian / StreetCrafter preprocessing layout).

Directory layout of one processed scene::

    <scene>/
        images/{frame:06d}_{cam}.jpg          RGB
        dynamic_mask/{frame:06d}_{cam}.jpg    dynamic-region mask (optional)
        intrinsics/{cam}.txt                  fx fy cx cy k1 k2 p1 p2 k3
        extrinsics/{cam}.txt                  4x4 camera -> ego
        ego_pose/{frame:06d}.txt              4x4 ego -> world
        ego_pose/{frame:06d}_{cam}.txt        4x4 ego -> world, per camera
                                              (rolling-shutter compensated)
        track/track_info.txt                  per-frame 3D boxes, ego frame
        track/track_ids.json                  Waymo string id -> integer id
        timestamps.json                       {"FRAME": {"000000": t, ...}}
        lidar_depth/ or lidar/                sparse depth (optional)

One dataset item is one *clip*: a window of frames from which ``num_context``
frames are used as input and ``num_target`` as supervision.  Every camera in
``cameras`` contributes a view at each selected frame, so the model sees
V = num_context * len(cameras) input views.

Boxes are stored in the ego frame with a yaw-only heading, following the Waymo
convention (length along local x, width along y, height along z).  They are
lifted to world space here so the scene graph can work directly in world
coordinates.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..models.scene_graph import InstanceBoxes
from ..models.pointforward import SceneInput, TargetViews

# Waymo camera ids in the processed layout: 0 FRONT, 1 FRONT_LEFT,
# 2 FRONT_RIGHT, 3 SIDE_LEFT, 4 SIDE_RIGHT.
FRONT_CAMERAS = (0, 1, 2)
ALL_CAMERAS = (0, 1, 2, 3, 4)

RIGID_CLASSES = ("vehicle", "cyclist")


def _read_matrix(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64).reshape(4, 4)


def _read_intrinsics(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (K 3x3, distortion 5,).  Distortion is reported but not applied."""
    vals = np.loadtxt(path, dtype=np.float64).reshape(-1)
    fx, fy, cx, cy = vals[:4]
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = vals[4:9] if vals.shape[0] >= 9 else np.zeros(5)
    return k, dist


def _yaw_matrix(heading: float) -> np.ndarray:
    c, s = math.cos(heading), math.sin(heading)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass
class TrackBox:
    """One annotated 3D box at one frame, in the ego frame."""

    frame: int
    track_id: int
    object_class: str
    dims: np.ndarray  # (3,) length, width, height -- local x, y, z
    center: np.ndarray  # (3,)
    heading: float
    speed: float


def load_track_info(path: str) -> dict[int, list[TrackBox]]:
    """Parse ``track/track_info.txt`` into per-frame box lists."""
    per_frame: dict[int, list[TrackBox]] = {}
    with open(path, "r") as fh:
        header = fh.readline()  # column names
        if not header.startswith("frame_id"):
            raise ValueError(f"unexpected track_info header in {path}: {header!r}")
        for line in fh:
            parts = line.split()
            if len(parts) < 12:
                continue
            frame = int(parts[0])
            box = TrackBox(
                frame=frame,
                track_id=int(parts[1]),
                object_class=parts[2],
                # Column order is height, width, length; reorder to (l, w, h) so
                # the entries line up with the box-local (x, y, z) axes.
                dims=np.array([float(parts[6]), float(parts[5]), float(parts[4])]),
                center=np.array([float(parts[7]), float(parts[8]), float(parts[9])]),
                heading=float(parts[10]),
                speed=float(parts[11]),
            )
            per_frame.setdefault(frame, []).append(box)
    return per_frame


class WaymoClips(Dataset):
    """Clips of Waymo frames in the processed StreetGaussian layout.

    Args:
        root:        directory holding one subdirectory per scene.
        scenes:      scene names to use; defaults to every usable scene found.
        cameras:     camera ids to load.
        num_context: input frames per clip (each contributing len(cameras) views).
        num_target:  supervision frames per clip, drawn from inside the window.
        frame_stride: gap between consecutive context frames.
        resolution:  (H, W) the images are resized to.
        resolutions: optional pool of (H, W) sizes; one is drawn per clip, which
                     is the paper's multi-resolution training schedule.  The
                     choice is per clip, never per view, so a clip stays
                     internally consistent.
        rigid_classes: object classes promoted to scene-graph instance nodes.
        max_instances: cap on instances per clip, keeping the closest boxes.
        include_context_targets: also supervise on the input frames themselves,
                     rather than only on held-out ones.
        seed:        base seed; clips are drawn deterministically per index when
                     ``deterministic`` is set (used for evaluation).
        deterministic: pick the window and targets from the index, not at random.
    """

    def __init__(
        self,
        root: str,
        scenes: list[str] | None = None,
        cameras: tuple[int, ...] = FRONT_CAMERAS,
        num_context: int = 3,
        num_target: int = 2,
        frame_stride: int = 5,
        resolution: tuple[int, int] = (576, 768),
        resolutions: tuple[tuple[int, int], ...] | None = None,
        rigid_classes: tuple[str, ...] = RIGID_CLASSES,
        max_instances: int = 24,
        include_context_targets: bool = True,
        seed: int = 0,
        deterministic: bool = False,
    ) -> None:
        self.root = root
        self.cameras = tuple(cameras)
        self.num_context = num_context
        self.num_target = num_target
        self.frame_stride = frame_stride
        self.resolution = resolution
        self.resolutions = tuple(tuple(r) for r in resolutions) if resolutions else None
        self.rigid_classes = tuple(rigid_classes)
        self.max_instances = max_instances
        self.include_context_targets = include_context_targets
        self.seed = seed
        self.deterministic = deterministic

        names = scenes if scenes is not None else sorted(os.listdir(root))
        self.scenes: list[dict] = []
        for name in names:
            meta = self._scan_scene(os.path.join(root, name))
            if meta is not None:
                meta["name"] = name
                self.scenes.append(meta)

        if not self.scenes:
            raise RuntimeError(f"no usable scenes under {root}")

        # Every valid window start becomes one sample.
        self.index: list[tuple[int, int]] = []
        span = (num_context - 1) * frame_stride
        for si, meta in enumerate(self.scenes):
            for start in range(0, len(meta["frames"]) - span):
                self.index.append((si, start))

    # ------------------------------------------------------------------ scan

    def _scan_scene(self, path: str) -> dict | None:
        """Collect a scene's frame list and calibration, or None if unusable."""
        img_dir = os.path.join(path, "images")
        if not os.path.isdir(img_dir):
            return None

        # A frame is usable only if every requested camera has an image.
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
        if len(frames) < (self.num_context - 1) * self.frame_stride + 1:
            return None

        try:
            intrinsics, extrinsics = {}, {}
            for cam in self.cameras:
                k, _ = _read_intrinsics(os.path.join(path, "intrinsics", f"{cam}.txt"))
                intrinsics[cam] = k
                extrinsics[cam] = _read_matrix(os.path.join(path, "extrinsics", f"{cam}.txt"))
        except (OSError, ValueError):
            return None

        track_path = os.path.join(path, "track", "track_info.txt")
        tracks = load_track_info(track_path) if os.path.exists(track_path) else {}

        ts_path = os.path.join(path, "timestamps.json")
        timestamps: dict[int, float] = {}
        if os.path.exists(ts_path):
            with open(ts_path, "r") as fh:
                raw = json.load(fh).get("FRAME", {})
            timestamps = {int(k): float(v) for k, v in raw.items()}

        return {
            "path": path,
            "frames": frames,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "tracks": tracks,
            "timestamps": timestamps,
            "has_dynamic_mask": os.path.isdir(os.path.join(path, "dynamic_mask")),
        }

    # --------------------------------------------------------------- loading

    def __len__(self) -> int:
        return len(self.index)

    def _cam_to_world(self, meta: dict, frame: int, cam: int) -> np.ndarray:
        """Camera-to-world, preferring the per-camera rolling-shutter ego pose."""
        base = os.path.join(meta["path"], "ego_pose")
        per_cam = os.path.join(base, f"{frame:06d}_{cam}.txt")
        path = per_cam if os.path.exists(per_cam) else os.path.join(base, f"{frame:06d}.txt")
        return _read_matrix(path) @ meta["extrinsics"][cam]

    def _load_image(
        self, meta: dict, frame: int, cam: int, sub: str, resolution: tuple[int, int]
    ) -> tuple[Tensor, float, float]:
        """Load and resize one image; also returns the (x, y) resize scale."""
        from PIL import Image

        path = os.path.join(meta["path"], sub, f"{frame:06d}_{cam}.jpg")
        if not os.path.exists(path):
            path = os.path.join(meta["path"], sub, f"{frame:06d}_{cam}.png")
        with Image.open(path) as im:
            im = im.convert("RGB")
            orig_w, orig_h = im.size
            h, w = resolution
            im = im.resize((w, h), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1), w / orig_w, h / orig_h

    def _build_boxes(self, meta: dict, frames: list[int], ego_poses: dict[int, np.ndarray]) -> InstanceBoxes | None:
        """Lift the tracked boxes of these frames into world-space InstanceBoxes."""
        tracks = meta["tracks"]
        if not tracks:
            return None

        # Instances present in the window, restricted to rigid classes.
        present: dict[int, dict[int, TrackBox]] = {}
        for slot, frame in enumerate(frames):
            for box in tracks.get(frame, []):
                if box.object_class not in self.rigid_classes:
                    continue
                present.setdefault(box.track_id, {})[slot] = box

        if not present:
            return None

        # Keep the instances closest to the ego vehicle when there are many.
        def closeness(item: tuple[int, dict[int, TrackBox]]) -> float:
            return min(float(np.linalg.norm(b.center)) for b in item[1].values())

        chosen = sorted(present.items(), key=closeness)[: self.max_instances]

        o, b = len(chosen), len(frames)
        centers = np.zeros((o, b, 3))
        dims = np.ones((o, b, 3))
        rots = np.tile(np.eye(3), (o, b, 1, 1))
        times = np.zeros((o, b))
        valid = np.zeros((o, b), dtype=bool)

        for i, (_, slots) in enumerate(chosen):
            for slot, box in slots.items():
                ego = ego_poses[frames[slot]]
                # Ego-frame box -> world.
                centers[i, slot] = ego[:3, :3] @ box.center + ego[:3, 3]
                rots[i, slot] = ego[:3, :3] @ _yaw_matrix(box.heading)
                dims[i, slot] = box.dims
                valid[i, slot] = True

        # Scene-graph rotations map world -> box-local (column convention), so
        # store the transpose of the world <- box matrices built above.
        rots = np.swapaxes(rots, -1, -2)

        norm_times = self._normalised_times(meta, frames)
        times[:] = norm_times.numpy()[None, :]

        return InstanceBoxes(
            centers=torch.from_numpy(centers).float(),
            dims=torch.from_numpy(dims).float(),
            rots=torch.from_numpy(rots).float(),
            times=torch.from_numpy(times).float(),
            valid=torch.from_numpy(valid),
        )

    def _normalised_times(self, meta: dict, frames: list[int]) -> Tensor:
        """Timestamps of the window mapped to [0, 1]."""
        stamps = meta["timestamps"]
        if stamps and all(f in stamps for f in frames):
            raw = np.array([stamps[f] for f in frames], dtype=np.float64)
        else:
            raw = np.array(frames, dtype=np.float64)
        span = raw[-1] - raw[0]
        norm = (raw - raw[0]) / span if span > 1e-9 else np.zeros_like(raw)
        return torch.from_numpy(norm).float()

    # ------------------------------------------------------------------ item

    def __getitem__(self, idx: int) -> dict:
        si, start = self.index[idx]
        meta = self.scenes[si]
        all_frames = meta["frames"]

        rng = np.random.default_rng(self.seed + idx if self.deterministic else None)

        span = (self.num_context - 1) * self.frame_stride
        window = all_frames[start : start + span + 1]
        context_frames = [window[i * self.frame_stride] for i in range(self.num_context)]

        # Targets come from inside the window so the render timestamps are
        # interpolations rather than extrapolations.
        pool = [f for f in window if self.include_context_targets or f not in context_frames]
        if not pool:
            pool = context_frames
        take = min(self.num_target, len(pool))
        target_frames = [pool[i] for i in rng.choice(len(pool), size=take, replace=False)]

        frames = sorted(set(context_frames) | set(target_frames))
        slot_of = {f: i for i, f in enumerate(frames)}
        norm_times = self._normalised_times(meta, frames)

        ego_poses = {
            f: _read_matrix(os.path.join(meta["path"], "ego_pose", f"{f:06d}.txt")) for f in frames
        }

        # Multi-resolution training: one size per clip, shared by all its views.
        if self.resolutions:
            resolution = self.resolutions[int(rng.integers(len(self.resolutions)))]
        else:
            resolution = self.resolution

        ctx = self._collect_views(meta, context_frames, slot_of, norm_times, resolution)
        tgt = self._collect_views(meta, target_frames, slot_of, norm_times, resolution)

        boxes = self._build_boxes(meta, frames, ego_poses)

        return {
            "scene": meta["name"],
            "context": ctx,
            "target": tgt,
            "boxes": boxes,
            "frames": torch.tensor(frames, dtype=torch.long),
            "resolution": resolution,
        }

    def _collect_views(
        self,
        meta: dict,
        frames: list[int],
        slot_of: dict[int, int],
        norm_times: Tensor,
        resolution: tuple[int, int],
    ) -> dict:
        """Load every camera of every given frame into a flat view list."""
        images, intrinsics, cam_to_world, times, frame_index, masks = [], [], [], [], [], []

        for frame in frames:
            for cam in self.cameras:
                img, sx, sy = self._load_image(meta, frame, cam, "images", resolution)
                images.append(img)

                k = meta["intrinsics"][cam].copy()
                k[0, :] *= sx  # rescale for the resize
                k[1, :] *= sy
                intrinsics.append(torch.from_numpy(k).float())

                cam_to_world.append(
                    torch.from_numpy(self._cam_to_world(meta, frame, cam)).float()
                )
                times.append(norm_times[slot_of[frame]])
                frame_index.append(slot_of[frame])

                if meta["has_dynamic_mask"]:
                    try:
                        m, _, _ = self._load_image(meta, frame, cam, "dynamic_mask", resolution)
                        masks.append(m[:1])
                    except (OSError, FileNotFoundError):
                        masks.append(torch.zeros(1, *resolution))

        out = {
            "images": torch.stack(images),
            "intrinsics": torch.stack(intrinsics),
            "cam_to_world": torch.stack(cam_to_world),
            "times": torch.stack(times),
            "frame_index": torch.tensor(frame_index, dtype=torch.long),
        }
        if masks:
            out["dynamic_mask"] = torch.stack(masks)
        return out


def collate_clip(batch: list[dict]) -> list[dict]:
    """Identity collate: the model consumes one scene at a time.

    Instance counts and box layouts differ per clip, so stacking would need
    padding that buys nothing -- the training loop accumulates over the list.
    """
    return batch


def to_scene_input(clip: dict, device: torch.device) -> tuple[SceneInput, TargetViews, Tensor]:
    """Convert a dataset item into the model's input dataclasses.

    Returns the context :class:`SceneInput`, the :class:`TargetViews` to render,
    and the ground-truth target images.
    """
    ctx, tgt = clip["context"], clip["target"]
    boxes = clip["boxes"]
    if boxes is not None:
        boxes = InstanceBoxes(
            centers=boxes.centers.to(device),
            dims=boxes.dims.to(device),
            rots=boxes.rots.to(device),
            times=boxes.times.to(device),
            valid=boxes.valid.to(device),
        )

    scene = SceneInput(
        images=ctx["images"].to(device),
        intrinsics=ctx["intrinsics"].to(device),
        cam_to_world=ctx["cam_to_world"].to(device),
        times=ctx["times"].to(device),
        frame_index=ctx["frame_index"].to(device),
        boxes=boxes,
        # Present only for layouts that carry per-frame LiDAR; the depth wrapper
        # forwards them as InfiniDepth's prompt_depth / prompt_mask.
        sparse_depth=ctx["sparse_depth"].to(device) if "sparse_depth" in ctx else None,
        sparse_mask=ctx["sparse_mask"].to(device) if "sparse_mask" in ctx else None,
    )
    targets = TargetViews(
        intrinsics=tgt["intrinsics"].to(device),
        cam_to_world=tgt["cam_to_world"].to(device),
        times=tgt["times"].to(device),
        frame_index=tgt["frame_index"].to(device),
    )
    return scene, targets, tgt["images"].to(device)
