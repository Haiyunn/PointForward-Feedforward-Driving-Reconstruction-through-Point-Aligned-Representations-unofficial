"""Read-only dataset audit and deterministic scene/edit plan; never runs inference.

Run from the project root with ``python -m scripts.select_diverse_scenes``.
Only the output JSON and optional first-frame previews are written. Global integer
instance keys are NOT graph-node slots: map them anew for every 20-frame chunk
using render_vehicle_removal.chosen_instance_keys. Empty removal lists mean skip
editing, never remove all vehicles as a fallback.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw


FPS = 10
CAMERAS = [0, 1, 2]
PREFIXES = [20, 60, 180]
THRESHOLDS = {
    "median_speed_mps": 1.5, "net_displacement_m": 3.0,
    "prefix_observations": 10, "prefix_displacement_m": 1.0,
    "observation_density": 0.9, "maximum_gap_frames": 3,
    "maximum_speed_mps": 55.0, "corridor_distance_m": 2.0,
    "corridor_time_radius_frames": 30, "direction_cosine": 0.8,
    "corridor_support_fraction": 0.8, "visible_prefix_frames": 10,
    "road_mask_frame_support": 0.6, "road_mask_min_frames": 3,
}


def motion_evidence(frames, positions, prefix_frames, fps=FPS):
    """Measure world motion (not camera-relative motion) and track continuity."""
    frames = np.asarray(frames, dtype=int)
    positions = np.asarray(positions, dtype=float)
    if len(frames) < 2 or positions.shape != (len(frames), 3):
        return {"eligible": False, "reasons": ["insufficient_track"]}
    if not np.isfinite(positions).all() or np.any(np.diff(frames) <= 0):
        return {"eligible": False, "reasons": ["invalid_track"]}
    gaps = np.diff(frames)
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) * fps / gaps
    prefix = np.isin(frames, prefix_frames)
    prefix_pos = positions[prefix]
    median = float(np.median(speed))
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    prefix_disp = float(np.linalg.norm(prefix_pos[-1] - prefix_pos[0])) if len(prefix_pos) > 1 else 0.0
    density = float(len(frames) / (frames[-1] - frames[0] + 1))
    checks = {
        "below_median_speed": median >= THRESHOLDS["median_speed_mps"],
        "below_net_displacement": displacement >= THRESHOLDS["net_displacement_m"],
        "insufficient_prefix_observations": int(prefix.sum()) >= THRESHOLDS["prefix_observations"],
        "below_prefix_displacement": prefix_disp >= THRESHOLDS["prefix_displacement_m"],
        "low_track_density": density >= THRESHOLDS["observation_density"],
        "track_gap": int(gaps.max()) <= THRESHOLDS["maximum_gap_frames"],
        "implausible_speed": float(speed.max()) <= THRESHOLDS["maximum_speed_mps"],
        "unstable_speed": float(np.percentile(speed, 95)) <= 2.5 * median + 2.0,
    }
    reasons = [reason for reason, passed in checks.items() if not passed]
    return {"eligible": not reasons, "reasons": reasons,
            "median_speed_mps": median, "net_displacement_m": displacement,
            "prefix_observations": int(prefix.sum()), "prefix_displacement_m": prefix_disp,
            "observation_density": density, "maximum_gap_frames": int(gaps.max()),
            "maximum_speed_mps": float(speed.max()),
            "confidence_status": "continuity_and_kinematics_proxy_no_annotation_score"}


def ego_path_distance(point, frame, ego_frames, ego_positions, velocity=None,
                      time_radius=30):
    """Distance to actual nearby-time swept segments, never an infinite lane line.

    XY distance is used; a direction gate rejects transverse crossing traffic.
    No extrapolation beyond the observed path, and no matching distant revisits.
    """
    ego_frames = np.asarray(ego_frames)
    xy = np.asarray(ego_positions, dtype=float)[:, :2]
    a, b = xy[:-1], xy[1:]
    delta = b - a
    lengths2 = np.sum(delta * delta, axis=1)
    keep = ((np.abs(ego_frames[:-1] - frame) <= time_radius)
            & (np.abs(ego_frames[1:] - frame) <= time_radius) & (lengths2 > 1e-4)
            & (np.diff(ego_frames) == 1))
    if velocity is not None:
        v = np.asarray(velocity, dtype=float)[:2]
        cosine = delta @ v / np.maximum(np.sqrt(lengths2) * np.linalg.norm(v), 1e-12)
        keep &= cosine >= THRESHOLDS["direction_cosine"]
    if not keep.any():
        return {"distance_m": None, "matched_frame": None, "supported": False}
    ids = np.flatnonzero(keep)
    fraction = np.clip(np.sum((np.asarray(point)[:2] - a[keep]) * delta[keep], axis=1)
                       / lengths2[keep], 0, 1)
    nearest = a[keep] + fraction[:, None] * delta[keep]
    distances = np.linalg.norm(nearest - np.asarray(point)[:2], axis=1)
    slot = int(np.argmin(distances))
    distance = float(distances[slot])
    return {"distance_m": distance, "matched_frame": int(ego_frames[ids[slot]]),
            "supported": distance <= THRESHOLDS["corridor_distance_m"]}


def select_global_ids(evidence, chosen_keys, all_vehicle_keys, limit=2):
    """Return global annotation keys, not node slots; never a blanket deletion."""
    if not 0 <= limit <= 2:
        raise ValueError("Only zero to two vehicles may be removed")
    slots = {int(key): slot for slot, key in enumerate(chosen_keys)}
    eligible = [e for e in evidence if e.get("eligible") and e["global_instance_id"] in slots]
    eligible.sort(key=lambda e: (-int(e.get("road_mask", {}).get("validated", False)),
                                 -e.get("corridor", {}).get("support_fraction", 0),
                                 slots[e["global_instance_id"]]))
    limit = min(limit, max(0, len(set(all_vehicle_keys)) - 1))
    return [int(e["global_instance_id"]) for e in eligible[:limit]]


def valid_transform(matrix):
    return (matrix.shape == (4, 4) and np.isfinite(matrix).all()
            and np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
            and np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=0.03)
            and abs(np.linalg.det(matrix[:3, :3]) - 1) < 0.03)


class SceneAudit:
    """Small per-scene caches: headers/poses/sweep verdicts and bounded masks."""

    def __init__(self, dataset, meta):
        self.dataset, self.meta = dataset, meta
        self.path = Path(meta["path"])
        self.poses, self.headers, self.lidar = {}, {}, {}

    def pose(self, frame):
        if frame not in self.poses:
            pose = self.dataset._ego_pose(self.meta, frame)
            if not valid_transform(pose):
                raise ValueError(f"invalid ego pose {frame}")
            self.poses[frame] = pose
        return self.poses[frame]

    def image_size(self, frame, camera):
        key = frame, camera
        if key not in self.headers:
            path = self.path / "images" / f"{frame:03d}_{camera}.jpg"
            with Image.open(path) as image:
                if image.format != "JPEG" or min(image.size) <= 0:
                    raise ValueError(f"invalid JPEG header {path.name}")
                self.headers[key] = image.size
        return self.headers[key]

    def valid_lidar(self, frame):
        if frame in self.lidar:
            return self.lidar[frame]
        path = self.path / "lidar" / f"{frame:03d}.bin"
        valid = False
        if path.is_file() and path.stat().st_size > 0 and path.stat().st_size % 4 == 0:
            raw = np.memmap(path, dtype=np.float32, mode="r")
            # Same field/layout tests and precedence as the loader, additionally
            # reject empty/nonfinite/degenerate XYZ. No sweep copies or images.
            for cols in (14, 10):
                if raw.size % cols:
                    continue
                a = raw.reshape(-1, cols)
                if (np.isin(a[:, -1], (0, 1, 2, 3, 4)).all()
                        and np.isin(a[:, -4], (0, 1)).all()):
                    xyz = a[:, 3:6]
                    valid = bool(np.isfinite(xyz).all() and len(xyz) >= 10
                                 and np.count_nonzero(np.linalg.norm(xyz, axis=1) > 1) >= 10)
                    break
            del raw
        self.lidar[frame] = valid
        return valid

    @lru_cache(maxsize=24)
    def mask(self, frame, camera):
        path = self.path / "road_masks" / f"{frame:03d}_{camera}.png"
        if not path.exists():
            return None
        with Image.open(path) as image:
            arr = np.asarray(image.convert("L"))
        if not np.isin(arr, (0, 255)).all():
            raise ValueError(f"road mask is not binary 0/255: {path.name}")
        return arr

    def window(self, n):
        failures = []
        for start in range(min(19, max(0, len(self.meta["frames"]) - n + 1))):
            frames = self.meta["frames"][start:start + n]
            if any(b != a + 1 for a, b in zip(frames, frames[1:])):
                failures.append({"start_index": start, "reason": "nonconsecutive_frames"})
                continue
            try:
                for camera in CAMERAS:
                    k, ext = self.meta["intrinsics"][camera], self.meta["extrinsics"][camera]
                    if not (np.isfinite(k).all() and k[0, 0] > 0 and k[1, 1] > 0
                            and valid_transform(ext)):
                        raise ValueError(f"invalid calibration camera {camera}")
                for frame in frames:
                    self.pose(frame)
                    for camera in CAMERAS:
                        self.image_size(frame, camera)
                context = [frames[pos + min(i * self.dataset.frame_stride, 19)]
                           for pos in range(0, n, 20) for i in range(self.dataset.num_context)]
                for frame in sorted(set(context)):
                    if not self.valid_lidar(frame):
                        raise ValueError(f"invalid context lidar {frame}")
                if not any(f in track for track in self.meta["instances"].values() for f in frames):
                    raise ValueError("no rigid instances in window")
                return start, frames, {"image_triplets": n, "ego_poses": n,
                                       "context_lidar_frames": sorted(set(context)),
                                       "calibration_cameras": CAMERAS,
                                       "image_validation": "all_JPEG_headers_not_full_pixel_decode",
                                       "earlier_window_failures": failures}
            except (OSError, ValueError) as exc:
                failures.append({"start_index": start, "reason": str(exc)})
        raise ValueError(f"no valid {n}-frame window in starts 0..18: {failures}")


def project_box(audit, frame, camera, pose, size):
    """Amodal geometric visibility, not occlusion-verified visibility."""
    signs = np.array([[-1, -1, -1], [-1, 1, -1], [1, -1, -1], [1, 1, -1],
                      [-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]])
    world = (signs * size / 2) @ pose[:3, :3].T + pose[:3, 3]
    inv = np.linalg.inv(audit.pose(frame) @ audit.meta["extrinsics"][camera])
    xyz = world @ inv[:3, :3].T + inv[:3, 3]
    if np.any(xyz[:, 2] <= 0.5):
        return None
    uv = (xyz / xyz[:, 2:]) @ audit.meta["intrinsics"][camera].T
    w, h = audit.image_size(frame, camera)
    lo, hi = uv[:, :2].min(axis=0), uv[:, :2].max(axis=0)
    center = (lo + hi) / 2
    area = np.maximum(0, np.minimum(hi, [w, h]) - np.maximum(lo, 0)).prod()
    if not (0 <= center[0] < w and 0 <= center[1] < h and area >= 64):
        return None
    return {"bbox": [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])],
            "bottom_uv": uv[:4, :2], "image_size": (w, h)}


def road_mask_support(audit, frame, camera, projected):
    """Probe just below the projected bottom footprint, not occluded car pixels."""
    mask = audit.mask(frame, camera)
    if mask is None:
        return None
    bottom = projected["bottom_uv"]
    left, right = float(bottom[:, 0].min()), float(bottom[:, 0].max())
    base = float(bottom[:, 1].max())
    # Narrow horizontal support excludes the uncertain outer sides of the box.
    xs = np.linspace(left + 0.2 * (right - left), right - 0.2 * (right - left), 5)
    offsets = (2, 5, 9)
    w, h = projected["image_size"]
    mh, mw = mask.shape
    values, pixels = [], []
    for offset in offsets:
        for x in xs:
            y = base + offset
            if 0 <= x < w and 0 <= y < h:
                u, v = min(mw - 1, int(x * mw / w)), min(mh - 1, int(y * mh / h))
                values.append(bool(mask[v, u] == 255))
                pixels.append([u, v])
    fraction = float(np.mean(values)) if values else 0.0
    return {"frame": int(frame), "camera": camera, "points": len(values),
            "foreground_fraction": fraction, "supported": len(values) >= 10 and fraction >= 0.6,
            "mask_pixels": pixels}


def vehicle_evidence(audit, frames, raw):
    from scripts.render_vehicle_removal import chosen_instance_keys

    meta, dataset = audit.meta, audit.dataset
    keys = chosen_instance_keys(dataset, meta, frames[:20])
    ego_positions = np.array([audit.pose(f)[:3, 3] for f in frames])
    evidence = []
    all_vehicle_keys = [int(k) for k, entry in raw.items() if entry.get("class_name") == "Vehicle"]
    mask_present = (audit.path / "road_masks").is_dir()
    for key in all_vehicle_keys:
        item = {"global_instance_id": key, "class_name": "Vehicle", "eligible": False,
                "reasons": [], "first_chunk_local_slot": keys.index(key) if key in keys else None}
        track = meta["instances"].get(key, {})
        observed = [f for f in frames if f in track]
        positions = [track[f][0][:3, 3] for f in observed]
        motion = motion_evidence(observed, positions, frames[:20])
        item["motion"] = motion
        item["reasons"].extend(motion["reasons"])
        if key not in keys:
            item["reasons"].append("not_in_first_chunk_model_node_cap")
        if item["reasons"]:
            evidence.append(item)
            continue
        prefix = [f for f in observed if f in frames[:20]]
        samples, visible, projections = [], [], {}
        for f in prefix:
            i = observed.index(f)
            before, after = max(0, i - 2), min(len(observed) - 1, i + 2)
            velocity = np.asarray(positions[after]) - positions[before]
            corridor = ego_path_distance(track[f][0][:3, 3], f, frames, ego_positions, velocity)
            samples.append({"frame": f, **corridor})
            cams = []
            for camera in CAMERAS:
                box = project_box(audit, f, camera, *track[f])
                if box is not None:
                    projections[f, camera] = box
                    cams.append(camera)
            visible.append({"frame": f, "cameras": cams})
        support = float(np.mean([s["supported"] for s in samples]))
        item["corridor"] = {"status": "actual_ego_swept_path_conservative_proxy_not_road_semantics",
                            "support_fraction": support, "samples": samples,
                            "time_radius_frames": 30, "maximum_distance_m": 2.0,
                            "same_direction_cosine_min": 0.8}
        visible_count = sum(bool(v["cameras"]) for v in visible)
        item["visibility"] = {"status": "projected_amodal_box_not_occlusion_verified",
                              "visible_prefix_frames": visible_count, "samples": visible}
        if support < THRESHOLDS["corridor_support_fraction"]:
            item["reasons"].append("insufficient_time_local_same_direction_corridor_support")
        if visible_count < THRESHOLDS["visible_prefix_frames"]:
            item["reasons"].append("insufficient_front_triad_projected_visibility")
        road = {"present": mask_present, "validated": False,
                "semantic_status": "unverified_source_assume_255_road_0_background",
                "sampling": "2_5_9_pixels_below_projected_bottom_footprint",
                "samples": []}
        if mask_present and not item["reasons"]:
            selected_frames = [prefix[i] for i in np.unique(np.linspace(0, len(prefix) - 1, 6).astype(int))]
            frame_votes = []
            try:
                for f in selected_frames:
                    votes = []
                    for camera in CAMERAS:
                        if (f, camera) not in projections:
                            continue
                        sample = road_mask_support(audit, f, camera, projections[f, camera])
                        if sample is not None:
                            road["samples"].append(sample)
                            votes.append(sample["supported"])
                    frame_votes.append(bool(votes) and any(votes))
                road["supported_frames"] = sum(frame_votes)
                road["tested_frames"] = len(frame_votes)
                road["support_fraction"] = float(np.mean(frame_votes))
                road["validated"] = bool(sum(frame_votes) >= THRESHOLDS["road_mask_min_frames"]
                                          and np.mean(frame_votes) >= THRESHOLDS["road_mask_frame_support"])
            except (OSError, ValueError) as exc:
                road["error"] = str(exc)
            if not road["validated"]:
                item["reasons"].append("road_mask_present_but_support_insufficient_or_missing")
        elif not mask_present:
            road["semantic_status"] = "no_mask_corridor_proxy_only_not_verified_semantic_road"
        item["road_mask"] = road
        item["eligible"] = not item["reasons"]
        evidence.append(item)
    present_vehicle_keys = [key for key in all_vehicle_keys
                            if any(f in meta["instances"].get(key, {}) for f in frames)]
    ids = select_global_ids(evidence, keys, present_vehicle_keys)
    for item in evidence:
        item["selected"] = item["global_instance_id"] in ids
        if item["eligible"] and not item["selected"]:
            item["reasons"].append("preserved_by_two_vehicle_limit_or_no_blanket_deletion")
    mapping = []
    for pos in range(0, len(frames), 20):
        chunk_keys = chosen_instance_keys(dataset, meta, frames[pos:pos + 20])
        mapping.append({"start_index_in_window": pos, "global_keys_by_local_slot": chunk_keys,
                        "remove_local_slots": [i for i, k in enumerate(chunk_keys) if k in ids]})
    return ids, {"status": "selected" if ids else "skipped_no_confident_onroad_moving_vehicle",
                 "id_namespace": "integer_keys_of_instances_info_not_uuid_or_local_node_slot",
                 "thresholds": THRESHOLDS, "candidate_tracks": evidence,
                 "window_vehicle_instance_ids": present_vehicle_keys,
                 "selected_tracks": [e for e in evidence if e.get("selected")],
                 "preserve_instance_ids": [k for k in all_vehicle_keys if k not in ids],
                 "chunk_instance_mapping": mapping,
                 "renderer_rule": "Map global IDs with chosen_instance_keys each chunk; absent/capped IDs are skipped. No all-Vehicle or static-branch fallback."}


def ego_attributes(audit, frames):
    poses = np.array([audit.pose(f) for f in frames])
    positions = poses[:, :3, 3]
    yaw = np.unwrap(np.arctan2(poses[:, 1, 0], poses[:, 0, 0]))
    turn = float(np.degrees(yaw[-1] - yaw[0]))
    yaw_range = float(np.degrees(np.ptp(yaw)))
    travel = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    speed = float(np.median(np.linalg.norm(np.diff(positions, axis=0), axis=1) * FPS))
    if travel < 3:
        label = "stationary_ego"
    elif yaw_range < 12:
        label = "straight"
    elif turn >= 20:
        label = "left_turn"
    elif turn <= -20:
        label = "right_turn"
    else:
        label = "curving_or_mixed"
    return {"ego_motion_layout_proxy": label,
            "layout_semantics": "PROXY_from_ego_poses_not_actual_road_or_intersection_label",
            "ego_yaw_change_deg": turn, "ego_yaw_range_deg": yaw_range,
            "ego_travel_m": travel, "ego_median_speed_mps": speed,
            "ego_speed_bin": "slow" if speed < 3 else "medium" if speed < 10 else "fast"}


def diverse_select(candidates, count, seed):
    if len(candidates) < count:
        raise ValueError(f"Need exactly {count} valid scenes, found only {len(candidates)}")
    rare = [c for c in candidates if str(c["attributes"].get("time_of_day", "")).lower()
            in ("night", "dawn/dusk", "dawn", "dusk")]
    if len(rare) > count:
        raise ValueError(f"{count} slots cannot retain all {len(rare)} night/dawn/dusk scenes")
    rng = np.random.default_rng(seed)
    tie = {c["name"]: float(rng.random()) for c in sorted(candidates, key=lambda c: c["name"])}
    selected = sorted(rare, key=lambda c: (c["attributes"].get("time_of_day") != "Night", c["name"]))
    fields = ("location", "ego_motion_layout_proxy", "ego_speed_bin", "time_of_day", "weather")
    counts = {field: Counter(c["attributes"].get(field, "unknown") for c in selected) for field in fields}
    remaining = [c for c in candidates if c not in selected]
    while len(selected) < count:
        def score(c):
            a = c["attributes"]
            unseen = sum(counts[f][a.get(f, "unknown")] == 0
                         for f in ("location", "ego_motion_layout_proxy", "weather"))
            balance = sum(1 / (1 + counts[f][a.get(f, "unknown")]) for f in fields)
            return unseen, bool(c["remove_instance_ids"]), balance, tie[c["name"]]
        chosen = max(remaining, key=score)
        selected.append(chosen)
        remaining.remove(chosen)
        for field in fields:
            counts[field][chosen["attributes"].get(field, "unknown")] += 1
    return selected


def histogram(scenes, field):
    return dict(sorted(Counter(str(c["attributes"].get(field, "unknown")) for c in scenes).items()))


def save_preview(audit, scene, directory):
    directory.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (960, 270), "#151922")
    draw = ImageDraw.Draw(canvas)
    frame = scene["frames"][0]
    for slot, camera in enumerate(CAMERAS):
        with Image.open(audit.path / "images" / f"{frame:03d}_{camera}.jpg") as source:
            image = source.convert("RGB").resize((320, 214))
        canvas.paste(image, (slot * 320, 36))
        for key in scene["remove_instance_ids"]:
            track = audit.meta["instances"][key]
            if frame not in track:
                continue
            box = project_box(audit, frame, camera, *track[frame])
            if box is None:
                continue
            w, h = box["image_size"]
            x1, y1, x2, y2 = box["bbox"]
            rect = (slot * 320 + max(0, x1) * 320 / w, 36 + max(0, y1) * 214 / h,
                    slot * 320 + min(w - 1, x2) * 320 / w, 36 + min(h - 1, y2) * 214 / h)
            draw.rectangle(rect, outline="#00ff99", width=2)
            draw.text(rect[:2], f"ID {key}", fill="#00ff99")
        draw.text((slot * 320 + 4, 253), f"camera {camera}", fill="white")
    draw.text((8, 8), f"scene {scene['name']} frame {frame} | remove {scene['remove_instance_ids']} | amodal boxes", fill="white")
    path = directory / f"scene{scene['name']}.jpg"
    canvas.save(path, quality=85)
    return str(path)


def create_plan(config_path, count=50, frames=180, seed=3, preview_dir=None):
    """Return the plan without writing it; optional previews are output-only."""
    from pointforward.data.drivestudio import WaymoDriveStudioClips

    if frames != 180:
        raise ValueError("This exact-checkpoint protocol requires --frames 180 (20/60/180 prefixes)")
    if count < 1:
        raise ValueError("--num-scenes must be positive")
    config_path = Path(config_path).absolute()
    cfg = yaml.safe_load(config_path.read_text())
    data = cfg["data"]
    if list(data.get("cameras", CAMERAS)) != CAMERAS or data.get("clip_length", 20) != 20:
        raise ValueError("Protocol requires cameras [0,1,2] and clip_length 20")
    if data.get("layout", "drivestudio") != "drivestudio":
        raise ValueError("Only DriveStudio world-space annotations are supported")
    root = Path(data["root"])
    split_path = Path(data["split_file"])
    if not split_path.is_absolute():
        split_path = config_path.parent.parent / split_path
    split = json.loads(split_path.read_text())
    train, val = set(split["train"]), set(split["val"])
    if train & val:
        raise ValueError("Train/val scene overlap")
    metadata, exclusions = {}, []
    directories = sorted(p for p in root.iterdir() if p.is_dir())
    for path in directories:
        info = path / "frame_info.json"
        if info.exists():
            try:
                value = json.loads(info.read_text())
                if not isinstance(value, dict):
                    raise ValueError("metadata must be an object")
                metadata[path.name] = value
            except (OSError, ValueError) as exc:
                exclusions.append({"name": path.name, "reason": f"metadata: {exc}"})
    names = sorted((train | val) | set(metadata))
    candidates, preview_sources = [], {}
    for name in names:
        try:
            dataset = WaymoDriveStudioClips(
                root=str(root), scenes=[name], cameras=tuple(CAMERAS),
                clip_length=20, num_context=data.get("num_context", 4),
                frame_stride=data.get("frame_stride", 5),
                resolution=tuple(data.get("resolution", (576, 768))),
                max_instances=data.get("max_instances", 24),
                require_lidar=True, require_instances=True, deterministic=True)
            meta = dataset.scenes[0]
            raw = json.loads((Path(meta["path"]) / "instances/instances_info.json").read_text())
            # The loader zip() would silently truncate malformed annotations.
            for key, entry in raw.items():
                int(key)
                ann = entry.get("frame_annotations", {})
                indices = ann.get("frame_idx", [])
                if (len(indices) != len(set(indices)) or
                        len({len(indices), len(ann.get("obj_to_world", [])), len(ann.get("box_size", []))}) != 1):
                    raise ValueError(f"malformed annotation arrays: {key}")
            for key, track in meta["instances"].items():
                for pose, size in track.values():
                    if not valid_transform(pose) or size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
                        raise ValueError(f"invalid instance box: {key}")
            audit = SceneAudit(dataset, meta)
            start, window, validation = audit.window(frames)
            attributes = {**metadata.get(name, {}), **ego_attributes(audit, window),
                          "metadata_available": name in metadata}
            ids, editing = vehicle_evidence(audit, window, raw)
            scene = {"name": name, "start_index": start, "frames": window,
                     "split": "train" if name in train else "val" if name in val else "other",
                     "attributes": attributes, "remove_instance_ids": ids,
                     "vehicle_selection": editing, "validation": validation}
            candidates.append(scene)
            preview_sources[name] = (dataset, meta)
            audit.mask.cache_clear()
            print(f"AUDITED {name}: {attributes.get('time_of_day', 'unknown')} / "
                  f"{attributes['ego_motion_layout_proxy']} / remove={ids}", flush=True)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            exclusions.append({"name": name, "reason": str(exc)})
            print(f"EXCLUDED {name}: {exc}", flush=True)
    selected = diverse_select(candidates, count, seed)
    if preview_dir is not None:
        for scene in selected:
            dataset, meta = preview_sources[scene["name"]]
            scene["preview"] = save_preview(SceneAudit(dataset, meta), scene, Path(preview_dir))
    fields = ("time_of_day", "weather", "location", "ego_motion_layout_proxy", "ego_speed_bin")
    summary = {"scene_directories": len(directories), "metadata_scenes": len(metadata),
               "directories_without_metadata": len(directories) - len(metadata),
               "split_train": len(train), "split_val": len(val), "scanned_candidates": len(names),
               "metadata_extra_scenes": sorted(set(metadata) - train - val),
               "metadata_category_counts": {f: dict(Counter(str(m.get(f, 'unknown')) for m in metadata.values()))
                                            for f in ("time_of_day", "weather", "location")},
               "valid_candidates": len(candidates), "excluded": exclusions,
               "candidate_attributes": {f: histogram(candidates, f) for f in fields},
               "selected_attributes": {f: histogram(selected, f) for f in fields},
               "selected_split": dict(Counter(s["split"] for s in selected)),
               "selected_with_edit": sum(bool(s["remove_instance_ids"]) for s in selected),
               "selected_without_edit": [s["name"] for s in selected if not s["remove_instance_ids"]],
               "candidate_inventory": [{"name": s["name"], "attributes": s["attributes"],
                                         "remove_instance_ids": s["remove_instance_ids"]} for s in candidates]}
    return {"schema_version": 1, "config": str(config_path), "config_snapshot": copy.deepcopy(cfg),
            "dataset_root": str(root), "checkpoint_requirement": {"completed_updates": 20000,
                "filename_label": "ckpt020000", "stored_step_zero_based": 19999,
                "verification": "not_loaded_by_selector_renderer_must_check_step_plus_one_equals_20000"},
            "frames": frames, "fps": FPS, "cameras": CAMERAS, "durations": PREFIXES,
            "duration_units": "frames_prefixes_of_same_180_frame_window", "seed": seed,
            "num_scenes": count, "candidates_summary": summary,
            "selection_policy": "all valid night/dawn first; unseen location/layout-proxy/weather; editing availability; inverse category counts; seeded ties; first valid start in 0..18",
            "limitations": [
                "All observed metadata weather is sunny; no fabricated rain/fog/snow categories.",
                "Metadata is scanned across scene directories; expensive asset audit is restricted to train/val union plus metadata-bearing extras.",
                "Includes training scenes: this is a diverse rendering/audit plan, not a held-out benchmark.",
                "Ego-motion layout and swept-path corridor are conservative proxies, not verified lane, roadway, intersection or road-type semantics.",
                "No provided track confidence or camera occlusion score: continuity, kinematics and amodal projected visibility are proxies.",
                "Road masks have unknown source; 255 is assumed foreground road and 0 background. Mask validation is conditional on that assumption.",
                "Where masks are absent, accepted editing uses an explicitly unverified conservative ego-corridor proxy. Static and rejected vehicles are preserved.",
                "Moving-object evidence is measured over this window and its first20 prefix; later frames can leave view or stop. IDs remain global and fixed across prefixes.",
                "No image full-decode or model inference; all 180 JPEG triplet headers, ego poses, calibration, annotations and all 20-frame-chunk context LiDAR validated.",
                "Only starts 0..18 are considered. No inference/checkpoint deserialization; exact checkpoint must be independently enforced by renderer.",
            ], "scenes": selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-scenes", type=int, default=50)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args()
    out = Path(args.out).absolute()
    try:
        plan = create_plan(args.config, args.num_scenes, args.frames, args.seed,
                           None if args.no_previews else out.parent / "selection_preview")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Selection failed; no plan written: {exc}\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(out), "selected": len(plan["scenes"]),
                      "with_edit": plan["candidates_summary"]["selected_with_edit"],
                      "without_edit": plan["candidates_summary"]["selected_without_edit"]}))


if __name__ == "__main__":
    main()