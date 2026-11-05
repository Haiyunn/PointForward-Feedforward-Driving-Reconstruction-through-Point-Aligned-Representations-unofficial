"""Dynamic-only vs. full-image PSNR/SSIM/D-RMSE, plus a tagged GT|pred MP4
per sampled clip, for visual coverage inspection across scene conditions.

Not part of the paper's plain quantitative pipeline (``scripts/eval.py``,
left untouched) -- this reuses its ``build_model``/``build_dataset`` helpers,
the same pattern ``scripts/sample_images.py`` and ``scripts/render_continuous.py``
already follow, to additionally report the masked "dynamic-only" variant of
Table 1's interp-protocol metrics (PSNR/SSIM restricted to
``dynamic_mask``-covered pixels; D-RMSE restricted to LiDAR returns that also
fall on a dynamic pixel) and render a labelled sample gallery.

Each sampled clip's target frames (camera 0, GT on top / prediction on
bottom) are encoded to a short MP4 named with that clip's scene conditions --
weather, time of day, ego motion, turning, other-vehicle motion, pedestrian
motion -- read from ``frame_info.json``, ``ego_pose/``, and the
``dynamic_masks/{vehicle,human}`` layers, so a quick look at the output
directory's filenames shows how diverse the 200-clip sample is.

Usage:
    python -m scripts.eval_tagged_samples --config configs/waymo_ds_paper.yaml \
        --ckpt runs/waymo_ds_paper/ckpt_014000.pt --max-clips 200 \
        --out runs/waymo_ds_paper/tagged_samples_014000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from pointforward.data import WaymoDriveStudioClips, collate_clip, to_scene_input
from pointforward.losses import depth_rmse, psnr, ssim

from scripts.eval import build_model
from scripts.train import build_dataset


def _sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower()) or "na"


def _scene_conditions(scene_path: str) -> tuple[str, str]:
    info_path = os.path.join(scene_path, "frame_info.json")
    if not os.path.isfile(info_path):
        return "unknown", "unknown"
    with open(info_path, "r") as fh:
        info = json.load(fh)
    return str(info.get("weather", "unknown")), str(info.get("time_of_day", "unknown"))


def _ego_motion_tags(dataset, meta: dict, first_frame: int, last_frame: int) -> tuple[str, str]:
    p0 = dataset._ego_pose(meta, first_frame)
    p1 = dataset._ego_pose(meta, last_frame)
    dist = float(np.linalg.norm(p1[:3, 3] - p0[:3, 3]))
    yaw0 = math.atan2(p0[1, 0], p0[0, 0])
    yaw1 = math.atan2(p1[1, 0], p1[0, 0])
    dyaw_deg = math.degrees(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0)))
    ego_tag = "egomoving" if dist > 1.0 else "egostatic"
    if abs(dyaw_deg) < 3.0:
        turn_tag = "straight"
    else:
        turn_tag = "turnleft" if dyaw_deg > 0 else "turnright"
    return ego_tag, turn_tag


def _mask_has_motion(scene_path: str, layer: str, frames: list[int], cams: tuple[int, ...], thresh: float = 0.001) -> bool:
    for frame in frames:
        for cam in cams:
            p = os.path.join(scene_path, "dynamic_masks", layer, f"{frame:03d}_{cam}.png")
            if not os.path.isfile(p):
                continue
            with Image.open(p) as im:
                arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
            if (arr > 0.5).mean() > thresh:
                return True
    return False


def _avg(sums: dict, counts: dict) -> dict:
    return {k: (sums[k] / counts[k] if counts[k] else None) for k in sums}


def _stratified_indices(dataset, max_clips: int) -> list[int]:
    """Evenly cover every scene, and the full temporal range within each.

    ``dataset.index`` is built scene-by-scene (all of scene 0's sliding
    windows, then all of scene 1's, ...), so taking the first N indices
    concentrates almost entirely on the first one or two scenes. Instead,
    bucket indices by scene, then take an evenly-spaced (``linspace``)
    subset of each scene's bucket so both scene identity (weather/
    time_of_day) and within-scene time (ego motion, turning, dynamic
    object presence) vary across the sample.
    """
    by_scene: dict[int, list[int]] = {}
    for i, (si, _start) in enumerate(dataset.index):
        by_scene.setdefault(si, []).append(i)

    scene_ids = sorted(by_scene)
    n_scenes = len(scene_ids)
    base, extra = divmod(max_clips, n_scenes)

    selected: list[int] = []
    for j, si in enumerate(scene_ids):
        k = base + (1 if j < extra else 0)
        block = by_scene[si]
        if k >= len(block):
            picks = block
        else:
            picks = [block[int(round(x))] for x in np.linspace(0, len(block) - 1, k)]
        selected.extend(picks)
    return selected


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/waymo_ds_paper.yaml")
    ap.add_argument("--ckpt", default="runs/waymo_ds_paper/ckpt_014000.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-clips", type=int, default=200)
    ap.add_argument("--out", default="runs/waymo_ds_paper/tagged_samples")
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument(
        "--scenes", default=None,
        help="comma-separated scene names, overriding the split file entirely. "
        "For one-off scenes outside the configured split (e.g. a train-split "
        "scene sampled only for visual coverage, not held-out metrics).",
    )
    args = ap.parse_args()

    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # D-RMSE needs the target view's LiDAR projected too (same as scripts/eval.py).
    d = cfg.setdefault("data", {})
    d["target_lidar"] = True
    if args.scenes:
        dataset = WaymoDriveStudioClips(
            root=d["root"],
            scenes=args.scenes.split(","),
            cameras=tuple(d.get("cameras", (0, 1, 2))),
            num_context=d.get("num_context", 4),
            num_target=d.get("num_target", 4),
            frame_stride=d.get("frame_stride", 5),
            resolution=tuple(d.get("resolution", (576, 768))),
            resolutions=None,
            max_instances=d.get("max_instances", 24),
            deterministic=True,
            seed=cfg.get("seed", 0),
            clip_length=d.get("clip_length", 20),
            load_lidar=d.get("load_lidar", True),
            target_lidar=d.get("target_lidar", False),
            require_lidar=d.get("require_lidar", False),
            require_instances=d.get("require_instances", False),
        )
    else:
        dataset = build_dataset(cfg, args.split)
    print(f"{args.split}: {len(dataset)} clips over {len(dataset.scenes)} scenes")

    indices = _stratified_indices(dataset, args.max_clips)
    print(f"sampling {len(indices)} clips stratified across {len(dataset.scenes)} scenes")
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_clip)
    model = build_model(cfg, device, args.ckpt)

    scenes_by_name = {m["name"]: m for m in dataset.scenes}
    cams = tuple(dataset.cameras)

    os.makedirs(args.out, exist_ok=True)
    tmp_root = os.path.join(args.out, "_frames_tmp")
    os.makedirs(tmp_root, exist_ok=True)

    sums = {"full": {"psnr": 0.0, "ssim": 0.0, "drmse": 0.0}, "dynamic": {"psnr": 0.0, "ssim": 0.0, "drmse": 0.0}}
    counts = {"full": {"psnr": 0, "ssim": 0, "drmse": 0}, "dynamic": {"psnr": 0, "ssim": 0, "drmse": 0}}
    manifest_rows = []

    clip_idx = 0
    for batch in loader:
        if clip_idx >= args.max_clips:
            break
        for clip in batch:
            if clip_idx >= args.max_clips:
                break

            scene, targets, gt = to_scene_input(clip, device)
            tgt = clip["target"]
            want_depth = "sparse_depth" in tgt
            out = model(scene, targets, render_depth=want_depth)
            pred = out["rgb"].clamp(0, 1)

            dyn_mask = tgt["dynamic_mask"].to(device) if "dynamic_mask" in tgt else None

            row = {"clip": clip_idx, "scene": clip["scene"]}

            full_psnr, full_ssim = float(psnr(pred, gt)), float(ssim(pred, gt))
            sums["full"]["psnr"] += full_psnr; counts["full"]["psnr"] += 1
            sums["full"]["ssim"] += full_ssim; counts["full"]["ssim"] += 1
            row["full_psnr"], row["full_ssim"] = full_psnr, full_ssim

            row["dyn_psnr"] = row["dyn_ssim"] = row["full_drmse"] = row["dyn_drmse"] = None
            if dyn_mask is not None and bool((dyn_mask > 0.5).any()):
                dyn_psnr, dyn_ssim = float(psnr(pred, gt, mask=dyn_mask)), float(ssim(pred, gt, mask=dyn_mask))
                sums["dynamic"]["psnr"] += dyn_psnr; counts["dynamic"]["psnr"] += 1
                sums["dynamic"]["ssim"] += dyn_ssim; counts["dynamic"]["ssim"] += 1
                row["dyn_psnr"], row["dyn_ssim"] = dyn_psnr, dyn_ssim

            if "render_depth" in out and "sparse_depth" in tgt:
                gt_depth = tgt["sparse_depth"].to(device)
                gt_mask = tgt["sparse_mask"].to(device) > 0.5
                if bool(gt_mask.any()):
                    full_drmse = float(depth_rmse(out["render_depth"], gt_depth, gt_mask))
                    sums["full"]["drmse"] += full_drmse; counts["full"]["drmse"] += 1
                    row["full_drmse"] = full_drmse
                if dyn_mask is not None:
                    combo = gt_mask & (dyn_mask > 0.5)
                    if bool(combo.any()):
                        dyn_drmse = float(depth_rmse(out["render_depth"], gt_depth, combo))
                        sums["dynamic"]["drmse"] += dyn_drmse; counts["dynamic"]["drmse"] += 1
                        row["dyn_drmse"] = dyn_drmse

            # -- scene tags --
            meta = scenes_by_name[clip["scene"]]
            weather, tod = _scene_conditions(meta["path"])
            frames_list = clip["frames"]
            ego_tag, turn_tag = _ego_motion_tags(dataset, meta, int(frames_list[0]), int(frames_list[-1]))
            raw_target_frames = sorted(int(f) for f in torch.unique(frames_list[tgt["frame_index"]]))
            veh_tag = "vehmoving" if _mask_has_motion(meta["path"], "vehicle", raw_target_frames, cams) else "vehstatic"
            ped_tag = "pedmoving" if _mask_has_motion(meta["path"], "human", raw_target_frames, cams) else "noped"
            row.update(
                weather=weather, time_of_day=tod, ego=ego_tag, turn=turn_tag,
                vehicle=veh_tag, pedestrian=ped_tag, target_frames=raw_target_frames,
            )
            manifest_rows.append(row)

            # -- tagged GT|pred video (camera 0, one PNG per target frame) --
            num_cams = len(cams)
            num_target_views = pred.shape[0] // num_cams
            clip_tmp = os.path.join(tmp_root, f"clip{clip_idx:03d}")
            os.makedirs(clip_tmp, exist_ok=True)
            for t in range(num_target_views):
                vi = t * num_cams  # camera index 0 within this clip's camera list
                pair = torch.cat((gt[vi], pred[vi]), dim=1)  # GT on top, prediction on bottom
                save_image(pair, os.path.join(clip_tmp, f"frame_{t:02d}.png"))

            prefix = "TRAINSPLIT_nonheldout__" if args.scenes else ""
            fname = (
                f"{prefix}clip{clip_idx:03d}__{_sanitize(clip['scene'])}"
                f"__{_sanitize(weather)}_{_sanitize(tod)}__{ego_tag}_{turn_tag}"
                f"__{veh_tag}__{ped_tag}.mp4"
            )
            subprocess.run(
                [
                    "ffmpeg", "-y", "-framerate", str(args.fps),
                    "-i", os.path.join(clip_tmp, "frame_%02d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-pix_fmt", "yuv420p", os.path.join(args.out, fname),
                ],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            shutil.rmtree(clip_tmp)

            clip_idx += 1
            if clip_idx % 20 == 0:
                print(f"  {clip_idx} clips  full PSNR {sums['full']['psnr']/counts['full']['psnr']:.2f}")

    shutil.rmtree(tmp_root, ignore_errors=True)

    summary = {
        "protocol": "interp",
        "checkpoint": args.ckpt,
        "scenes_override": args.scenes,
        "non_held_out": bool(args.scenes),
        "clips": counts["full"]["psnr"],
        "full_image": _avg(sums["full"], counts["full"]),
        "dynamic_only": _avg(sums["dynamic"], counts["dynamic"]),
        "dynamic_only_clips": counts["dynamic"]["psnr"],
        "dynamic_only_drmse_clips": counts["dynamic"]["drmse"],
    }

    print("\n" + "=" * 60)
    print(f"clips evaluated: {summary['clips']}")
    print("full image  ", {k: round(v, 4) for k, v in summary["full_image"].items() if v is not None})
    print("dynamic only", {k: round(v, 4) for k, v in summary["dynamic_only"].items() if v is not None},
          f"(n={summary['dynamic_only_clips']}, drmse n={summary['dynamic_only_drmse_clips']})")

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    fieldnames = [
        "clip", "scene", "target_frames", "weather", "time_of_day", "ego", "turn", "vehicle", "pedestrian",
        "full_psnr", "full_ssim", "full_drmse", "dyn_psnr", "dyn_ssim", "dyn_drmse",
    ]
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    print(f"\nwritten to {args.out}/ (summary.json, manifest.csv, {summary['clips']} .mp4 files)")


if __name__ == "__main__":
    main()
