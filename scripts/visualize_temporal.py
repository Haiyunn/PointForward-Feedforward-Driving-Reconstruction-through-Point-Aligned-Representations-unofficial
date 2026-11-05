"""Render the decoded 4D Gaussians across the FULL clip window (temporal demo).

scripts/visualize_pipeline.py demonstrates the pipeline stages but only renders
one target view (matching a context frame), which looks like a single static
snapshot. The actual "4D" part of PointForward is that decode() produces a set
of Gaussians that is time-INDEPENDENT: static Gaussians live in world space,
dynamic ones live in their instance's canonical space and are only pushed into
world space (pose_gaussians, Eq.7 temporal-opacity attenuation) at render time,
for whichever frame/timestamp is asked for.

This script decodes Gaussians ONCE from the 4 context frames, then renders that
SAME Gaussian set at every frame of the clip's full window (up to clip_length
frames, e.g. 20 frames / 2s at 10 FPS) from one fixed camera -- producing a GIF
(GT | prediction side by side) plus a static filmstrip PNG, so dynamic objects
(vehicles) can be seen moving while the static background stays put.

matplotlib is not installed in this environment; everything here uses only
numpy + PIL + torch.

Usage:
    python -m scripts.visualize_temporal --config configs/waymo_ds.yaml \
        --ckpt runs/waymo_ds_run1/ckpt_002250.pt --out runs/waymo_ds_run1/pipeline_vis
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from pointforward.data.drivestudio import WaymoDriveStudioClips
from pointforward.models import InstanceBoxes, SceneInput, TargetViews

from scripts.eval import build_model
from scripts.train import build_dataset


def to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    arr = img_chw.clamp(0, 1).detach().cpu().numpy()
    return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)


def label(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width, 18], fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 0))
    return np.array(pil)


def side_by_side(gt: np.ndarray, pred: np.ndarray, title: str) -> np.ndarray:
    pair = np.concatenate([gt, pred], axis=1)
    return label(pair, title)


def resize_max_width(arr: np.ndarray, max_w: int = 960) -> np.ndarray:
    h, w = arr.shape[:2]
    if w <= max_w:
        return arr
    scale = max_w / w
    new_w, new_h = max_w, max(1, int(round(h * scale)))
    return np.array(Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR))


def boxes_to(boxes: InstanceBoxes | None, device: torch.device) -> InstanceBoxes | None:
    if boxes is None:
        return None
    return InstanceBoxes(
        centers=boxes.centers.to(device),
        dims=boxes.dims.to(device),
        rots=boxes.rots.to(device),
        times=boxes.times.to(device),
        valid=boxes.valid.to(device),
    )


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", type=int, default=0, help="position within data.cameras to animate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-instances", type=int, default=1)
    ap.add_argument("--duration-ms", type=int, default=250)
    args = ap.parse_args()

    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(cfg, args.split)
    assert isinstance(dataset, WaymoDriveStudioClips), "this script assumes the drivestudio layout"
    print(f"{args.split}: {len(dataset)} clips over {len(dataset.scenes)} scenes")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(dataset))
    chosen_idx = None
    for idx in order[:300]:
        si, start = dataset.index[int(idx)]
        meta = dataset.scenes[si]
        window = meta["frames"][start : start + dataset.clip_length]
        present = sum(1 for per_frame in meta["instances"].values() if any(f in per_frame for f in window))
        if present >= args.min_instances:
            chosen_idx = int(idx)
            break
    assert chosen_idx is not None, "no clip with a dynamic instance found in 300 tries"

    si, start = dataset.index[chosen_idx]
    meta = dataset.scenes[si]
    window = meta["frames"][start : start + dataset.clip_length]
    print(f"scene={meta['name']} window={window}")

    resolution = dataset.resolution
    world_from = np.linalg.inv(dataset._ego_pose(meta, window[0]))
    norm_times = dataset._normalised_times(window)
    slot_of = {f: i for i, f in enumerate(window)}

    context_frames = sorted(
        {window[min(i * dataset.frame_stride, len(window) - 1)] for i in range(dataset.num_context)}
    )
    print(f"context_frames={context_frames}")

    ctx = dataset._collect(
        meta, context_frames, slot_of, norm_times, resolution, world_from, lidar=dataset.load_lidar
    )
    boxes = dataset._build_boxes(meta, window, world_from)
    assert boxes is not None and boxes.num_instances > 0, "no dynamic instances in this window"
    print(f"num_instances={boxes.num_instances} box_slots={boxes.valid.shape[1]}")

    scene = SceneInput(
        images=ctx["images"].to(device),
        intrinsics=ctx["intrinsics"].to(device),
        cam_to_world=ctx["cam_to_world"].to(device),
        times=ctx["times"].to(device),
        frame_index=ctx["frame_index"].to(device),
        boxes=boxes_to(boxes, device),
        sparse_depth=ctx["sparse_depth"].to(device) if "sparse_depth" in ctx else None,
        sparse_mask=ctx["sparse_mask"].to(device) if "sparse_mask" in ctx else None,
    )

    model = build_model(cfg, device, args.ckpt)

    depth = model.predict_depth(scene)
    sampled = model.build_queries(scene, depth)
    gaussians = model.decode(scene, depth, sampled)

    # Dense GT + target views across the FULL window, one fixed camera.
    dense = dataset._collect(meta, window, slot_of, norm_times, resolution, world_from, lidar=False)
    num_cams = len(dataset.cameras)
    sel = torch.arange(args.camera, dense["images"].shape[0], num_cams)

    gt_images = dense["images"][sel]
    targets = TargetViews(
        intrinsics=dense["intrinsics"][sel].to(device),
        cam_to_world=dense["cam_to_world"][sel].to(device),
        times=dense["times"][sel].to(device),
        frame_index=dense["frame_index"][sel].to(device),
    )

    h, w = scene.image_size
    out = model.render(gaussians, boxes_to(boxes, device), targets, height=h, width=w, render_depth=False)
    pred = out["rgb"]  # (T, 3, H, W)

    os.makedirs(args.out, exist_ok=True)
    frames_png = []
    for t in range(pred.shape[0]):
        gt_np = to_uint8(gt_images[t])
        pred_np = to_uint8(pred[t])
        time_val = float(dense["times"][sel][t])
        title = f"t={time_val:.2f}  frame={window[t]}  GT | Prediction (same Gaussians, different time)"
        frames_png.append(resize_max_width(side_by_side(gt_np, pred_np, title)))

    gif_path = os.path.join(args.out, "temporal_4d.gif")
    pil_frames = [Image.fromarray(f) for f in frames_png]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=args.duration_ms, loop=0)
    print(f"saved {gif_path}")

    n = len(frames_png)
    idxs = np.linspace(0, n - 1, min(8, n)).round().astype(int)
    strip = np.concatenate([frames_png[i] for i in idxs], axis=0)
    strip_path = os.path.join(args.out, "temporal_4d_filmstrip.png")
    Image.fromarray(strip).save(strip_path)
    print(f"saved {strip_path}")


if __name__ == "__main__":
    main()
