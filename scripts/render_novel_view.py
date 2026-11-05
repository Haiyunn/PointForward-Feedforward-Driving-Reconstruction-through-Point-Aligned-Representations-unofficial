"""Render a genuinely novel-viewpoint video from a decoded 4D scene.

Unlike scripts/visualize_temporal.py (which replays a REAL camera across the
clip's frames), this renders a VIRTUAL camera pose that was never captured by
any of the 3 real frontal cameras: a fixed rigid offset (lateral + vertical)
from the real front camera's pose at each frame, applied across the whole
window.  Since PointForward's render() only needs a cam_to_world + intrinsics
matrix, no real sensor pose is required -- this is genuine novel-view
synthesis, not interpolation between existing views.

Decodes Gaussians ONCE from the context frames, then calls render() once per
frame of the window at the virtual pose, and muxes the resulting PNGs into an
mp4 with ffmpeg (real per-frame video, playback fps = capture fps).

Usage:
    python -m scripts.render_novel_view --config configs/waymo_ds.yaml \
        --ckpt runs/waymo_ds_run1/ckpt_006000.pt --out runs/waymo_ds_run1/novel_view \
        --num-frames 20 --lateral 2.0 --vertical 1.5
"""

from __future__ import annotations

import argparse
import os
import subprocess

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


def boxes_to(boxes, device):
    if boxes is None:
        return None
    return InstanceBoxes(
        centers=boxes.centers.to(device),
        dims=boxes.dims.to(device),
        rots=boxes.rots.to(device),
        times=boxes.times.to(device),
        valid=boxes.valid.to(device),
    )


def offset_pose(cam_to_world: np.ndarray, lateral: float, vertical: float) -> np.ndarray:
    """Shift a camera pose by lateral metres along its own right axis and
    vertical metres along world +Z, keeping orientation unchanged."""
    out = cam_to_world.copy()
    right = cam_to_world[:3, 0]  # OpenCV convention: x = right
    out[:3, 3] = cam_to_world[:3, 3] + right * lateral + np.array([0.0, 0.0, vertical])
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--split', default='val')
    ap.add_argument('--out', required=True)
    ap.add_argument('--camera', type=int, default=0, help='which real camera pose to offset from')
    ap.add_argument('--num-frames', type=int, default=20, help='how many frames of the clip window to render (<= clip_length)')
    ap.add_argument('--lateral', type=float, default=2.0, help='metres, along camera right axis')
    ap.add_argument('--vertical', type=float, default=1.5, help='metres, world +Z (up)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-instances', type=int, default=1)
    args = ap.parse_args()

    with open(args.config, 'r') as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = build_dataset(cfg, args.split)
    assert isinstance(dataset, WaymoDriveStudioClips)
    print(f'{args.split}: {len(dataset)} clips over {len(dataset.scenes)} scenes')

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(dataset))
    chosen_idx = None
    for idx in order[:300]:
        si, start = dataset.index[int(idx)]
        meta = dataset.scenes[si]
        window = meta['frames'][start : start + dataset.clip_length]
        present = sum(1 for per_frame in meta['instances'].values() if any(f in per_frame for f in window))
        if present >= args.min_instances:
            chosen_idx = int(idx)
            break
    assert chosen_idx is not None

    si, start = dataset.index[chosen_idx]
    meta = dataset.scenes[si]
    window = meta['frames'][start : start + dataset.clip_length]
    n_frames = min(args.num_frames, len(window))
    window = window[:n_frames]
    print(f'scene={meta["name"]} window={window}')

    resolution = dataset.resolution
    world_from = np.linalg.inv(dataset._ego_pose(meta, window[0]))
    norm_times = dataset._normalised_times(window)
    slot_of = {f: i for i, f in enumerate(window)}

    context_frames = sorted(
        {window[min(i * dataset.frame_stride, len(window) - 1)] for i in range(dataset.num_context)}
    )
    print(f'context_frames={context_frames}')

    ctx = dataset._collect(meta, context_frames, slot_of, norm_times, resolution, world_from, lidar=dataset.load_lidar)
    boxes = dataset._build_boxes(meta, window, world_from)
    assert boxes is not None and boxes.num_instances > 0
    print(f'num_instances={boxes.num_instances}')

    scene = SceneInput(
        images=ctx['images'].to(device),
        intrinsics=ctx['intrinsics'].to(device),
        cam_to_world=ctx['cam_to_world'].to(device),
        times=ctx['times'].to(device),
        frame_index=ctx['frame_index'].to(device),
        boxes=boxes_to(boxes, device),
        sparse_depth=ctx['sparse_depth'].to(device) if 'sparse_depth' in ctx else None,
        sparse_mask=ctx['sparse_mask'].to(device) if 'sparse_mask' in ctx else None,
    )

    model = build_model(cfg, device, args.ckpt)
    depth = model.predict_depth(scene)
    sampled = model.build_queries(scene, depth)
    gaussians = model.decode(scene, depth, sampled)

    dense = dataset._collect(meta, window, slot_of, norm_times, resolution, world_from, lidar=False)
    num_cams = len(dataset.cameras)
    sel = torch.arange(args.camera, dense['images'].shape[0], num_cams)

    real_c2w = dense['cam_to_world'][sel].cpu().numpy()
    virtual_c2w = np.stack([offset_pose(p, args.lateral, args.vertical) for p in real_c2w], axis=0)

    targets = TargetViews(
        intrinsics=dense['intrinsics'][sel].to(device),
        cam_to_world=torch.from_numpy(virtual_c2w).to(device=device, dtype=dense['cam_to_world'].dtype),
        times=dense['times'][sel].to(device),
        frame_index=dense['frame_index'][sel].to(device),
    )

    h, w = scene.image_size
    out = model.render(gaussians, boxes_to(boxes, device), targets, height=h, width=w, render_depth=False)
    pred = out['rgb']

    os.makedirs(args.out, exist_ok=True)
    for t in range(pred.shape[0]):
        img = to_uint8(pred[t])
        time_val = float(dense['times'][sel][t])
        img = label(img, f'novel view (+{args.lateral:.1f}m lateral, +{args.vertical:.1f}m up)  t={time_val:.2f}  frame={window[t]}')
        Image.fromarray(img).save(os.path.join(args.out, f'frame_{t:03d}.png'))

    mp4_path = os.path.join(args.out, 'novel_view.mp4')
    subprocess.run([
        'ffmpeg', '-y', '-framerate', '10', '-i', os.path.join(args.out, 'frame_%03d.png'),
        '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', mp4_path, '-loglevel', 'error',
    ], check=True)
    print(f'saved {mp4_path}')


if __name__ == '__main__':
    main()
