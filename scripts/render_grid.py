'''Render a grid of (duration x novel-view) videos from one checkpoint.

Decodes Gaussians ONCE per clip_length window (shared across all views --
geometry does not depend on where we render from), then renders each of a
fixed set of virtual camera offsets from that same decode.  All views share
one world_from anchor so ego motion stays spatially continuous across
windows.  After rendering the full sequence per view, every requested
duration is produced as a plain frame-count truncation via ffmpeg -- no
extra inference needed per duration.

Usage:
    python -m scripts.render_grid --config configs/waymo_ds.yaml \
        --ckpt runs/waymo_ds_run1/ckpt_006000.pt --out runs/waymo_ds_run1/grid \
        --num-frames 190 --duration-step 10
'''

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


VIEWS = [
    ('real', 0.0, 0.0),
    ('left2m', -2.0, 0.0),
    ('right2m', 2.0, 0.0),
    ('up1p5m', 0.0, 1.5),
]


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
    out = cam_to_world.copy()
    right = cam_to_world[:3, 0]
    out[:3, 3] = cam_to_world[:3, 3] + right * lateral + np.array([0.0, 0.0, vertical])
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--split', default='val')
    ap.add_argument('--out', required=True)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--num-frames', type=int, default=190)
    ap.add_argument('--duration-step', type=int, default=10)
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
    candidates = [si for si, meta in enumerate(dataset.scenes) if len(meta['frames']) >= args.num_frames]
    assert candidates, f'no scene has >= {args.num_frames} contiguous frames'
    assert args.num_frames % dataset.clip_length == 0, 'num_frames must be a multiple of clip_length to avoid short final windows breaking the fixed-view-count fusion module'

    chosen = None
    for si in rng.permutation(candidates):
        meta = dataset.scenes[si]
        n = len(meta['frames'])
        for start in rng.permutation(n - args.num_frames + 1)[:20]:
            window = meta['frames'][start : start + args.num_frames]
            present = sum(1 for per_frame in meta['instances'].values() if any(f in per_frame for f in window))
            if present >= args.min_instances:
                chosen = (int(si), int(start))
                break
        if chosen:
            break
    assert chosen is not None
    si, start = chosen
    meta = dataset.scenes[si]
    global_window = meta['frames'][start : start + args.num_frames]
    print(f'scene={meta["name"]} start={start} num_frames={len(global_window)}')

    resolution = dataset.resolution
    num_cams = len(dataset.cameras)
    world_from = np.linalg.inv(dataset._ego_pose(meta, global_window[0]))

    model = build_model(cfg, device, args.ckpt)

    view_dirs = {}
    for name, _, _ in VIEWS:
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        view_dirs[name] = d

    base = 0
    pos = 0
    chunk_idx = 0
    while pos < len(global_window):
        chunk = global_window[pos : pos + dataset.clip_length]
        pos += len(chunk)
        chunk_idx += 1

        slot_of = {f: i for i, f in enumerate(chunk)}
        norm_times = dataset._normalised_times(chunk)
        context_frames = sorted(
            {chunk[min(i * dataset.frame_stride, len(chunk) - 1)] for i in range(dataset.num_context)}
        )

        ctx = dataset._collect(meta, context_frames, slot_of, norm_times, resolution, world_from, lidar=dataset.load_lidar)
        boxes = dataset._build_boxes(meta, chunk, world_from)

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

        depth = model.predict_depth(scene)
        sampled = model.build_queries(scene, depth)
        gaussians = model.decode(scene, depth, sampled)

        dense = dataset._collect(meta, chunk, slot_of, norm_times, resolution, world_from, lidar=False)
        sel = torch.arange(args.camera, dense['images'].shape[0], num_cams)
        real_c2w = dense['cam_to_world'][sel].cpu().numpy()

        h, w = resolution
        num_instances = 0 if boxes is None else boxes.num_instances

        for name, lateral, vertical in VIEWS:
            if lateral == 0.0 and vertical == 0.0:
                virtual_c2w = dense['cam_to_world'][sel].to(device)
            else:
                offset = np.stack([offset_pose(p, lateral, vertical) for p in real_c2w], axis=0)
                virtual_c2w = torch.from_numpy(offset).to(device=device, dtype=dense['cam_to_world'].dtype)

            targets = TargetViews(
                intrinsics=dense['intrinsics'][sel].to(device),
                cam_to_world=virtual_c2w,
                times=dense['times'][sel].to(device),
                frame_index=dense['frame_index'][sel].to(device),
            )
            out = model.render(gaussians, boxes_to(boxes, device), targets, height=h, width=w, render_depth=False)
            pred = out['rgb']

            for t in range(pred.shape[0]):
                img = to_uint8(pred[t])
                img = label(img, f'{name}  frame={chunk[t]}  window={chunk_idx}  inst={num_instances}')
                Image.fromarray(img).save(os.path.join(view_dirs[name], f'frame_{base + t:03d}.png'))
        print(f'window {chunk_idx}: frames {chunk[0]}..{chunk[-1]}  gaussians={gaussians.count}  instances={num_instances}')
        base += len(chunk)

    videos_dir = os.path.join(args.out, 'videos')
    os.makedirs(videos_dir, exist_ok=True)
    durations = list(range(args.duration_step, args.num_frames + 1, args.duration_step))
    made = 0
    for name, _, _ in VIEWS:
        for d in durations:
            out_path = os.path.join(videos_dir, f'{name}_{d:03d}f.mp4')
            subprocess.run([
                'ffmpeg', '-y', '-framerate', '10', '-i', os.path.join(view_dirs[name], 'frame_%03d.png'),
                '-frames:v', str(d), '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                out_path, '-loglevel', 'error',
            ], check=True)
            made += 1
    print(f'wrote {made} videos to {videos_dir}')


if __name__ == '__main__':
    main()
