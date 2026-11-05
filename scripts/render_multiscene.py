'''Render one continuous multi-scene video per view (frames 0..N-1 across scenes).

No single scene has 1000 contiguous frames, so a 0..1000-frame request must
span multiple *different* scenes back to back.  Each scene gets its own
independent world_from (scenes have unrelated coordinate frames -- there is
no meaningful shared spatial anchor across scene boundaries, unlike windows
within one scene).  Within a scene, consecutive 20-frame windows still share
that scene's world_from for spatial continuity, same as render_continuous.py.

Decoding is done once per (scene, window) and reused across all requested
views -- only render() differs per view, via a rigid camera-pose offset.

Usage:
    python -m scripts.render_multiscene --config configs/waymo_ds.yaml \
        --ckpt runs/waymo_ds_run1/ckpt_006000.pt --out runs/waymo_ds_run1/multiscene \
        --frames-per-scene 200 --num-scenes 3
'''

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import numpy as np
import torch
import yaml
from PIL import Image

from pointforward.data.drivestudio import WaymoDriveStudioClips
from pointforward.models import InstanceBoxes, SceneInput, TargetViews

from scripts.eval import build_model
from scripts.train import build_dataset


VIEWS = [
    ('real', 0.0, 0.0),
    ('left3p5m', -3.5, 0.0),
    ('right3p5m', 3.5, 0.0),
]


def to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    arr = img_chw.clamp(0, 1).detach().cpu().numpy()
    return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)


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


def pick_window(dataset, si, num_frames, rng, min_instances):
    meta = dataset.scenes[si]
    n = len(meta['frames'])
    if n < num_frames:
        return None
    for start in rng.permutation(n - num_frames + 1)[:20]:
        window = meta['frames'][start : start + num_frames]
        present = sum(1 for per_frame in meta['instances'].values() if any(f in per_frame for f in window))
        if present >= min_instances:
            return int(start)
    return None


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--split', default='val')
    ap.add_argument('--out', required=True)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--frames-per-scene', type=int, default=100)
    ap.add_argument('--num-scenes', type=int, default=10)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--min-instances', type=int, default=1)
    args = ap.parse_args()

    with open(args.config, 'r') as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = build_dataset(cfg, args.split)
    assert isinstance(dataset, WaymoDriveStudioClips)
    print(f'{args.split}: {len(dataset)} clips over {len(dataset.scenes)} scenes')
    assert args.frames_per_scene % dataset.clip_length == 0, 'frames_per_scene must be a multiple of clip_length'

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(dataset.scenes))

    plan = []
    for si in order:
        start = pick_window(dataset, int(si), args.frames_per_scene, rng, args.min_instances)
        if start is not None:
            plan.append((int(si), start))
        if len(plan) >= args.num_scenes:
            break
    assert len(plan) >= args.num_scenes, f'only found {len(plan)} usable scenes, need {args.num_scenes}'

    resolution = dataset.resolution
    num_cams = len(dataset.cameras)
    model = build_model(cfg, device, args.ckpt)

    videos_dir = os.path.join(args.out, 'videos')
    os.makedirs(videos_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pointforward_render_') as temp_dir:
        view_dirs = {}
        for name, _, _ in VIEWS:
            d = os.path.join(temp_dir, name)
            os.makedirs(d, exist_ok=True)
            view_dirs[name] = d

        global_frame = 0
        scene_bounds = []
        for scene_idx, (si, start) in enumerate(plan):
            meta = dataset.scenes[si]
            scene_window = meta['frames'][start : start + args.frames_per_scene]
            world_from = np.linalg.inv(dataset._ego_pose(meta, scene_window[0]))
            print(f'scene {scene_idx+1}/{len(plan)}: name={meta[chr(110)+chr(97)+chr(109)+chr(101)]} start={start} frames={len(scene_window)} global_start={global_frame}')
            scene_bounds.append((meta['name'], global_frame, global_frame + len(scene_window)))

            pos = 0
            while pos < len(scene_window):
                chunk = scene_window[pos : pos + dataset.clip_length]
                pos += len(chunk)

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
                        Image.fromarray(to_uint8(pred[t])).save(
                            os.path.join(view_dirs[name], f'frame_{global_frame + t:04d}.png')
                        )
                global_frame += len(chunk)

        print('scene boundaries (name, global_start, global_end):')
        for b in scene_bounds:
            print(' ', b)

        for scene_name, first, last in scene_bounds:
            for name, _, _ in VIEWS:
                out_path = os.path.join(
                    videos_dir, f'scene{scene_name}_{name}_{last-first:04d}f.mp4'
                )
                subprocess.run([
                    'ffmpeg', '-nostdin', '-y', '-framerate', '10',
                    '-start_number', str(first),
                    '-i', os.path.join(view_dirs[name], 'frame_%04d.png'),
                    '-frames:v', str(last - first),
                    '-c:v', 'libx264', '-crf', '18', '-threads', '4',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                    out_path, '-loglevel', 'error',
                ], check=True)
                print(f'saved {out_path}')


if __name__ == '__main__':
    main()
