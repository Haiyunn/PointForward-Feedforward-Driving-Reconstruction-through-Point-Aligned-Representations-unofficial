"""Render checkpoint reconstructions beside vehicle-node removal, MP4 only.

Both panels share decoded Gaussians, cameras and timestamps. No image inpainting
is applied: newly exposed surfaces are only those reconstructed by the model.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from pointforward.data.drivestudio import WaymoDriveStudioClips
from pointforward.models import SceneInput, TargetViews
from pointforward.models.pointforward import Gaussians
from pointforward.models.scene_graph import assign_instances
from scripts.eval import build_model
from scripts.render_multiscene import boxes_to, to_uint8
from scripts.train import build_dataset


def chosen_instance_keys(dataset, meta, frames):
    """Match the loader's stable nearest-instance ordering, including tie order."""
    present = {}
    for frame in frames:
        for key, per_frame in meta['instances'].items():
            if frame in per_frame:
                present.setdefault(key, []).append(per_frame[frame][0])
    origin = dataset._ego_pose(meta, frames[0])[:3, 3]
    return sorted(present, key=lambda key: min(
        float(np.linalg.norm(pose[:3, 3] - origin)) for pose in present[key]
    ))[:dataset.max_instances]


def subset_gaussians(gaussians, keep):
    values = {f.name: getattr(gaussians, f.name)[keep]
              for f in fields(Gaussians) if f.name != 'extras'}
    values['extras'] = {
        key: value[keep] if value.ndim and value.shape[0] == gaussians.count else value
        for key, value in gaussians.extras.items()
    }
    return Gaussians(**values)


def remove_vehicles(dataset, meta, frames, world_from, boxes, sampled, gaussians, classes):
    keys = chosen_instance_keys(dataset, meta, frames)
    assert boxes is not None and len(keys) == boxes.num_instances
    is_vehicle = torch.tensor([classes[key] == 'Vehicle' for key in keys],
                              device=gaussians.means.device, dtype=torch.bool)
    ids = gaussians.instance_id
    remove = (ids >= 0) & is_vehicle[ids.clamp_min(0)]

    # Vehicles beyond the training-time node cap can live in the static branch.
    # Test those queries against ALL annotated vehicle boxes at their source time.
    vehicle_meta = dict(meta)
    vehicle_meta['instances'] = {key: value for key, value in meta['instances'].items()
                                 if classes[key] == 'Vehicle'}
    uncapped = copy.copy(dataset)
    uncapped.max_instances = len(vehicle_meta['instances'])
    vehicle_boxes = boxes_to(uncapped._build_boxes(vehicle_meta, frames, world_from),
                             gaussians.means.device)
    static = ids < 0
    if vehicle_boxes is not None and bool(static.any()):
        assigned = assign_instances(sampled['world_xyz'][static], vehicle_boxes,
                                    sampled['box_index'][static], chunk=16384)
        remove[static] |= assigned >= 0
    return subset_gaussians(gaussians, ~remove), int(remove.sum())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scenes', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--frames-per-scene', type=int, default=100)
    ap.add_argument('--seed', type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(4)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    assert checkpoint['step'] + 1 == 20000, 'Expected 20000 completed updates'
    del checkpoint
    assert torch.cuda.is_available(), 'CUDA is required for this renderer'
    device = torch.device('cuda')
    dataset = build_dataset(cfg, args.split)
    assert isinstance(dataset, WaymoDriveStudioClips)
    assert args.frames_per_scene > 0 and args.frames_per_scene % dataset.clip_length == 0
    names = args.scenes.split(',')
    assert len(names) == len(set(names))
    metas = {meta['name']: meta for meta in dataset.scenes}
    assert all(name in metas for name in names)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg, device, args.ckpt)
    h, w = dataset.resolution
    header = 48
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    camera_slot = dataset.cameras.index(0)
    for name in names:
        torch.manual_seed(args.seed + int(name))
        rng = np.random.default_rng(args.seed + int(name))
        meta = metas[name]
        n = args.frames_per_scene
        candidates = [i for i in range(len(meta['frames']) - n + 1)
                      if meta['frames'][i+n-1] - meta['frames'][i] == n - 1]
        assert candidates, f'{name}: no continuous {n}-frame window'
        start = int(rng.choice(candidates))
        scene_frames = meta['frames'][start:start+n]
        world_from = np.linalg.inv(dataset._ego_pose(meta, scene_frames[0]))
        with open(Path(meta['path']) / 'instances/instances_info.json') as fh:
            classes = {int(key): entry['class_name'] for key, entry in json.load(fh).items()}
        final = out_dir / f'scene{name}_step20000_original_vs_vehicle_removed_100f.mp4'
        if n != 100:
            final = final.with_name(final.name.replace('_100f', f'_{n}f'))
        assert not final.exists(), f'Refusing to overwrite {final}'
        partial = final.with_name(final.stem + '.partial.mp4')
        command = ['ffmpeg', '-nostdin', '-y', '-loglevel', 'error', '-f', 'rawvideo',
                   '-pix_fmt', 'rgb24', '-s', f'{2*w}x{h+header}', '-r', '10', '-i', '-',
                   '-an', '-c:v', 'libx264', '-crf', '18', '-threads', '4',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        removed_total = 0
        print(f'Scene {name}: frames {scene_frames[0]}..{scene_frames[-1]}', flush=True)
        try:
            assert encoder.stdin is not None
            for pos in range(0, n, dataset.clip_length):
                chunk = scene_frames[pos:pos+dataset.clip_length]
                slots = {frame: i for i, frame in enumerate(chunk)}
                times = dataset._normalised_times(chunk)
                context = sorted({chunk[min(i*dataset.frame_stride, len(chunk)-1)]
                                  for i in range(dataset.num_context)})
                ctx = dataset._collect(meta, context, slots, times, dataset.resolution,
                                       world_from, lidar=dataset.load_lidar)
                boxes = boxes_to(dataset._build_boxes(meta, chunk, world_from), device)
                scene = SceneInput(
                    **{key: ctx[key].to(device) for key in
                       ('images', 'intrinsics', 'cam_to_world', 'times', 'frame_index')},
                    boxes=boxes,
                    sparse_depth=ctx['sparse_depth'].to(device) if 'sparse_depth' in ctx else None,
                    sparse_mask=ctx['sparse_mask'].to(device) if 'sparse_mask' in ctx else None)
                depth = model.predict_depth(scene)
                sampled = model.build_queries(scene, depth)
                gaussians = model.decode(scene, depth, sampled)
                edited, removed = remove_vehicles(dataset, meta, chunk, world_from, boxes,
                                                    sampled, gaussians, classes)
                removed_total += removed
                del scene, depth, sampled
                dense = dataset._collect(meta, chunk, slots, times, dataset.resolution,
                                         world_from, lidar=False)
                # Render one frame per call to avoid retaining 20 feature maps on GPU.
                for slot, frame in enumerate(chunk):
                    index = slot * len(dataset.cameras) + camera_slot
                    target = TargetViews(**{key: dense[key][index:index+1].to(device)
                                            for key in ('intrinsics', 'cam_to_world',
                                                        'times', 'frame_index')})
                    panels = []
                    for gs in (gaussians, edited):
                        result = model.render(gs, boxes, target, height=h, width=w)
                        panels.append(to_uint8(result['rgb'][0]))
                        del result
                    image = Image.new('RGB', (2*w, h+header), (20, 24, 32))
                    image.paste(Image.fromarray(np.concatenate(panels, axis=1)), (0, header))
                    draw = ImageDraw.Draw(image)
                    draw.text((14, 13), f'Scene {name} | Step 20000 | Original render', font=font,
                              fill=(235, 239, 245))
                    draw.text((w+14, 13), f'Vehicles removed | Frame {frame:03d}', font=font,
                              fill=(113, 225, 184))
                    draw.line((w, header, w, h+header-1), fill=(235, 239, 245), width=2)
                    encoder.stdin.write(np.asarray(image).tobytes())
                del gaussians, edited, boxes, dense
                print(f'  {name}: {pos+len(chunk)}/{n} frames, removed {removed} Gaussians',
                      flush=True)
            encoder.stdin.close()
            assert encoder.wait() == 0, 'ffmpeg encoding failed'
            assert removed_total > 0, f'{name}: no vehicle Gaussians removed'
            probe = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                'stream=codec_name,pix_fmt,nb_frames', '-of', 'json', str(partial)]))['streams'][0]
            assert probe['codec_name'] == 'h264' and probe['pix_fmt'] == 'yuv420p'
            assert int(probe['nb_frames']) == n
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(partial),
                            '-f', 'null', '-'], check=True)
            os.replace(partial, final)
            print(f'SAVED {final}', flush=True)
        except BaseException:
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            if encoder.poll() is None:
                encoder.terminate()
            encoder.wait()
            partial.unlink(missing_ok=True)
            raise


if __name__ == '__main__':
    main()