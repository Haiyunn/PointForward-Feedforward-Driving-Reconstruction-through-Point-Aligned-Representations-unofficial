"""Resumable manifest-driven rendering and lossless native Gaussian assets.

Each .gs.pt is an indexed sequence asset referring to shared, full-precision
20-frame .pt chunks and the saved RGB refiner. It is NOT a standard RGB/SH PLY.
Camera shifts are along ego-left (Waymo +Y), not each camera's optical X axis.
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from pointforward.models import SceneInput, TargetViews
from pointforward.models.pointforward import Gaussians
from scripts.eval import build_model
from scripts.render_multiscene import boxes_to, to_uint8
from scripts.render_vehicle_removal import chosen_instance_keys, subset_gaussians
from scripts.train import build_dataset

CAMERAS = {0: 'front', 1: 'front_left', 2: 'front_right'}
TARGET_KEYS = ('intrinsics', 'cam_to_world', 'times', 'frame_index')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(temp, path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    torch.save(value, temp)
    os.replace(temp, path)


def cpu_fields(obj):
    if obj is None:
        return None
    return {field.name: ({k: v.detach().cpu() for k, v in obj.extras.items()}
                        if field.name == 'extras' else getattr(obj, field.name).detach().cpu())
            for field in dataclasses.fields(obj)}


def selected_mask(gs, keys, remove_ids):
    """Only selected graph nodes; never remove static/background Gaussians."""
    mask = torch.zeros_like(gs.instance_id, dtype=torch.bool)
    for key in remove_ids:
        if key in keys:
            mask |= gs.instance_id == keys.index(key)
    assert not bool((mask & (gs.instance_id < 0)).any())
    return mask


def probe_video(path, frames):
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=codec_name,pix_fmt,nb_frames', '-of', 'json', str(path)]))['streams'][0]
    assert probe['codec_name'] == 'h264' and probe['pix_fmt'] == 'yuv420p'
    assert int(probe['nb_frames']) == frames, (path, probe, frames)


class Encoder:
    def __init__(self, path, height, width, fps):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.path.with_name(self.path.stem + '.partial.mp4')
        self.process = subprocess.Popen([
            'ffmpeg', '-nostdin', '-y', '-v', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps),
            '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-threads', '1',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.partial)],
            stdin=subprocess.PIPE)

    def write(self, image):
        self.process.stdin.write(image.tobytes())

    def finish(self, frames):
        self.process.stdin.close()
        assert self.process.wait() == 0, f'ffmpeg failed: {self.path}'
        probe_video(self.partial, frames)
        os.replace(self.partial, self.path)

    def abort(self):
        if self.process.poll() is None:
            self.process.terminate()
        if not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        self.process.wait()


def camera_targets(dataset, meta, frames, world_from, context):
    """Construct calibrated targets without loading 60 unnecessary RGB images."""
    cameras = len(dataset.cameras)
    poses = [dataset._cam_to_world(meta, frame, camera, world_from)
             for frame in frames for camera in dataset.cameras]
    return dict(
        intrinsics=context['intrinsics'][:cameras].repeat(len(frames), 1, 1),
        cam_to_world=torch.from_numpy(np.stack(poses)).float(),
        times=dataset._normalised_times(frames).repeat_interleave(cameras),
        frame_index=torch.arange(len(frames)).repeat_interleave(cameras))


def render_scene(args, record, dataset, model, root, fingerprint):
    device = torch.device('cuda')
    name = record['name']
    frames = record['frames'][:args.max_frames]
    durations = [n for n in (20, 60, 180) if n <= len(frames)]
    assert len(frames) in (20, 60, 180)
    meta = next(m for m in dataset.scenes if m['name'] == name)
    assert all(f in meta['frames'] for f in frames)
    assert all(b == a + 1 for a, b in zip(frames, frames[1:]))
    scene_dir = root / f'scene{name}'
    scene_dir.mkdir(exist_ok=True)
    with (scene_dir / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done_path = scene_dir / 'complete.json'
        if done_path.exists():
            done = json.loads(done_path.read_text())
            assert done['fingerprint'] == fingerprint
            paths = done['artifacts'] + done['chunks']
            if all((root / p).is_file() and (root / p).stat().st_size > 0 for p in paths):
                print(f'SKIP complete scene{name}', flush=True)
                return
        remove_ids = record['remove_instance_ids']
        assert len(remove_ids) <= 2
        eligible = {e['global_instance_id'] for e in
                    record['vehicle_selection']['selected_tracks'] if e['eligible']}
        assert set(remove_ids) <= eligible
        assert not (set(remove_ids) & set(record['vehicle_selection']['preserve_instance_ids']))
        branches = {'base': (0.0, False), 'left3p5m': (3.5, False),
                    'right3p5m': (-3.5, False)}
        if remove_ids:
            branches.update({'remove_moving': (0.0, True),
                             'left3p5m_remove_moving': (3.5, True),
                             'right3p5m_remove_moving': (-3.5, True)})
        world_from = np.linalg.inv(dataset._ego_pose(meta, frames[0]))
        height, width = dataset.resolution
        artifacts, chunk_paths = [], []
        encoders = {}
        removed_total = 0
        try:
            for pos in range(0, len(frames), dataset.clip_length):
                start = time.monotonic()
                chunk = frames[pos:pos + dataset.clip_length]
                chunk_path = scene_dir / 'chunks' / f'window_{pos:03d}.pt'
                if chunk_path.exists():
                    payload = torch.load(chunk_path, map_location='cpu', weights_only=False)
                    assert payload['fingerprint'] == fingerprint and payload['frames'] == chunk
                    gs = Gaussians(**{k: ({a: b.to(device) for a, b in v.items()}
                                           if k == 'extras' else v.to(device))
                                     for k, v in payload['gaussians'].items()})
                    from pointforward.models import InstanceBoxes
                    boxes = (InstanceBoxes(**{k: v.to(device) for k, v in payload['boxes'].items()})
                             if payload['boxes'] else None)
                    targets = payload['targets']
                    keys = payload['global_instance_keys']
                    del payload
                else:
                    torch.manual_seed(args.seed + int(name) * 1000 + pos)
                    slots = {frame: i for i, frame in enumerate(chunk)}
                    times = dataset._normalised_times(chunk)
                    context_frames = sorted({chunk[min(i * dataset.frame_stride, len(chunk) - 1)]
                                             for i in range(dataset.num_context)})
                    context = dataset._collect(meta, context_frames, slots, times,
                                               dataset.resolution, world_from, lidar=dataset.load_lidar)
                    boxes = boxes_to(dataset._build_boxes(meta, chunk, world_from), device)
                    scene = SceneInput(**{k: context[k].to(device) for k in ('images', *TARGET_KEYS)},
                                       boxes=boxes,
                                       sparse_depth=context['sparse_depth'].to(device),
                                       sparse_mask=context['sparse_mask'].to(device))
                    depth = model.predict_depth(scene)
                    sampled = model.build_queries(scene, depth)
                    gs = model.decode(scene, depth, sampled)
                    targets = camera_targets(dataset, meta, chunk, world_from, context)
                    keys = chosen_instance_keys(dataset, meta, chunk)
                    assert len(keys) == (boxes.num_instances if boxes else 0)
                    payload = dict(schema_version=1, fingerprint=fingerprint, frames=chunk,
                                   world_from=torch.from_numpy(world_from),
                                   ego_left=torch.tensor(np.stack([
                                       (world_from @ dataset._ego_pose(meta, f))[:3, 1]
                                       for f in chunk]), dtype=torch.float32),
                                   global_instance_keys=keys, gaussians=cpu_fields(gs),
                                   boxes=cpu_fields(boxes), targets=targets,
                                   resolution=[height, width], cameras=list(dataset.cameras))
                    atomic_save(chunk_path, payload)
                    del scene, depth, sampled, context, payload
                chunk_paths.append(str(chunk_path.relative_to(root)))
                mask = selected_mask(gs, keys, remove_ids)
                removed = int(mask.sum())
                removed_total += removed
                edited = subset_gaussians(gs, ~mask) if remove_ids else None
                for branch, (lateral, removal) in branches.items():
                    for camera_slot, camera in enumerate(dataset.cameras):
                        cam_name = CAMERAS[camera]
                        for n in durations:
                            key = branch, camera, n
                            if pos == 0:
                                video = scene_dir / branch / f'{cam_name}_{n:03d}f.mp4'
                                encoders[key] = Encoder(video, height, width, 10)
                        for slot, frame in enumerate(chunk):
                            index = slot * len(dataset.cameras) + camera_slot
                            values = {k: targets[k][index:index + 1].to(device) for k in TARGET_KEYS}
                            if lateral:
                                ego_left = (world_from @ dataset._ego_pose(meta, frame))[:3, 1]
                                values['cam_to_world'] = values['cam_to_world'].clone()
                                values['cam_to_world'][0, :3, 3] += torch.as_tensor(
                                    ego_left * lateral, device=device, dtype=torch.float32)
                            result = model.render(edited if removal else gs, boxes,
                                                  TargetViews(**values), height=height, width=width)
                            image = to_uint8(result['rgb'][0])
                            del result
                            for n in durations:
                                if pos + slot < n:
                                    encoders[branch, camera, n].write(image)
                        for n in durations:
                            if pos + len(chunk) == n:
                                encoder = encoders.pop((branch, camera, n))
                                encoder.finish(n)
                                video = encoder.path
                                gs_path = video.with_suffix('.gs.pt')
                                atomic_save(gs_path, dict(
                                    schema_version=1, format='pointforward_indexed_gaussian_sequence',
                                    root_relative_to_asset='../..', fingerprint=fingerprint,
                                    chunks=list(chunk_paths), refiner='rgb_refiner.pt',
                                    camera=camera, frames=frames[:n], fps=10,
                                    resolution=[height, width], lateral_ego_left_m=lateral,
                                    remove_global_instance_ids=remove_ids if removal else [],
                                    removal_policy='selected_dynamic_nodes_only_preserve_all_static',
                                    checkpoint=str(Path(args.ckpt).resolve())))
                                artifacts.extend([str(video.relative_to(root)), str(gs_path.relative_to(root))])
                atomic_json(scene_dir / 'progress.json', dict(
                    scene=name, frames_done=pos + len(chunk), total_frames=len(frames),
                    removed_gaussians_this_chunk=removed, artifacts=artifacts,
                    seconds_this_chunk=round(time.monotonic() - start, 2)))
                print(f'scene{name} {pos + len(chunk)}/{len(frames)} '
                      f'gs={gs.count} removed={removed} {time.monotonic() - start:.1f}s', flush=True)
                del gs, edited, boxes, targets
                torch.cuda.empty_cache()
            if remove_ids:
                assert removed_total > 0, f'scene{name}: no selected vehicle Gaussian removed'
            atomic_json(done_path, dict(fingerprint=fingerprint, artifacts=artifacts,
                                       chunks=chunk_paths, removed_gaussians=removed_total))
            (scene_dir / 'error.json').unlink(missing_ok=True)
        finally:
            for encoder in encoders.values():
                encoder.abort()


@torch.no_grad()
def worker(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_bytes = Path(args.manifest).read_bytes()
    manifest = json.loads(manifest_bytes)
    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    assert checkpoint['step'] + 1 == 20000
    digest = hashlib.sha256(manifest_bytes)
    with open(args.ckpt, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    digest.update(str(args.max_frames).encode())
    fingerprint = digest.hexdigest()
    cfg = manifest['config_snapshot']
    assert cfg['data']['cameras'] == [0, 1, 2] and cfg['data']['clip_length'] == 20
    model = build_model(cfg, torch.device('cuda'), None)
    missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
    assert not [k for k in missing if not k.startswith('depth_model.')], missing
    assert not unexpected, unexpected
    del checkpoint
    with (root / '.refiner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (root / 'rgb_refiner.pt').exists():
            atomic_save(root / 'rgb_refiner.pt', dict(fingerprint=fingerprint,
                        state_dict={k: v.cpu() for k, v in model.refiner.state_dict().items()}))
        else:
            assert torch.load(root / 'rgb_refiner.pt', weights_only=False)['fingerprint'] == fingerprint
    records = manifest['scenes'][args.worker_index::args.workers]
    if args.scenes:
        records = [s for s in records if s['name'] in args.scenes.split(',')]
    datasets = {}
    failed = []
    for record in records:
        try:
            split = record['split']
            if split not in datasets:
                datasets[split] = build_dataset(cfg, split)
            render_scene(args, record, datasets[split], model, root, fingerprint)
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            failed.append(record['name'])
            atomic_json(root / f'scene{record["name"]}' / 'error.json', dict(error=error))
            torch.cuda.empty_cache()
    atomic_json(root / f'worker_{args.worker_index}_status.json',
                dict(state='failed' if failed else 'complete', failed_scenes=failed,
                     scenes=[r['name'] for r in records]))
    if failed:
        raise RuntimeError(f'Failed scenes: {failed}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--gpus', help='launch one independent scene-sharded worker per GPU')
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-frames', type=int, choices=(20, 60, 180), default=180)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--scenes', default='')
    args = parser.parse_args()
    if not args.gpus:
        worker(args)
        return
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    processes = []
    for i, gpu in enumerate(args.gpus.split(',')):
        command = [sys.executable, '-u', '-m', 'scripts.render_sample50',
                   '--manifest', args.manifest, '--ckpt', args.ckpt, '--out', args.out,
                   '--worker-index', str(i), '--workers', str(len(args.gpus.split(','))),
                   '--max-frames', str(args.max_frames), '--seed', str(args.seed)]
        if args.scenes:
            command += ['--scenes', args.scenes]
        with (root / f'worker_{i}.log').open('a') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu,
                                         'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
        processes.append(proc)
    atomic_json(root / 'launch.json', dict(pids=[p.pid for p in processes], gpus=args.gpus,
                                          checkpoint=args.ckpt, manifest=args.manifest))
    codes = [p.wait() for p in processes]
    atomic_json(root / 'job_status.json', dict(state='complete' if not any(codes) else 'failed',
                                              worker_exit_codes=codes))
    if any(codes):
        raise SystemExit(1)


if __name__ == '__main__':
    main()