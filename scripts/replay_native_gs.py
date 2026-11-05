"""Re-render a sample50 .gs.pt sequence using only saved native assets.

No source RGB, LiDAR, depth model, or original training checkpoint is required.
Keep the scene chunks, sequence index, and root rgb_refiner.pt together.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from pointforward.models import InstanceBoxes, TargetViews
from pointforward.models.pointforward import Gaussians, PointForward
from pointforward.models.renderer import FeatureToRGB
from scripts.render_multiscene import to_uint8
from scripts.render_sample50 import Encoder, TARGET_KEYS, selected_mask, subset_gaussians


class NativeRenderer(PointForward):
    """Only the saved feature-to-RGB stage is needed to render decoded GS."""

    def __init__(self, state):
        nn.Module.__init__(self)
        self.refiner = FeatureToRGB(in_channels=128)
        self.refiner.load_state_dict(state, strict=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=0, help='optional verification frame limit')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda')
    asset_path = Path(args.asset).resolve()
    asset = torch.load(asset_path, map_location='cpu', weights_only=False)
    root = (asset_path.parent / asset['root_relative_to_asset']).resolve()
    refiner = torch.load(root / asset['refiner'], map_location='cpu', weights_only=False)
    assert refiner['fingerprint'] == asset['fingerprint']
    renderer = NativeRenderer(refiner['state_dict']).to(device).eval()
    height, width = asset['resolution']
    encoder = Encoder(args.out, height, width, asset['fps'])
    count = 0
    expected = min(args.limit, len(asset['frames'])) if args.limit else len(asset['frames'])
    try:
        for path in asset['chunks']:
            chunk = torch.load(root / path, map_location='cpu', weights_only=False)
            assert chunk['fingerprint'] == asset['fingerprint']
            gs = Gaussians(**{k: ({a: b.to(device) for a, b in v.items()} if k == 'extras'
                                  else v.to(device)) for k, v in chunk['gaussians'].items()})
            boxes = (InstanceBoxes(**{k: v.to(device) for k, v in chunk['boxes'].items()})
                     if chunk['boxes'] else None)
            mask = selected_mask(gs, chunk['global_instance_keys'], asset['remove_global_instance_ids'])
            if bool(mask.any()):
                gs = subset_gaussians(gs, ~mask)
            targets = chunk['targets']
            camera_slot = chunk['cameras'].index(asset['camera'])
            for slot, frame in enumerate(chunk['frames']):
                if count == expected:
                    break
                assert frame == asset['frames'][count]
                index = slot * len(chunk['cameras']) + camera_slot
                values = {k: targets[k][index:index + 1].to(device) for k in TARGET_KEYS}
                lateral = asset['lateral_ego_left_m']
                if lateral:
                    values['cam_to_world'] = values['cam_to_world'].clone()
                    values['cam_to_world'][0, :3, 3] += chunk['ego_left'][slot].to(device) * lateral
                result = renderer.render(gs, boxes, TargetViews(**values), height=height, width=width)
                encoder.write(to_uint8(result['rgb'][0]))
                count += 1
            del gs, boxes, chunk
            if count == expected:
                break
        assert count == expected
        encoder.finish(count)
    finally:
        encoder.abort()
    print(f'Saved {count} frames: {args.out}', flush=True)


if __name__ == '__main__':
    main()