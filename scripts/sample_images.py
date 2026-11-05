"""Save rendered vs. ground-truth image comparisons for visual inspection.

Not part of the paper quantitative pipeline -- this is a debugging/inspection
aid so a human can look at reconstruction quality, complementing scripts/eval.py.

Usage:
    python -m scripts.sample_images --config configs/waymo_ds_run1.yaml \
        --ckpt runs/waymo_ds_run1/ckpt_002250.pt --out runs/waymo_ds_run1/samples_2250 \
        --num-clips 6
"""

from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from pointforward.data import collate_clip, to_scene_input

from scripts.eval import build_model
from scripts.train import build_dataset


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--protocol", choices=("interp", "context"), default="interp")
    ap.add_argument("--num-clips", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(cfg, args.split)
    print(f"{args.split}: {len(dataset)} clips over {len(dataset.scenes)} scenes")

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_clip)
    model = build_model(cfg, device, args.ckpt)

    os.makedirs(args.out, exist_ok=True)

    saved = 0
    for batch in loader:
        if saved >= args.num_clips:
            break
        for clip in batch:
            if saved >= args.num_clips:
                break
            scene, targets, gt = to_scene_input(clip, device)

            if args.protocol == "context":
                from pointforward.models import TargetViews

                targets = TargetViews(
                    intrinsics=scene.intrinsics,
                    cam_to_world=scene.cam_to_world,
                    times=scene.times,
                    frame_index=scene.frame_index,
                )
                gt = scene.images

            out = model(scene, targets, render_depth=False)
            pred = out["rgb"].clamp(0, 1)

            v = pred.shape[0]
            for view in range(v):
                pair = torch.cat((gt[view], pred[view]), dim=1)
                save_image(pair, os.path.join(args.out, f"clip{saved:02d}_view{view:02d}_gt-top_pred-bottom.png"))

            saved += 1
            print(f"  saved clip {saved}/{args.num_clips} ({v} views)")

    print(f"done, {saved} clips written to {args.out}")


if __name__ == "__main__":
    main()
