"""Evaluate PointForwardRec incrementally over consecutive windows."""

from __future__ import annotations

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from pointforward.data import collate_windows, to_scene_input
from pointforward.losses import psnr, ssim
from pointforward.variants import build_model
from pointforward.paths import add_path_args, apply_path_args
from scripts.train import build_depth
from scripts.train_rec import build_windows_dataset


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-items", type=int, default=20)
    add_path_args(parser)
    args = parser.parse_args()
    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
        apply_path_args(config, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, build_depth(config), inference=True).to(device)
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    model.eval()
    loader = DataLoader(
        build_windows_dataset(config, args.split), batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_windows,
    )
    totals = {"psnr": 0.0, "ssim": 0.0, "windows": 0}
    for index, batch in enumerate(loader):
        if index >= args.max_items:
            break
        for item in batch:
            cache = None
            for window in item["windows"]:
                scene, targets, ground_truth = to_scene_input(window, device)
                output, cache, _ = model.reconstruct_window(scene, targets, cache)
                totals["psnr"] += float(psnr(output["rgb"], ground_truth))
                totals["ssim"] += float(ssim(output["rgb"], ground_truth))
                totals["windows"] += 1
    count = max(totals["windows"], 1)
    print(f"PSNR {totals['psnr'] / count:.3f} SSIM {totals['ssim'] / count:.4f} windows {totals['windows']}")


if __name__ == "__main__":
    main()