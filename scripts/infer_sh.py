"""Inference and rendering for fixed- and variable-view SH variants."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from pointforward.data import collate_clip, to_scene_input
from pointforward.losses import depth_rmse, psnr, ssim
from pointforward.models import TargetViews
from pointforward.variants import build_model as build_variant_model
from scripts.train import build_dataset, build_depth


def build_model_sh(config: dict, device: torch.device, checkpoint_path: str | None):
    """Construct the configured SH model and optionally load a checkpoint."""
    model = build_variant_model(config, build_depth(config), inference=True).to(device)
    if hasattr(model, "sh_degree"):
        print(f"SH degree {model.sh_degree}  ->  {model.decoder.num_sh} coeffs/channel")

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint.get("model", checkpoint)
        current = model.state_dict()
        incompatible = [
            key for key, value in state.items()
            if key in current and current[key].shape != value.shape
        ]
        filtered = {key: value for key, value in state.items() if key not in incompatible}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        missing = [key for key in missing if not key.startswith("depth_model.")]
        if incompatible:
            print(f"warning: {len(incompatible)} shape-incompatible keys skipped")
        if missing:
            print(f"warning: {len(missing)} missing keys, e.g. {missing[:3]}")
        if unexpected:
            print(f"warning: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    model.eval()
    return model


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.clamp(0, 1).detach().cpu().numpy()
    return (array.transpose(1, 2, 0) * 255).astype(np.uint8)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="the run's config.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--protocol", choices=("interp", "context"), default="interp")
    parser.add_argument("--max-clips", type=int, default=20)
    parser.add_argument("--out", default=None, help="directory for rendered PNGs and metrics.json")
    parser.add_argument("--no-images", action="store_true", help="compute metrics without PNGs")
    parser.add_argument(
        "--side-by-side", action="store_true",
        help="save prediction and ground truth side by side",
    )
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.setdefault("data", {})["target_lidar"] = True
    dataset = build_dataset(config, args.split)
    print(f"{args.split}: {len(dataset)} clips over {len(dataset.scenes)} scenes")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_clip,
    )
    model = build_model_sh(config, device, args.ckpt)

    if args.out and not args.no_images:
        os.makedirs(args.out, exist_ok=True)

    sums = {"psnr": 0.0, "ssim": 0.0, "drmse": 0.0}
    counts = {"psnr": 0, "ssim": 0, "drmse": 0}
    for index, batch in enumerate(loader):
        if index >= args.max_clips:
            break
        for clip in batch:
            scene, targets, ground_truth = to_scene_input(clip, device)
            if args.protocol == "context":
                targets = TargetViews(
                    intrinsics=scene.intrinsics, cam_to_world=scene.cam_to_world,
                    times=scene.times, frame_index=scene.frame_index,
                )
                ground_truth = scene.images

            want_depth = "sparse_depth" in clip["target"] or args.protocol == "context"
            output = model(scene, targets, render_depth=want_depth)
            prediction = output["rgb"].clamp(0, 1)
            sums["psnr"] += float(psnr(prediction, ground_truth))
            sums["ssim"] += float(ssim(prediction, ground_truth))
            counts["psnr"] += 1
            counts["ssim"] += 1

            reference = clip["context"] if args.protocol == "context" else clip["target"]
            if "render_depth" in output and "sparse_depth" in reference:
                ground_truth_depth = reference["sparse_depth"].to(device)
                ground_truth_mask = reference["sparse_mask"].to(device) > 0.5
                if bool(ground_truth_mask.any()):
                    sums["drmse"] += float(depth_rmse(
                        output["render_depth"], ground_truth_depth, ground_truth_mask
                    ))
                    counts["drmse"] += 1

            if args.out and not args.no_images:
                for view in range(prediction.shape[0]):
                    image = _to_uint8(prediction[view])
                    if args.side_by_side:
                        image = np.concatenate((image, _to_uint8(ground_truth[view])), axis=1)
                    filename = f"{clip['scene']}_clip{index:04d}_view{view:02d}.png"
                    Image.fromarray(image).save(os.path.join(args.out, filename))

        if (index + 1) % 10 == 0:
            completed = counts["psnr"]
            print(f"  {completed} clips: PSNR {sums['psnr']/max(completed, 1):.2f}", flush=True)

    metrics = {key: sums[key] / counts[key] for key in sums if counts[key]}
    metrics.update(
        clips=counts["psnr"], protocol=args.protocol, split=args.split, ckpt=args.ckpt
    )
    print("\n" + "=" * 50)
    for key, value in metrics.items():
        print(f"  {key:10s} {value}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        metrics_path = os.path.join(args.out, "metrics.json")
        with open(metrics_path, "w") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"metrics -> {metrics_path}")
        if not args.no_images:
            print(f"images  -> {args.out}/")


if __name__ == "__main__":
    main()