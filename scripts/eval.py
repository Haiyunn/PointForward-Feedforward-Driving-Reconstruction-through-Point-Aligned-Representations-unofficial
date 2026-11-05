"""Evaluation: PSNR, SSIM, D-RMSE and (optionally) FID.

Two protocols, matching the paper:

* ``interp``  -- context frames are inputs, the model renders the *remaining*
  frames of the clip.  This is the novel-view-synthesis setting of Table 1.
* ``context`` -- the model renders the context frames themselves, the
  reconstruction setting of Table 2.

D-RMSE compares the rasterised expected depth against the LiDAR sweep projected
into the target view, scored only on pixels that actually have a return.  FID is
reported for the extrapolated-view experiments and needs many samples to mean
anything, so it is off unless asked for.

Usage:
    python -m scripts.eval --config configs/waymo_ds.yaml --ckpt runs/waymo_ds/ckpt_045000.pt
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import yaml
from torch.utils.data import DataLoader

from pointforward.data import collate_clip, to_scene_input
from pointforward.losses import depth_rmse, psnr, ssim
from pointforward.models import PointForward
from pointforward.variants import build_model as build_variant_model

from scripts.train import build_dataset, build_depth


def build_model(cfg: dict, device: torch.device, ckpt_path: str | None) -> PointForward:
    model = build_variant_model(cfg, build_depth(cfg), inference=True).to(device)

    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)["model"]
        # The frozen depth prior is not stored in the checkpoint.
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("depth_model.")]
        if missing:
            print(f"warning: {len(missing)} missing keys, e.g. {missing[:3]}")
        if unexpected:
            print(f"warning: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--protocol", choices=("interp", "context"), default="interp")
    ap.add_argument("--max-clips", type=int, default=200)
    ap.add_argument("--fid", action="store_true", help="also compute FID (needs many clips)")
    ap.add_argument(
        "--lpips",
        dest="lpips",
        action="store_true",
        default=True,
        help="also emit LPIPS via pointforward.metrics_cvpr (on by default)",
    )
    ap.add_argument(
        "--no-lpips",
        dest="lpips",
        action="store_false",
        help="disable the LPIPS metric",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # D-RMSE scores rendered depth against the *target* view's LiDAR, which the
    # training loader does not bother to project.
    cfg.setdefault("data", {})["target_lidar"] = True
    dataset = build_dataset(cfg, args.split)
    print(f"{args.split}: {len(dataset)} clips over {len(dataset.scenes)} scenes")

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_clip
    )
    model = build_model(cfg, device, args.ckpt)

    fid = None
    if args.fid:
        from torchmetrics.image.fid import FrechetInceptionDistance

        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # LPIPS is additive and optional: a failure to import the shared harness
    # leaves every existing metric untouched.
    score_frames = None
    if args.lpips:
        try:
            from pointforward.metrics_cvpr import score_frames
        except Exception as exc:  # noqa: BLE001
            print(f"warning: --lpips requested but metrics_cvpr unavailable ({exc}); continuing without")
            score_frames = None

    sums = {"psnr": 0.0, "ssim": 0.0, "drmse": 0.0}
    counts = {"psnr": 0, "ssim": 0, "drmse": 0}
    # Separate LPIPS accumulators so the base metrics dict is unchanged when
    # LPIPS is off or unavailable.
    lp_sums = {"lpips": 0.0, "lpips_non_sky": 0.0, "lpips_dynamic": 0.0}
    lp_counts = {"lpips": 0, "lpips_non_sky": 0, "lpips_dynamic": 0}

    for i, batch in enumerate(loader):
        if i >= args.max_clips:
            break
        for clip in batch:
            scene, targets, gt = to_scene_input(clip, device)

            if args.protocol == "context":
                # Render the input views themselves (Table 2's setting).
                from pointforward.models import TargetViews

                targets = TargetViews(
                    intrinsics=scene.intrinsics,
                    cam_to_world=scene.cam_to_world,
                    times=scene.times,
                    frame_index=scene.frame_index,
                )
                gt = scene.images

            want_depth = "sparse_depth" in clip["target"] or args.protocol == "context"
            out = model(scene, targets, render_depth=want_depth)
            pred = out["rgb"].clamp(0, 1)

            sums["psnr"] += float(psnr(pred, gt)); counts["psnr"] += 1
            sums["ssim"] += float(ssim(pred, gt)); counts["ssim"] += 1

            # D-RMSE against the LiDAR sweep of whichever views we rendered.
            ref = clip["context"] if args.protocol == "context" else clip["target"]
            if "render_depth" in out and "sparse_depth" in ref:
                gt_depth = ref["sparse_depth"].to(device)
                gt_mask = ref["sparse_mask"].to(device) > 0.5
                if bool(gt_mask.any()):
                    sums["drmse"] += float(depth_rmse(out["render_depth"], gt_depth, gt_mask))
                    counts["drmse"] += 1

            if fid is not None:
                fid.update(gt.clamp(0, 1), real=True)
                fid.update(pred, real=False)

            # LPIPS (plus optional non-sky / dynamic-region LPIPS) via the
            # shared harness; wrapped so any failure leaves the loop intact.
            if score_frames is not None:
                try:
                    src = clip["context"] if args.protocol == "context" else clip["target"]
                    sky_mask = src.get("sky_mask") if isinstance(src, dict) else None
                    dyn_mask = src.get("dynamic_mask") if isinstance(src, dict) else None
                    scored = score_frames(
                        pred, gt, sky_mask=sky_mask, dyn_bbox_mask=dyn_mask, device=device
                    )
                    if scored["full"]["lpips"] is not None:
                        lp_sums["lpips"] += scored["full"]["lpips"]; lp_counts["lpips"] += 1
                    for cat, key in (("non_sky", "lpips_non_sky"), ("dynamic_bbox", "lpips_dynamic")):
                        region = scored[cat]
                        if region is not None and region["lpips"] is not None:
                            lp_sums[key] += region["lpips"]; lp_counts[key] += 1
                except Exception as exc:  # noqa: BLE001
                    if counts["psnr"] == 1:  # warn once
                        print(f"warning: LPIPS scoring failed ({exc}); continuing without")
                    score_frames = None

        if (i + 1) % 20 == 0:
            done = counts["psnr"]
            print(f"  {done} clips: PSNR {sums['psnr']/max(done,1):.2f}")

    metrics = {k: sums[k] / counts[k] for k in sums if counts[k]}
    metrics["clips"] = counts["psnr"]
    metrics["protocol"] = args.protocol
    if fid is not None:
        metrics["fid"] = float(fid.compute())
    # Merge LPIPS averages in additively; keys appear only when values exist.
    for k in ("lpips", "lpips_non_sky", "lpips_dynamic"):
        if lp_counts[k]:
            metrics[k] = lp_sums[k] / lp_counts[k]

    print("\n" + "=" * 50)
    for k, v in metrics.items():
        print(f"  {k:10s} {v}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
