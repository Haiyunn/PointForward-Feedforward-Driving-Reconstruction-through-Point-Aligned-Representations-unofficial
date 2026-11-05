"""Enumerate usable scenes and write a train/val split.

Splitting is by *scene*, never by clip: clips from one scene overlap heavily in
both viewpoint and content, so a clip-level split would leak the validation set
into training and report meaningless generalisation numbers.

Usage:
    python -m scripts.make_split --root /mnt/data/dataset/notr/processed/training \
        --out splits/waymo_ds.json --val-frac 0.15
"""

from __future__ import annotations

import argparse
import json
import os

from pointforward.data import WaymoDriveStudioClips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--require-lidar", action="store_true", default=True)
    ap.add_argument("--no-require-lidar", dest="require_lidar", action="store_false")
    ap.add_argument("--require-instances", action="store_true", default=True)
    ap.add_argument("--no-require-instances", dest="require_instances", action="store_false")
    args = ap.parse_args()

    ds = WaymoDriveStudioClips(
        root=args.root,
        require_lidar=args.require_lidar,
        require_instances=args.require_instances,
        load_lidar=False,          # scanning only; no need to project sweeps
        resolution=(64, 96),
    )
    names = sorted(s["name"] for s in ds.scenes)
    total = len(names)
    n_val = max(1, round(total * args.val_frac))

    # Deterministic interleaved pick, so the split does not depend on an RNG and
    # val scenes are spread across the (numerically ordered) scene list rather
    # than clustered at one end.
    stride = max(total // n_val, 1)
    val = [names[i] for i in range(0, total, stride)][:n_val]
    train = [n for n in names if n not in set(val)]

    split = {
        "root": args.root,
        "require_lidar": args.require_lidar,
        "require_instances": args.require_instances,
        "train": train,
        "val": val,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(split, fh, indent=2)

    print(f"usable scenes: {total}")
    print(f"  train: {len(train)}")
    print(f"  val:   {len(val)}  {val}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
