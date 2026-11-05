"""Validate the frozen InfiniDepth wrapper against the real checkpoint.

The wrapper has to match InfiniDepth's actual API, not a guess at it: the
implicit head is queried at continuous (y, x) coordinates in [-1, 1] with a
pixel-centre convention, and the depth-sensor variant takes an extra
``prompt_depth`` whose expected shape is the thing most likely to be wrong.

Checks:
  * the model builds and stays frozen (no parameter requires grad, eval mode
    survives a parent .train() call)
  * dense prediction returns (V, H, W) with positive, finite, plausible depths
  * arbitrary output resolution works, since depth is an implicit field
  * the LiDAR prompt actually drives the predicted metric scale

On that last point: ``InfiniDepth_DepthSensor`` does *not* condition densely
on the LiDAR prompt. ``InfiniDepth/utils/warp_utils.py:WarpMedian.warp`` takes
only the *median* of ``prompt_depth[prompt_mask > 0]`` and uses it as a single
global scale factor (predict on ``depth / median``, then multiply back). So
splitting a LiDAR sweep into two random halves and prompting with each gives
almost the same median once there are more than a handful of points -- "denser
prompting doesn't reduce held-out error" is the *expected* behaviour of a
median-based global rescale, not evidence of a shape/unit bug. The meaningful
check is whether the prompt's scale reaches the output at all: scale
``prompt_depth`` by a known factor and confirm the output median scales with
it.

Usage:
    python -m scripts.check_depth --config configs/waymo_ds.yaml
    python -m scripts.check_depth --config configs/waymo_ds.yaml --real-clip
"""

from __future__ import annotations

import argparse

import torch
import yaml

from pointforward.models import FrozenInfiniDepth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--real-clip", action="store_true", help="use a real clip instead of noise")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    print("building InfiniDepth ...")
    model = FrozenInfiniDepth(
        repo_root=cfg["depth"]["repo_root"],
        checkpoint=cfg["depth"]["checkpoint"],
        variant=cfg["depth"].get("variant", "InfiniDepth_DepthSensor"),
        encoder=cfg["depth"].get("encoder", "vitl16"),
    )
    print(f"  variant: {model.variant}")

    n_params = sum(p.numel() for p in model.parameters())
    n_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params {n_params/1e6:.1f}M, requiring grad: {n_grad}")
    assert n_grad == 0, "depth model must be fully frozen"

    # A parent .train() must not put the depth model back into training mode.
    model.train()
    assert not model.model.training, "depth model left training mode"
    print("  stays in eval mode after .train()  OK")

    device = torch.device("cuda")
    sparse_depth = sparse_mask = None

    if args.real_clip:
        from pointforward.data import WaymoDriveStudioClips, to_scene_input

        ds = WaymoDriveStudioClips(
            root=cfg["data"]["root"],
            cameras=tuple(cfg["data"].get("cameras", (0, 1, 2))),
            num_context=cfg["data"].get("num_context", 4),
            num_target=1,
            resolution=(args.height, args.width),
            require_lidar=True,
            require_instances=True,
            deterministic=True,
        )
        scene, _, _ = to_scene_input(ds[0], device)
        images = scene.images[: args.views]
        if scene.sparse_depth is not None:
            sparse_depth = scene.sparse_depth[: args.views]
            sparse_mask = scene.sparse_mask[: args.views]
        print(f"  using a real clip from scene {ds.scenes[0]['name']}")
    else:
        images = torch.rand(args.views, 3, args.height, args.width, device=device)

    print(f"\ninput images {tuple(images.shape)}")

    # InfiniDepth_DepthSensor hard-requires a LiDAR prompt (see depth.py) --
    # it cannot make an unprompted call at all.
    depth_sensor_needs_prompt = (
        model.variant == "InfiniDepth_DepthSensor" and sparse_depth is not None
    )

    # --- dense prediction at the input resolution -------------------------
    if depth_sensor_needs_prompt:
        depth = model(images, sparse_depth=sparse_depth, sparse_mask=sparse_mask)
    else:
        depth = model(images)
    print(f"depth {tuple(depth.shape)} dtype={depth.dtype}")
    assert depth.shape == (images.shape[0], args.height, args.width), depth.shape
    assert torch.isfinite(depth).all(), "non-finite depth"
    print(
        f"  range {float(depth.min()):.2f} - {float(depth.max()):.2f} m, "
        f"median {float(depth.median()):.2f} m"
    )
    assert float(depth.min()) > 0, "depth must be positive"

    # --- arbitrary output resolution --------------------------------------
    h2, w2 = args.height + 37, args.width + 53   # deliberately not a round multiple
    if depth_sensor_needs_prompt:
        depth2 = model(images, height=h2, width=w2, sparse_depth=sparse_depth, sparse_mask=sparse_mask)
    else:
        depth2 = model(images, height=h2, width=w2)
    assert depth2.shape == (images.shape[0], h2, w2), depth2.shape
    print(f"  off-grid resolution {h2}x{w2} works (implicit field)  OK")

    # --- LiDAR prompting: global metric-scale correction -------------------
    if depth_sensor_needs_prompt:
        cov = float((sparse_mask > 0.5).float().mean())
        print(f"\nprompt_depth {tuple(sparse_depth.shape)}, {cov:.1%} LiDAR coverage")

        base = depth  # already prompted with the real sparse_depth/sparse_mask above
        scale = 1.7
        scaled = model(images, sparse_depth=sparse_depth * scale, sparse_mask=sparse_mask)

        base_med = float(base.median())
        scaled_med = float(scaled.median())
        ratio = scaled_med / base_med
        print(f"  base median {base_med:.2f} m -> scaled-prompt median {scaled_med:.2f} m"
              f"  (ratio {ratio:.2f}, expected ~{scale})")
        assert abs(ratio - scale) < 0.15 * scale, (
            "output did not track the prompt's global scale -- check the prompt_depth"
            " units/shape (see WarpMedian.warp: it rescales by the prompt's median)"
        )
        print("  LiDAR prompt drives the global metric scale  OK")

        m = (sparse_mask.squeeze(1) > 0.5)
        if bool(m.any()):
            err = (base[m] - sparse_depth.squeeze(1)[m]).abs().median()
            print(f"  median |prompted depth - lidar| at prompt points: {float(err):.3f} m")
    elif sparse_depth is not None:
        print("\n(plain InfiniDepth variant -- LiDAR prompting check only applies to"
              " InfiniDepth_DepthSensor)")
    else:
        print("\nno LiDAR available for prompting (pass --real-clip on a lidar scene)")

    print("\nall interface checks passed")


if __name__ == "__main__":
    main()
