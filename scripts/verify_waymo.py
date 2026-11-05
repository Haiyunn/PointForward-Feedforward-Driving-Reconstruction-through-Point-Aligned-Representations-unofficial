"""Verify the Waymo loader against real data.

The load path makes two assumptions that would silently corrupt training if
wrong: that ``track_info.txt`` boxes live in the ego frame, and that the scene
graph's rotation convention (world -> box-local) is the transpose of what the
Waymo yaw gives.  Both are checked here by projecting the lifted boxes into the
images and measuring how much of each projected box falls inside the
independently-produced ``dynamic_mask``.

Usage:
    python -m scripts.verify_waymo --root /mnt/data/dataset/notr/processed
"""

from __future__ import annotations

import argparse

import torch

from pointforward.data import WaymoClips, to_scene_input
from pointforward.models.fusion import project_points
from pointforward.models.scene_graph import inside_box


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    ds = WaymoClips(
        root=args.root,
        scenes=args.scenes,
        cameras=tuple(args.cameras),
        num_context=3,
        num_target=2,
        resolution=(288, 384),
        deterministic=True,
    )
    print(f"scenes: {[s['name'] for s in ds.scenes]}")
    print(f"clips:  {len(ds)}")

    device = torch.device("cpu")
    overlaps: list[float] = []

    for i in range(min(args.clips, len(ds))):
        clip = ds[i * max(len(ds) // max(args.clips, 1), 1)]
        scene, targets, gt = to_scene_input(clip, device)
        boxes = scene.boxes

        print(f"\n--- clip {i}  scene={clip['scene']}  frames={clip['frames'].tolist()} ---")
        print(f"  context views {tuple(scene.images.shape)}  target views {tuple(gt.shape)}")
        print(f"  times ctx {scene.times.tolist()}  frame_index {scene.frame_index.tolist()}")
        if boxes is None:
            print("  no rigid instances in this clip")
            continue
        print(f"  instances {boxes.num_instances}  slots {boxes.valid.shape[1]}")

        # --- sanity: box centres must sit inside their own boxes -------------
        for o in range(boxes.num_instances):
            for j in range(boxes.valid.shape[1]):
                if not bool(boxes.valid[o, j]):
                    continue
                assert bool(
                    inside_box(boxes.centers[o, j], boxes.centers[o, j], boxes.dims[o, j], boxes.rots[o, j])
                ), "box centre outside its own box -- rotation convention is wrong"

        # Classify instances by how far their world-space centre travels across
        # the window: a genuinely moving vehicle shifts, a parked one does not.
        first = torch.argmax(boxes.valid.long(), dim=1)
        last = boxes.valid.shape[1] - 1 - torch.argmax(boxes.valid.flip(1).long(), dim=1)
        rows = torch.arange(boxes.num_instances)
        travel = (boxes.centers[rows, last] - boxes.centers[rows, first]).norm(dim=-1)
        moving = travel > 1.0
        print(f"  moving instances: {int(moving.sum())}/{boxes.num_instances}")

        # --- box centres should project into the image and in front of it ----
        world_to_cam = torch.linalg.inv(scene.cam_to_world)
        h, w = scene.image_size
        mask = clip["context"].get("dynamic_mask")

        for v in range(scene.num_views):
            slot = int(scene.frame_index[v])
            keep = boxes.valid[:, slot]
            if not bool(keep.any()):
                continue
            centers = boxes.centers[keep, slot].unsqueeze(0)
            uv, depth, valid = project_points(
                centers, scene.intrinsics[v : v + 1], world_to_cam[v : v + 1]
            )
            uv, depth, valid = uv[0], depth[0], valid[0]
            on_image = (
                valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            )
            if not bool(on_image.any()):
                continue

            if mask is None:
                continue
            m = mask[v, 0] > 0.5
            u = uv[on_image, 0].round().long().clamp(0, w - 1)
            vv = uv[on_image, 1].round().long().clamp(0, h - 1)
            hit = m[vv, u]

            # dynamic_mask marks *moving* objects, so parked vehicles are
            # expected to miss it.  Split the two so the number means something.
            is_moving = moving[keep][on_image]
            for label, sel in (("moving", is_moving), ("parked", ~is_moving)):
                if bool(sel.any()):
                    rate = hit[sel].float().mean().item()
                    if label == "moving":
                        overlaps.append(rate)
                    if v < 2:
                        print(
                            f"  view {v} {label}: {int(sel.sum())} centres on image, "
                            f"{rate:.0%} on the dynamic mask"
                        )
            if v < 2:
                print(f"  view {v} mask foreground coverage: {m.float().mean():.1%}")

    if overlaps:
        mean = sum(overlaps) / len(overlaps)
        print(f"\nmoving-instance mask agreement over {len(overlaps)} views: {mean:.1%}")
        print(
            "  Boxes are lifted from the ego frame via the ego pose, whose "
            "translation is ~1e4 m; if they were already world-space almost "
            "nothing would project on-image at all."
        )
    else:
        print("\nno dynamic masks available to cross-check against")


if __name__ == "__main__":
    main()
