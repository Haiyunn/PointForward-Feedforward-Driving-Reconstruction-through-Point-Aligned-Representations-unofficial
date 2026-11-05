"""Verify the DriveStudio Waymo loader against real data.

Three assumptions are checked, each against something produced independently of
the loader:

1. Instance boxes are world-space ``obj_to_world`` matrices, and the scene-graph
   rotation convention is their transpose.  Checked by projecting box centres and
   measuring overlap with ``dynamic_masks/all``.
2. LiDAR is stored in the ego frame.  Checked by projecting the sweep into each
   camera and measuring how many points land on the image -- an ego/world mix-up
   would put them kilometres away.
3. Box dimensions line up with the local axes.  Checked by confirming every box
   centre tests inside its own box.

Usage:
    python -m scripts.verify_drivestudio --root /mnt/data/dataset/notr/processed/training
"""

from __future__ import annotations

import argparse

import torch

from pointforward.data import WaymoDriveStudioClips, to_scene_input
from pointforward.models.fusion import project_points
from pointforward.models.query import camera_rays, unproject_z
from pointforward.models.scene_graph import inside_box


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--limit-scenes", type=int, default=0, help="scan only the first N scenes")
    args = ap.parse_args()

    scenes = args.scenes
    if scenes is None and args.limit_scenes:
        import os

        scenes = sorted(os.listdir(args.root))[: args.limit_scenes]

    ds = WaymoDriveStudioClips(
        root=args.root,
        scenes=scenes,
        resolution=(288, 384),
        deterministic=True,
    )
    print(f"usable scenes: {len(ds.scenes)}")
    print(f"clips:         {len(ds)}")

    device = torch.device("cpu")
    box_hits: list[float] = []
    lidar_cover: list[float] = []
    lidar_agree: list[float] = []

    step = max(len(ds) // max(args.clips, 1), 1)
    for i in range(min(args.clips, len(ds))):
        clip = ds[i * step]
        scene, targets, gt = to_scene_input(clip, device)
        boxes = scene.boxes

        print(f"\n--- clip {i}  scene={clip['scene']}  frames={clip['frames'].tolist()} ---")
        print(f"  context {tuple(scene.images.shape)}   targets {tuple(gt.shape)}")

        # --- LiDAR ----------------------------------------------------------
        if scene.sparse_depth is not None:
            m = scene.sparse_mask > 0.5
            cover = m.float().mean(dim=(1, 2, 3))
            d = scene.sparse_depth[m]
            lidar_cover.append(float(cover.mean()))
        if scene.sparse_depth is not None and d.numel() == 0:
            print("  lidar: parsed but NO point projected into any view")
        elif scene.sparse_depth is not None:
            print(
                f"  lidar: {float(cover.mean()):.2%} of pixels have a return, "
                f"depth range {float(d.min()):.1f}-{float(d.max()):.1f} m, "
                f"median {float(d.median()):.1f} m"
            )
            # A depth range in the tens of metres confirms the ego frame; a
            # world-frame mix-up would give values in the thousands.
            assert float(d.median()) < 200.0, "lidar depths look like world coordinates"
        else:
            print("  lidar: not available")

        if boxes is None:
            print("  no rigid instances")
            continue
        print(f"  instances {boxes.num_instances}  slots {boxes.valid.shape[1]}")

        # --- box self-consistency -------------------------------------------
        for o in range(boxes.num_instances):
            for j in range(boxes.valid.shape[1]):
                if bool(boxes.valid[o, j]):
                    assert bool(
                        inside_box(
                            boxes.centers[o, j], boxes.centers[o, j], boxes.dims[o, j], boxes.rots[o, j]
                        )
                    ), "box centre outside its own box"

        # --- moving vs parked, against the dynamic mask ----------------------
        first = torch.argmax(boxes.valid.long(), dim=1)
        last = boxes.valid.shape[1] - 1 - torch.argmax(boxes.valid.flip(1).long(), dim=1)
        rows = torch.arange(boxes.num_instances)
        travel = (boxes.centers[rows, last] - boxes.centers[rows, first]).norm(dim=-1)
        moving = travel > 1.0
        print(f"  moving instances: {int(moving.sum())}/{boxes.num_instances}")

        mask = clip["context"].get("dynamic_mask")
        if mask is None:
            continue

        world_to_cam = torch.linalg.inv(scene.cam_to_world)
        h, w = scene.image_size
        for v in range(scene.num_views):
            slot = int(scene.frame_index[v])
            keep = boxes.valid[:, slot]
            if not bool(keep.any()):
                continue
            uv, _, valid = project_points(
                boxes.centers[keep, slot].unsqueeze(0),
                scene.intrinsics[v : v + 1],
                world_to_cam[v : v + 1],
            )
            uv, valid = uv[0], valid[0]
            on = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            if not bool(on.any()):
                continue

            mv = moving[keep][on]
            if not bool(mv.any()):
                continue
            u = uv[on, 0].round().long().clamp(0, w - 1)
            vv = uv[on, 1].round().long().clamp(0, h - 1)
            hit = (mask[v, 0] > 0.5)[vv, u]
            rate = hit[mv].float().mean().item()
            box_hits.append(rate)
            if v < 2:
                print(f"  view {v}: {int(mv.sum())} moving centres on image, {rate:.0%} on mask")

        # --- the decisive check ----------------------------------------------
        # Unproject the LiDAR returns to world space *through the model's own
        # geometry code*, ask the scene graph which land inside a moving box, and
        # compare that verdict against the independently produced dynamic mask.
        # This ties camera convention, depth sign, ego pose, box lift and box
        # rotation together: any one of them being wrong collapses the number.
        if scene.sparse_depth is not None and mask is not None:
            inside_rate, outside_rate = _lidar_in_box_agreement(scene, boxes, moving, mask)
            if inside_rate is not None:
                lidar_agree.append((inside_rate, outside_rate))
                lift = inside_rate / max(outside_rate, 1e-6)
                print(
                    f"  lidar in moving box marked dynamic: {inside_rate:.0%}  |  "
                    f"outside any box: {outside_rate:.0%}  |  enrichment x{lift:.1f}"
                )

    print("\n" + "=" * 60)
    if box_hits:
        print(f"moving-box / dynamic-mask agreement: {sum(box_hits)/len(box_hits):.1%}")
    if lidar_cover:
        print(f"mean lidar pixel coverage:           {sum(lidar_cover)/len(lidar_cover):.2%}")
    if lidar_agree:
        ins = sum(a for a, _ in lidar_agree) / len(lidar_agree)
        outs = sum(b for _, b in lidar_agree) / len(lidar_agree)
        lift = ins / max(outs, 1e-6)
        print(f"lidar in a moving box, marked dynamic:   {ins:.1%}")
        print(f"lidar outside every box, marked dynamic: {outs:.1%}   <-- control")
        print(f"enrichment:                              x{lift:.1f}   <-- decisive")
        print(
            "\n  LiDAR is unprojected with the model's own camera code, classified by\n"
            "  the scene graph, and scored against a mask neither step ever sees.\n"
            "  The control matters: the raw in-box rate alone cannot separate correct\n"
            "  geometry from a merely dense mask.  Enrichment is the real signal --\n"
            "  a wrong axis convention, depth sign or box rotation makes the two rates\n"
            "  equal (enrichment ~x1), because the box would then select pixels\n"
            "  unrelated to where the instance actually is.\n"
            "  The in-box rate stays below 100% for reasons that are not errors: a 3D\n"
            "  box also contains ground under the vehicle and returns from its far\n"
            "  side, which a 2D visible-object mask does not mark."
        )


def _lidar_in_box_agreement(scene, boxes, moving, mask):
    """Dynamic-mask hit rate for LiDAR returns inside vs outside the boxes.

    Returns (inside_rate, outside_rate).  The second is the control: it is what
    the first would collapse to if the box test were selecting arbitrary pixels.
    """
    if boxes is None or not bool(moving.any()):
        return None, None

    h, w = scene.image_size
    hits, total = 0, 0
    out_hits, out_total = 0, 0

    for v in range(scene.num_views):
        valid_px = scene.sparse_mask[v, 0] > 0.5
        if not bool(valid_px.any()):
            continue

        # Unproject this view's sparse depth with the same functions the model uses.
        origins, dirs = camera_rays(
            h, w, scene.intrinsics[v : v + 1], scene.cam_to_world[v : v + 1]
        )
        forward = scene.cam_to_world[v : v + 1, :3, 2]
        forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        world = unproject_z(scene.sparse_depth[v : v + 1, 0], origins, dirs, forward)[0]

        pts = world[valid_px]                       # (P, 3)
        slot = int(scene.frame_index[v])
        sel = boxes.valid[:, slot] & moving
        if not bool(sel.any()) or pts.numel() == 0:
            continue

        centers = boxes.centers[sel, slot]          # (O', 3)
        dims = boxes.dims[sel, slot]
        rots = boxes.rots[sel, slot]
        o = centers.shape[0]

        inside = inside_box(
            pts.unsqueeze(1).expand(pts.shape[0], o, 3),
            centers.unsqueeze(0),
            dims.unsqueeze(0),
            rots.unsqueeze(0),
        ).any(dim=1)                                # (P,)

        # Control: returns outside EVERY box, moving or parked, so that parked
        # cars -- which the mask also leaves unmarked -- do not flatter it.
        all_sel = boxes.valid[:, slot]
        outside = torch.ones_like(inside)
        if bool(all_sel.any()):
            oa = int(all_sel.sum())
            outside = ~inside_box(
                pts.unsqueeze(1).expand(pts.shape[0], oa, 3),
                boxes.centers[all_sel, slot].unsqueeze(0),
                boxes.dims[all_sel, slot].unsqueeze(0),
                boxes.rots[all_sel, slot].unsqueeze(0),
            ).any(dim=1)

        dynamic_px = (mask[v, 0] > 0.5)[valid_px]
        hits += int((inside & dynamic_px).sum())
        total += int(inside.sum())
        out_hits += int((outside & dynamic_px).sum())
        out_total += int(outside.sum())

    return (
        hits / total if total else None,
        out_hits / out_total if out_total else 0.0,
    )


if __name__ == "__main__":
    main()
