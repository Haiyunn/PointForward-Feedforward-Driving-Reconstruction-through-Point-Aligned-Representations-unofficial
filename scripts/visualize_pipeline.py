"""Visualize every intermediate product of the PointForward pipeline for one clip.

Not part of the paper quantitative pipeline -- a debugging/inspection aid, in the
same spirit as scripts/sample_images.py, but instead of only comparing the final
RGB against ground truth, this dumps the full chain for a single context view:

    input RGB -> InfiniDepth depth -> back-projected 3D queries
              -> scene-graph static/dynamic split (BEV)
              -> decoded + posed Gaussians (BEV)
              -> pre-refiner rasterized features/alpha
              -> post-refiner RGB

matplotlib is not installed in this environment, so all colormaps and scatter
plots below are hand-rolled with numpy + PIL only.

Usage:
    python -m scripts.visualize_pipeline --config configs/waymo_ds_run1.yaml \
        --ckpt runs/waymo_ds_run1/ckpt_002250.pt --out runs/waymo_ds_run1/pipeline_vis
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from pointforward.data import collate_clip, to_scene_input
from pointforward.models import TargetViews

from scripts.eval import build_model
from scripts.train import build_dataset

PALETTE = np.array(
    [
        [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
        [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
    ],
    dtype=np.uint8,
)


def to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    arr = img_chw.clamp(0, 1).detach().cpu().numpy()
    return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)


def colorize_depth(depth: torch.Tensor) -> np.ndarray:
    """Rainbow (near=red, far=blue) colormap, hand-rolled since matplotlib is absent."""
    d = depth.detach().cpu().numpy().astype(np.float32)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    t = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    hue = t * 0.66  # 0=red (near) -> 0.66=blue (far)
    h6 = hue * 6.0
    c = 1.0
    x = c * (1 - np.abs(np.mod(h6, 2) - 1))
    z = np.zeros_like(h6)
    conds = [
        (h6 < 1), (h6 < 2), (h6 < 3), (h6 < 4), (h6 < 5), (h6 <= 6),
    ]
    r = np.select(conds, [c, x, z, z, x, c], default=c)
    g = np.select(conds, [x, c, c, x, z, z], default=z)
    b = np.select(conds, [z, z, x, c, c, x], default=x)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def colorize_alpha(alpha: torch.Tensor) -> np.ndarray:
    a = alpha.detach().cpu().numpy().astype(np.float32)
    a = np.clip(a, 0.0, 1.0)
    gray = (a * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def pca_pseudo_rgb(features: torch.Tensor) -> np.ndarray:
    """Project a (C, H, W) feature map to pseudo-RGB via its top-3 principal components."""
    c, h, w = features.shape
    x = features.detach().permute(1, 2, 0).reshape(-1, c).float()
    x = x - x.mean(dim=0, keepdim=True)
    try:
        u, s, v = torch.pca_lowrank(x, q=3)
        proj = x @ v[:, :3]
    except Exception:
        proj = x[:, :3]
    proj = proj.cpu().numpy()
    lo = np.percentile(proj, 1, axis=0)
    hi = np.percentile(proj, 99, axis=0)
    proj = np.clip((proj - lo) / np.clip(hi - lo, 1e-6, None), 0.0, 1.0)
    return (proj.reshape(h, w, 3) * 255).astype(np.uint8)


def instance_colors(instance_id: torch.Tensor) -> np.ndarray:
    """Static (-1) -> light gray; dynamic instances -> a distinct palette color each."""
    iid = instance_id.detach().cpu().numpy()
    colors = np.full((iid.shape[0], 3), 180, dtype=np.uint8)
    dyn = iid >= 0
    if dyn.any():
        colors[dyn] = PALETTE[iid[dyn] % len(PALETTE)]
    return colors


def splat_points(canvas: np.ndarray, px: np.ndarray, py: np.ndarray, colors: np.ndarray, radius: int = 1) -> None:
    h, w = canvas.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x = px + dx
            y = py + dy
            m = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            canvas[y[m], x[m]] = colors[m]


def project_to_view(world_xyz: torch.Tensor, intrinsics: torch.Tensor, cam_to_world: torch.Tensor):
    world_to_cam = torch.linalg.inv(cam_to_world)
    ones = torch.ones_like(world_xyz[:, :1])
    homog = torch.cat([world_xyz, ones], dim=-1)
    cam = (world_to_cam @ homog.T).T[:, :3]
    z = cam[:, 2]
    uvw = (intrinsics @ cam.T).T
    u = uvw[:, 0] / z.clamp_min(1e-6)
    v = uvw[:, 1] / z.clamp_min(1e-6)
    return u, v, z


def query_overlay(image_chw: torch.Tensor, u: torch.Tensor, v: torch.Tensor, z: torch.Tensor, colors: np.ndarray) -> np.ndarray:
    base = to_uint8(image_chw).copy()
    h, w = base.shape[:2]
    front = (z > 0).cpu().numpy()
    px = u.detach().cpu().numpy().astype(np.int64)
    py = v.detach().cpu().numpy().astype(np.int64)
    m = front & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    splat_points(base, px[m], py[m], colors[m], radius=1)
    return base


def bev_canvas(xy_list, colors_list, size: int = 640, margin: float = 3.0) -> np.ndarray:
    all_xy = np.concatenate(xy_list, axis=0)
    xmin, ymin = all_xy.min(axis=0) - margin
    xmax, ymax = all_xy.max(axis=0) + margin
    span = max(xmax - xmin, ymax - ymin, 1e-3)
    canvas = np.full((size, size, 3), 30, dtype=np.uint8)

    def to_px(xy):
        px = ((xy[:, 0] - xmin) / span * size).astype(np.int64)
        py = (size - (xy[:, 1] - ymin) / span * size).astype(np.int64)
        return px, py

    for xy, colors in zip(xy_list, colors_list):
        px, py = to_px(xy)
        m = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        splat_points(canvas, px[m], py[m], colors[m], radius=1)
    return canvas, (xmin, ymin, span, size)


def draw_boxes_bev(canvas: np.ndarray, frame_info, centers, dims, rots, valid) -> np.ndarray:
    xmin, ymin, span, size = frame_info
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    for c, d, r, ok in zip(centers, dims, rots, valid):
        if not ok:
            continue
        hx, hy = d[0].item() / 2, d[1].item() / 2
        corners_local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
        r2 = r[:2, :2].cpu().numpy()
        corners_world = corners_local @ r2.T + c[:2].cpu().numpy()
        px = ((corners_world[:, 0] - xmin) / span * size)
        py = (size - (corners_world[:, 1] - ymin) / span * size)
        poly = list(zip(px.tolist(), py.tolist()))
        draw.polygon(poly, outline=(255, 255, 255))
    return np.array(img)


def label(img: np.ndarray, text: str) -> Image.Image:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width, 18], fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 0))
    return pil


def make_grid(panels: list[tuple[str, np.ndarray]], cols: int, cell: int) -> Image.Image:
    rows = (len(panels) + cols - 1) // cols
    grid = Image.new("RGB", (cell * cols, cell * rows), (10, 10, 10))
    for i, (title, arr) in enumerate(panels):
        pil = Image.fromarray(arr).resize((cell, cell))
        pil = label(np.array(pil), title)
        r, c = divmod(i, cols)
        grid.paste(pil, (c * cell, r * cell))
    return grid


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=int, default=384)
    args = ap.parse_args()

    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(cfg, args.split)
    print(f"{args.split}: {len(dataset)} clips over {len(dataset.scenes)} scenes")

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate_clip)
    model = build_model(cfg, device, args.ckpt)

    scene = targets = gt = None
    for batch in loader:
        clip = batch[0]
        scene, targets, gt = to_scene_input(clip, device)
        if scene.boxes is not None and scene.boxes.num_instances > 0:
            break
    assert scene is not None

    depth = model.predict_depth(scene)
    sampled = model.build_queries(scene, depth)
    gaussians = model.decode(scene, depth, sampled)

    view = 0
    h, w = scene.image_size
    slot0 = int(scene.frame_index[view].item())

    n = sampled["world_xyz"].shape[0]
    sub = torch.randperm(n, device=device)[: min(20000, n)]
    q_xyz = sampled["world_xyz"][sub]
    q_colors = instance_colors(sampled["instance_id"][sub])

    u, v, z = project_to_view(q_xyz, scene.intrinsics[view], scene.cam_to_world[view])
    query_overlay_img = query_overlay(scene.images[view], u, v, z, q_colors)

    q_xy = q_xyz[:, :2].cpu().numpy()
    bev_q, frame_info = bev_canvas([q_xy], [q_colors])
    if scene.boxes is not None and scene.boxes.num_instances > 0:
        valid0 = scene.boxes.valid[:, slot0].cpu().numpy()
        bev_q = draw_boxes_bev(
            bev_q, frame_info, scene.boxes.centers[:, slot0], scene.boxes.dims[:, slot0],
            scene.boxes.rots[:, slot0], valid0,
        )

    posed = model.pose_gaussians(gaussians, scene.boxes, frame=slot0, target_time=scene.times[view])
    g_sub = torch.randperm(posed["means"].shape[0], device=device)[: min(20000, posed["means"].shape[0])]
    g_xyz = posed["means"][g_sub]
    g_colors = instance_colors(gaussians.instance_id[g_sub])
    g_xy = g_xyz[:, :2].cpu().numpy()
    bev_g, frame_info_g = bev_canvas([g_xy], [g_colors])
    if scene.boxes is not None and scene.boxes.num_instances > 0:
        bev_g = draw_boxes_bev(
            bev_g, frame_info_g, scene.boxes.centers[:, slot0], scene.boxes.dims[:, slot0],
            scene.boxes.rots[:, slot0], valid0,
        )

    targets_v0 = TargetViews(
        intrinsics=scene.intrinsics[view : view + 1],
        cam_to_world=scene.cam_to_world[view : view + 1],
        times=scene.times[view : view + 1],
        frame_index=scene.frame_index[view : view + 1],
    )
    out = model.render(gaussians, scene.boxes, targets_v0, height=h, width=w, render_depth=False)

    panels = [
        ("1 GT / input (view0)", to_uint8(scene.images[view])),
        ("2 InfiniDepth depth", colorize_depth(depth[view])),
        ("3 backprojected 3D queries", query_overlay_img),
        ("4 scene graph static/dynamic (BEV)", bev_q),
        ("5 decoded+posed Gaussians (BEV)", bev_g),
        ("6 pre-refiner alpha", colorize_alpha(out["alpha"][0, 0])),
        ("7 pre-refiner features (PCA)", pca_pseudo_rgb(out["features"][0])),
        ("8 post-refiner RGB (prediction)", to_uint8(out["rgb"][0])),
    ]

    import os

    os.makedirs(args.out, exist_ok=True)
    grid = make_grid(panels, cols=4, cell=args.cell)
    out_path = os.path.join(args.out, "pipeline_chain.png")
    grid.save(out_path)
    print(f"saved {out_path}")

    for title, arr in panels:
        name = title.split(" ", 1)[1].replace("/", "-").replace(" ", "_")
        Image.fromarray(arr).save(os.path.join(args.out, f"{title.split()[0]}_{name}.png"))
    print("saved individual panels too")


if __name__ == "__main__":
    main()
