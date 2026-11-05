"""Inference for rec variants with an arbitrary number of context frames.

This script rebuilds the model for ``len(context_frames) * len(cameras)`` input
views, then loads the checkpoint with ``strict=False`` while skipping any
shape-incompatible tensors (for example the fixed-view ``fusion.phi_v`` MLP).
It is intended for quick validation of whether a rec checkpoint still behaves
reasonably when the context count differs from training.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from pointforward.data import WaymoDriveStudioClips
from pointforward.models import InstanceBoxes, SceneInput, TargetViews
from scripts.train import build_depth
from pointforward.variants import build_model as build_variant_model
from pointforward.paths import add_path_args, apply_path_args


def _parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated nonempty integer list")
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.clamp(0, 1).detach().cpu().numpy()
    return (array.transpose(1, 2, 0) * 255).astype(np.uint8)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width, 20], fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 0))
    return np.array(canvas)


def _boxes_to_device(boxes: InstanceBoxes | None, device: torch.device) -> InstanceBoxes | None:
    if boxes is None:
        return None
    return InstanceBoxes(
        centers=boxes.centers.to(device),
        dims=boxes.dims.to(device),
        rots=boxes.rots.to(device),
        times=boxes.times.to(device),
        valid=boxes.valid.to(device),
    )


def _load_model(config: dict, checkpoint_path: str | None, device: torch.device):
    model = build_variant_model(config, build_depth(config), inference=True).to(device)
    incompatible: list[str] = []
    missing: list[str] = []
    unexpected: list[str] = []
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint.get("model", checkpoint)
        current = model.state_dict()
        incompatible = sorted(
            key for key, value in state.items()
            if key in current and current[key].shape != value.shape
        )
        filtered = {key: value for key, value in state.items() if key not in incompatible}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        missing = [key for key in missing if not key.startswith("depth_model.")]
    model.eval()
    return model, incompatible, missing, unexpected


def _find_scene(config: dict, split: str, requested_scene: str | None, required_frames: list[int]) -> str:
    if requested_scene:
        return requested_scene
    split_file = config["data"].get("split_file")
    if not split_file:
        raise ValueError("--scene is required when data.split_file is absent")
    with open(split_file, "r") as handle:
        scenes = json.load(handle)[split]
    dataset = WaymoDriveStudioClips(
        root=config["data"]["root"],
        scenes=scenes,
        cameras=tuple(config["data"].get("cameras", (0, 1, 2))),
        num_context=max(config["data"].get("num_context", 4), 1),
        num_target=max(config["data"].get("num_target", 1), 1),
        frame_stride=max(config["data"].get("frame_stride", 1), 1),
        clip_length=max(config["data"].get("clip_length", 1), 1),
        resolution=tuple(config["data"].get("resolution", (576, 768))),
        max_instances=config["data"].get("max_instances", 24),
        load_lidar=config["data"].get("load_lidar", True),
        target_lidar=False,
        require_lidar=config["data"].get("require_lidar", False),
        require_instances=config["data"].get("require_instances", False),
        deterministic=True,
        seed=config.get("seed", 0),
        require_contiguous=config["data"].get("require_contiguous", False),
        apply_opencv2waymo=config["data"].get("apply_opencv2waymo", True),
        clip_stride=max(config["data"].get("clip_stride", 1), 1),
        variable_views=False,
        min_cameras=config["data"].get("min_cameras", 2),
        target_boundary_pairs=False,
    )
    for meta in dataset.scenes:
        frames = set(meta["frames"])
        if all(frame in frames for frame in required_frames):
            return str(meta["name"])
    raise RuntimeError(f"no {split} scene contains all requested frames: {required_frames}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--scene", default=None, help="scene name; default picks the first matching scene")
    parser.add_argument("--context-frames", required=True, type=_parse_int_list)
    parser.add_argument("--target-frames", required=True, type=_parse_int_list)
    parser.add_argument("--cameras", default="0,1,2", type=_parse_int_list)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fusion-chunk", type=int, default=None)
    parser.add_argument("--decoder-chunk", type=int, default=None)
    parser.add_argument("--depth-view-chunk", type=int, default=None)
    parser.add_argument("--backbone-view-chunk", type=int, default=None)
    parser.add_argument("--refiner-view-chunk", type=int, default=None)
    add_path_args(parser)
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
        apply_path_args(config, args)

    context_frames = args.context_frames
    target_frames = args.target_frames
    if len(set(context_frames)) != len(context_frames):
        raise ValueError("context frames must be unique")
    if len(set(target_frames)) != len(target_frames):
        raise ValueError("target frames must be unique")

    cameras = tuple(args.cameras)
    selected_frames = sorted(set(context_frames) | set(target_frames))
    scene_name = _find_scene(config, args.split, args.scene, selected_frames)

    config = json.loads(json.dumps(config))
    config["data"]["num_context"] = len(context_frames)
    config["data"]["num_target"] = len(target_frames)
    config["data"]["cameras"] = list(cameras)
    if args.fusion_chunk is not None:
        config["model"]["fusion_chunk"] = args.fusion_chunk
    if args.decoder_chunk is not None:
        config["model"]["decoder_chunk"] = args.decoder_chunk
    if args.depth_view_chunk is not None:
        config["model"]["depth_view_chunk"] = args.depth_view_chunk
    if args.backbone_view_chunk is not None:
        config["model"]["backbone_view_chunk"] = args.backbone_view_chunk
    if args.refiner_view_chunk is not None:
        config["model"]["refiner_view_chunk"] = args.refiner_view_chunk

    dataset = WaymoDriveStudioClips(
        root=config["data"]["root"],
        scenes=[scene_name],
        cameras=cameras,
        num_context=len(context_frames),
        num_target=len(target_frames),
        frame_stride=max(config["data"].get("frame_stride", 1), 1),
        clip_length=max(config["data"].get("clip_length", 1), 1),
        resolution=tuple(config["data"].get("resolution", (576, 768))),
        max_instances=config["data"].get("max_instances", 24),
        load_lidar=config["data"].get("load_lidar", True),
        target_lidar=False,
        require_lidar=config["data"].get("require_lidar", False),
        require_instances=config["data"].get("require_instances", False),
        deterministic=True,
        seed=config.get("seed", 0),
        require_contiguous=config["data"].get("require_contiguous", False),
        apply_opencv2waymo=config["data"].get("apply_opencv2waymo", True),
        clip_stride=max(config["data"].get("clip_stride", 1), 1),
        variable_views=False,
        min_cameras=config["data"].get("min_cameras", 2),
        target_boundary_pairs=False,
    )
    meta = dataset.scenes[0]
    available = set(meta["frames"])
    missing_frames = [frame for frame in selected_frames if frame not in available]
    if missing_frames:
        raise ValueError(f"scene {scene_name} does not contain frames {missing_frames}")

    slot_of = {frame: index for index, frame in enumerate(selected_frames)}
    world_from = np.linalg.inv(dataset._ego_pose(meta, selected_frames[0]))
    norm_times = dataset._normalised_times(selected_frames)
    resolution = dataset.resolution
    context = dataset._collect(
        meta, context_frames, slot_of, norm_times, resolution, world_from,
        lidar=dataset.load_lidar, cameras=cameras,
    )
    target = dataset._collect(
        meta, target_frames, slot_of, norm_times, resolution, world_from,
        lidar=False, cameras=cameras,
    )
    boxes = dataset._build_boxes(meta, selected_frames, world_from)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene = SceneInput(
        images=context["images"].to(device),
        intrinsics=context["intrinsics"].to(device),
        cam_to_world=context["cam_to_world"].to(device),
        times=context["times"].to(device),
        frame_index=context["frame_index"].to(device),
        boxes=_boxes_to_device(boxes, device),
        sparse_depth=context.get("sparse_depth").to(device) if "sparse_depth" in context else None,
        sparse_mask=context.get("sparse_mask").to(device) if "sparse_mask" in context else None,
    )
    targets = TargetViews(
        intrinsics=target["intrinsics"].to(device),
        cam_to_world=target["cam_to_world"].to(device),
        times=target["times"].to(device),
        frame_index=target["frame_index"].to(device),
    )
    model, incompatible, missing, unexpected = _load_model(config, args.ckpt, device)
    output = model(scene, targets, render_depth=True)

    os.makedirs(args.out, exist_ok=True)
    prediction = output["rgb"].clamp(0, 1).cpu()
    ground_truth = target["images"].clamp(0, 1).cpu()
    alpha = output["alpha"].clamp(0, 1).cpu()

    views_per_frame = len(cameras)
    context_rows = []
    context_images = context["images"].cpu()
    for frame_index, frame in enumerate(context_frames):
        tiles = []
        for cam_index, cam in enumerate(cameras):
            image = _to_uint8(context_images[frame_index * views_per_frame + cam_index])
            tiles.append(_label(image, f"context frame {frame} cam {cam}"))
        context_rows.append(np.concatenate(tiles, axis=1))
    Image.fromarray(np.concatenate(context_rows, axis=0)).save(
        os.path.join(args.out, "context_contact_sheet.png")
    )

    for target_index, frame in enumerate(target_frames):
        for cam_index, cam in enumerate(cameras):
            flat = target_index * views_per_frame + cam_index
            pred_image = _to_uint8(prediction[flat])
            gt_image = _to_uint8(ground_truth[flat])
            side_by_side = np.concatenate(
                (
                    _label(gt_image, f"GT frame {frame} cam {cam}"),
                    _label(pred_image, f"Prediction frame {frame} cam {cam}"),
                ),
                axis=1,
            )
            Image.fromarray(side_by_side).save(
                os.path.join(args.out, f"frame{frame:03d}_cam{cam}_gt_pred.png")
            )
            Image.fromarray(gt_image).save(os.path.join(args.out, f"frame{frame:03d}_cam{cam}_gt.png"))
            Image.fromarray(pred_image).save(
                os.path.join(args.out, f"frame{frame:03d}_cam{cam}_pred.png")
            )
            Image.fromarray((alpha[flat, 0].numpy() * 255).clip(0, 255).astype(np.uint8)).save(
                os.path.join(args.out, f"frame{frame:03d}_cam{cam}_alpha.png")
            )

    metadata = {
        "scene": scene_name,
        "context_frames": context_frames,
        "target_frames": target_frames,
        "cameras": list(cameras),
        "resolution": list(resolution),
        "device": str(device),
        "checkpoint": args.ckpt,
        "num_context_views": int(scene.num_views),
        "num_target_views": int(targets.num_views),
        "incompatible_keys_skipped": incompatible,
        "missing_keys_after_load": missing,
        "unexpected_keys_after_load": unexpected,
        "model_variant": config["model"].get("variant"),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"scene={scene_name}")
    print(f"context_frames={context_frames}")
    print(f"target_frames={target_frames}")
    print(f"context_views={scene.num_views} target_views={targets.num_views}")
    print(f"skipped_shape_mismatch={len(incompatible)}")
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
