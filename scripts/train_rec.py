"""Training entry point for incremental PointForwardRec variants."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import yaml
from torch.utils.data import DataLoader

from pointforward.data import WaymoDriveStudioWindows, collate_windows, to_scene_input
from pointforward.losses import RenderLoss, psnr, ssim
from pointforward.models import PointForwardRec
from pointforward.variants import build_model
from pointforward.temporal_consistency import TemporalRenderConsistencyLoss
from scripts.train import build_depth, lr_at


def build_windows_dataset(config: dict, split: str) -> WaymoDriveStudioWindows:
    """Build a DriveStudio window sequence without dropping canonical data options."""
    data = config["data"]
    scenes = data.get(f"{split}_scenes")
    if data.get("split_file"):
        with open(data["split_file"], "r") as handle:
            scenes = json.load(handle)[split]
    if data.get("layout", "drivestudio") != "drivestudio":
        raise ValueError("Rec variants require data.layout=drivestudio")
    return WaymoDriveStudioWindows(
        num_windows=config.get("rec", {}).get("num_windows", 2),
        root=data["root"],
        scenes=scenes,
        cameras=tuple(data.get("cameras", (0, 1, 2))),
        num_context=data.get("num_context", 4),
        num_target=data.get("num_target", 4),
        frame_stride=data.get("frame_stride", 5),
        clip_length=data.get("clip_length", 20),
        clip_stride=data.get("clip_stride", 1),
        resolution=tuple(data.get("resolution", (576, 768))),
        resolutions=(
            tuple(tuple(value) for value in data["resolutions"])
            if split == "train" and data.get("resolutions") else None
        ),
        rigid_classes=tuple(data.get("rigid_classes", ("Vehicle", "Cyclist"))),
        max_instances=data.get("max_instances", 24),
        load_lidar=data.get("load_lidar", True),
        target_lidar=data.get("target_lidar", False),
        require_lidar=data.get("require_lidar", False),
        require_instances=data.get("require_instances", False),
        require_contiguous=data.get("require_contiguous", False),
        apply_opencv2waymo=data.get("apply_opencv2waymo", True),
        variable_views=data.get("variable_views", False),
        min_cameras=data.get("min_cameras", 2),
        target_boundary_pairs=data.get("target_boundary_pairs", False),
        deterministic=split != "train",
        seed=config.get("seed", 0),
    )


@torch.no_grad()
def evaluate(
    model: PointForwardRec, loader: DataLoader, device: torch.device, max_batches: int
) -> dict:
    model.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        for item in batch:
            cache = None
            for window in item["windows"]:
                scene, targets, ground_truth = to_scene_input(window, device)
                output, cache, _ = model.reconstruct_window(scene, targets, cache)
                total_psnr += float(psnr(output["rgb"], ground_truth))
                total_ssim += float(ssim(output["rgb"], ground_truth))
                count += 1
    model.train()
    if count == 0:
        return {"psnr": 0.0, "ssim": 0.0, "windows": 0}
    return {"psnr": total_psnr / count, "ssim": total_ssim / count, "windows": count}


def _warm_start(model: torch.nn.Module, state: dict) -> tuple[list[str], list[str], list[str]]:
    current = model.state_dict()
    incompatible = [
        key for key, value in state.items()
        if key in current and current[key].shape != value.shape
    ]
    filtered = {key: value for key, value in state.items() if key not in incompatible}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return list(missing), list(unexpected), incompatible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--depth", default=None, help="override depth.kind")
    parser.add_argument("--steps", type=int, default=None, help="override train.steps")
    parser.add_argument("--out", default=None, help="override output dir")
    parser.add_argument("--resume", default=None, help="resume optimiser and step")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    if args.depth:
        config["depth"]["kind"] = args.depth
    if args.steps is not None:
        config["train"]["steps"] = args.steps
    if args.out:
        config["output"] = args.out
    if args.smoke_test:
        if not args.out or args.resume:
            parser.error("--smoke-test requires separate --out and no --resume")
        config["data"].update(resolution=[128, 192], resolutions=[[128, 192]])
        config["model"].update(num_static=1024, num_dynamic=128)
        config["train"].update(
            steps=2, batch_size=1, num_workers=0, log_every=1,
            eval_every=2, eval_batches=1, save_every=2,
        )

    out_dir = config["output"]
    if not args.resume and os.path.exists(os.path.join(out_dir, "config.yaml")):
        raise ValueError(f"Existing run: {out_dir}; use --resume or a new output")
    torch.manual_seed(config.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(message: str) -> None:
        print(message, flush=True)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as handle:
        yaml.safe_dump(config, handle)

    train_set = build_windows_dataset(config, "train")
    val_set = build_windows_dataset(config, "val")
    num_windows = config.get("rec", {}).get("num_windows", 2)
    log(
        f"train windows-items {len(train_set)} over {len(train_set.scenes)} scenes "
        f"(K={num_windows})"
    )
    log(f"val   windows-items {len(val_set)} over {len(val_set.scenes)} scenes")

    workers = config["train"].get("num_workers", 8)
    train_loader = DataLoader(
        train_set, batch_size=config["train"]["batch_size"], shuffle=True,
        num_workers=workers, collate_fn=collate_windows, drop_last=True,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_windows,
    )

    num_views = config["data"].get("num_context", 4) * len(
        config["data"].get("cameras", (0, 1, 2))
    )
    actual_views = train_set[0]["windows"][0]["context"]["images"].shape[0]
    assert actual_views == num_views, (
        f"config implies V={num_views} but loader produced {actual_views} context views"
    )

    model = build_model(config, build_depth(config)).to(device)
    if not isinstance(model, PointForwardRec):
        raise TypeError(f"train_rec requires a Rec model, got {type(model).__name__}")
    if model.temporal_attention is not None:
        log(
            "temporal attention enabled: train=bidirectional-within-clip "
            f"eval={'causal' if model.temporal_attention.causal_inference else 'bidirectional-within-clip'} "
            f"memory={model.temporal_attention.max_memory_frames}"
        )
    criterion = RenderLoss(
        lpips_weight=config["train"].get("lpips_weight", 0.2),
        lpips_chunk=config["model"].get("lpips_chunk", 0),
    ).to(device)
    consistency_config = dict(config.get("rec", {}).get("render_consistency", {}))
    consistency_enabled = consistency_config.pop("enabled", False)
    consistency = TemporalRenderConsistencyLoss(**consistency_config).to(device) if consistency_enabled else None
    if consistency is not None:
        log("render consistency enabled: RGB reprojection, depth, GT temporal residual, cross-window")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    n_train = sum(parameter.numel() for parameter in trainable)
    n_frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    n_fusion = sum(parameter.numel() for parameter in model.scene_fusion.parameters())
    log(
        f"trainable params {n_train/1e6:.1f}M   frozen {n_frozen/1e6:.1f}M   "
        f"scene_fusion {n_fusion/1e6:.2f}M"
    )

    base_lr = config["train"]["lr"]
    optimizer = torch.optim.AdamW(
        trainable, lr=base_lr, weight_decay=config["train"].get("weight_decay", 0.05)
    )
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"] + 1
        log(f"resumed from {args.resume} at step {start_step}")
    elif config["train"].get("warm_start"):
        checkpoint_path = config["train"]["warm_start"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        missing, unexpected, incompatible = _warm_start(
            model, checkpoint.get("model", checkpoint)
        )
        expected_prefixes = (
            "scene_fusion.", "depth_model.", "temporal_attention.", "temporal_to_query."
        )
        base_missing = [
            key for key in missing
            if not key.startswith(expected_prefixes) and key not in incompatible
        ]
        log(f"warm-started from {checkpoint_path}")
        log(f"  shape-incompatible tensors skipped: {len(incompatible)}")
        log(
            "  initialized new non-depth tensors: "
            f"{sum(not key.startswith('depth_model.') for key in missing)}"
        )
        log(f"  base modules not found in ckpt: {base_missing if base_missing else 'none'}")
        if base_missing:
            raise ValueError("Missing base weights in warm-start checkpoint")
        if unexpected:
            log(f"  unexpected keys in ckpt (ignored): {len(unexpected)}")

    loss_keys = ["l1", "loss"] + (
        ["lpips"] if config["train"].get("lpips_weight", 0.2) > 0 else []
    )
    amp_dtype = torch.bfloat16 if config["train"].get("bf16", True) else torch.float32
    grad_clip = config["train"].get("grad_clip", 1.0)
    log_every = config["train"].get("log_every", 50)
    eval_every = config["train"].get("eval_every", 1000)
    save_every = config["train"].get("save_every", 1000)
    total_steps = config["train"]["steps"]

    model.train()
    log_path = os.path.join(out_dir, "log.jsonl")
    step = start_step
    running: dict[str, float] = {}
    interval_start = time.time()
    peak_gb = 0.0
    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break
            for group in optimizer.param_groups:
                group["lr"] = base_lr * lr_at(step, config)
            optimizer.zero_grad(set_to_none=True)
            accumulated: dict[str, float] = {}
            successful = 0
            denominator = max(sum(len(item["windows"]) for item in batch), 1)

            for item in batch:
                cache = None
                previous_render = None
                for window in item["windows"]:
                    scene, targets, ground_truth = to_scene_input(window, device)
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                        output, cache, _ = model.reconstruct_window(
                            scene, targets, cache, render_depth=consistency is not None
                        )
                        losses = criterion(output["rgb"].float(), ground_truth)
                    if consistency is not None:
                        masks = window["target"]
                        if "dynamic_mask" not in masks or "sky_mask" not in masks:
                            raise ValueError("Render consistency requires dynamic and sky masks")
                        with torch.autocast(device.type, enabled=False):
                            packed = consistency.pack(
                                output, targets, ground_truth,
                                masks["dynamic_mask"].to(device), masks["sky_mask"].to(device),
                            )
                            temporal_losses, previous_render = consistency(
                                packed, len(config["data"]["cameras"]), previous_render,
                            )
                            losses["loss"] = losses["loss"] + temporal_losses["consistency_loss"]
                            losses.update(temporal_losses)
                    loss = losses["loss"] / denominator
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"step {step}: non-finite loss on {window['scene']}"
                        )
                    loss.backward()
                    successful += 1
                    for key, value in losses.items():
                        accumulated[key] = accumulated.get(key, 0.0) + float(value)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            if successful:
                accumulated = {
                    key: value / successful for key, value in accumulated.items()
                }
            for key, value in accumulated.items():
                running[key] = running.get(key, 0.0) + value
            if device.type == "cuda":
                peak_gb = max(peak_gb, torch.cuda.max_memory_allocated(device) / 1e9)

            if (step + 1) % log_every == 0:
                record = {key: value / log_every for key, value in running.items()}
                record.update(
                    step=step + 1, lr=optimizer.param_groups[0]["lr"],
                    sec_per_step=(time.time() - interval_start) / log_every,
                    peak_gb=round(peak_gb, 1), grad_norm=float(grad_norm),
                    windows=successful,
                )
                print(
                    f"step {record['step']:>6}/{total_steps}  "
                    + "  ".join(
                        f"{key} {value:.4f}" for key, value in record.items()
                        if key in loss_keys
                    )
                    + f"  lr {record['lr']:.2e}  {record['sec_per_step']:.2f}s/it  "
                    f"peak {record['peak_gb']}G"
                )
                with open(log_path, "a") as handle:
                    handle.write(json.dumps(record) + "\n")
                running, interval_start = {}, time.time()

            if (step + 1) % eval_every == 0:
                metrics = evaluate(
                    model, val_loader, device, config["train"].get("eval_batches", 20)
                )
                metrics["step"] = step + 1
                print(
                    f"  eval  PSNR {metrics['psnr']:.2f}  SSIM {metrics['ssim']:.4f}  "
                    f"({metrics['windows']} windows)"
                )
                with open(os.path.join(out_dir, "eval.jsonl"), "a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                interval_start = time.time()

            if (step + 1) % save_every == 0 or step + 1 == total_steps:
                path = os.path.join(out_dir, f"ckpt_{step+1:06d}.pt")
                state = {
                    key: value for key, value in model.state_dict().items()
                    if not key.startswith("depth_model.")
                }
                torch.save(
                    {"model": state, "optimizer": optimizer.state_dict(),
                     "step": step, "config": config}, path,
                )
                log(f"saved {path}")
                interval_start = time.time()
            step += 1

    log("training finished")


if __name__ == "__main__":
    main()