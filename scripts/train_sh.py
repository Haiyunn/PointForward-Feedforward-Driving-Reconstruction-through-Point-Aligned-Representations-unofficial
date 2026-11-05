"""Training entry point for fixed- and variable-view SH variants."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from pointforward.data import collate_clip, to_scene_input
from pointforward.losses import RenderLoss
from pointforward.variants import build_model
from scripts.train import build_dataset, build_depth, evaluate, lr_at, setup_distributed


def _load_compatible_state(model: torch.nn.Module, state: dict) -> tuple[list[str], list[str], list[str]]:
    """Load matching checkpoint tensors while reporting shape-incompatible keys."""
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
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    if args.depth:
        config["depth"]["kind"] = args.depth
    if args.steps is not None:
        config["train"]["steps"] = args.steps
    if args.out:
        config["output"] = args.out

    out_dir = config["output"]
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    torch.manual_seed(config.get("seed", 0) + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    def log(message: str) -> None:
        if is_main:
            print(message, flush=True)

    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.yaml"), "w") as handle:
            yaml.safe_dump(config, handle)

    train_set = build_dataset(config, "train")
    val_set = build_dataset(config, "val")
    log(f"world size {world_size}")
    log(f"train clips {len(train_set)} over {len(train_set.scenes)} scenes")
    log(f"val   clips {len(val_set)} over {len(val_set.scenes)} scenes")

    global_batch = config["train"]["batch_size"]
    if global_batch % world_size:
        raise ValueError(f"batch_size {global_batch} must divide world size {world_size}")
    local_batch = global_batch // world_size
    log(f"global batch {global_batch} = {world_size} ranks x {local_batch} clips")

    train_sampler = (
        DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1 else None
    )
    workers = config["train"].get("num_workers", 8)
    train_loader = DataLoader(
        train_set, batch_size=local_batch, shuffle=train_sampler is None,
        sampler=train_sampler, num_workers=workers, collate_fn=collate_clip,
        drop_last=True, persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_clip,
    )

    num_views = config["data"].get("num_context", 4) * len(
        config["data"].get("cameras", (0, 1, 2))
    )
    if not config["data"].get("variable_views", False):
        actual_views = train_set[0]["context"]["images"].shape[0]
        assert actual_views == num_views, (
            f"config implies V={num_views} but the loader produced {actual_views} context views"
        )

    model = build_model(config, build_depth(config)).to(device)
    if hasattr(model, "sh_degree"):
        log(f"SH degree {model.sh_degree}  ->  {model.decoder.num_sh} coeffs/channel")
    criterion = RenderLoss(
        lpips_weight=config["train"].get("lpips_weight", 0.2),
        lpips_chunk=config["model"].get("lpips_chunk", 0),
    ).to(device)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    n_train = sum(parameter.numel() for parameter in trainable)
    n_frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    log(f"trainable params {n_train/1e6:.1f}M   frozen {n_frozen/1e6:.1f}M")

    core = model
    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    base_lr = config["train"]["lr"]
    optimizer = torch.optim.AdamW(
        trainable, lr=base_lr, weight_decay=config["train"].get("weight_decay", 0.05)
    )

    start_step = 0
    checkpoint_path = args.resume or config["train"].get("warm_start")
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        missing, unexpected, incompatible = _load_compatible_state(
            core, checkpoint.get("model", checkpoint)
        )
        if incompatible:
            log(f"shape-incompatible checkpoint tensors skipped: {len(incompatible)}")
        relevant_missing = [key for key in missing if not key.startswith("depth_model.")]
        if relevant_missing:
            log(f"missing checkpoint tensors: {len(relevant_missing)}")
        if unexpected:
            log(f"unexpected checkpoint tensors: {len(unexpected)}")
        if args.resume:
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = checkpoint.get("step", -1) + 1
            log(f"resumed from {checkpoint_path} at step {start_step}")
        else:
            log(f"warm-started from {checkpoint_path}")

    loss_keys = ["l1", "loss"] + (
        ["lpips"] if config["train"].get("lpips_weight", 0.2) > 0 else []
    )
    amp_dtype = torch.bfloat16 if config["train"].get("bf16", True) else torch.float32
    grad_clip = config["train"].get("grad_clip", 1.0)
    log_every = config["train"].get("log_every", 50)
    eval_every = config["train"].get("eval_every", 2000)
    save_every = config["train"].get("save_every", 2000)
    total_steps = config["train"]["steps"]
    max_seconds = float(config["train"].get("max_hours", 0)) * 3600.0

    model.train()
    log_path = os.path.join(out_dir, "log.jsonl")
    step = start_step
    running: dict[str, float] = {}
    interval_start = time.time()
    wall_start = time.time()
    stop = False

    while step < total_steps and not stop:
        if train_sampler is not None:
            train_sampler.set_epoch(step)
        for batch in train_loader:
            if step >= total_steps:
                break
            if max_seconds > 0 and time.time() - wall_start > max_seconds:
                if is_main:
                    path = os.path.join(out_dir, f"ckpt_{step:06d}.pt")
                    state = {
                        key: value for key, value in core.state_dict().items()
                        if not key.startswith("depth_model.")
                    }
                    torch.save(
                        {"model": state, "optimizer": optimizer.state_dict(),
                         "step": step - 1, "config": config}, path,
                    )
                    log(f"wall-clock limit reached at step {step}; saved {path}")
                stop = True
                break

            for group in optimizer.param_groups:
                group["lr"] = base_lr * lr_at(step, config)
            optimizer.zero_grad(set_to_none=True)
            accumulated: dict[str, float] = {}
            successful = 0

            for clip in batch:
                scene, targets, ground_truth = to_scene_input(clip, device)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                    output = model(scene, targets)
                    losses = criterion(output["rgb"].float(), ground_truth)
                loss = losses["loss"] / len(batch)
                if not torch.isfinite(loss):
                    log(f"step {step}: non-finite loss on {clip['scene']}, skipping clip")
                    continue
                loss.backward()
                successful += 1
                for key, value in losses.items():
                    accumulated[key] = accumulated.get(key, 0.0) + float(value)

            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            if world_size > 1:
                statistics = torch.tensor(
                    [accumulated.get(key, 0.0) for key in loss_keys] + [float(successful)],
                    device=device, dtype=torch.float32,
                )
                dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
                total_successful = max(float(statistics[-1]), 1.0)
                accumulated = {
                    key: float(statistics[index]) / total_successful
                    for index, key in enumerate(loss_keys)
                }
            elif successful:
                accumulated = {key: value / successful for key, value in accumulated.items()}
            else:
                accumulated = {}
            for key, value in accumulated.items():
                running[key] = running.get(key, 0.0) + value

            if (step + 1) % log_every == 0 and is_main:
                record = {key: value / log_every for key, value in running.items()}
                record.update(
                    step=step + 1, lr=optimizer.param_groups[0]["lr"],
                    sec_per_step=(time.time() - interval_start) / log_every,
                )
                print(
                    f"step {record['step']:>6}/{total_steps}  "
                    + "  ".join(
                        f"{key} {value:.4f}" for key, value in record.items()
                        if key in ("loss", "l1", "lpips")
                    )
                    + f"  lr {record['lr']:.2e}  {record['sec_per_step']:.2f}s/it"
                )
                with open(log_path, "a") as handle:
                    handle.write(json.dumps(record) + "\n")
                running, interval_start = {}, time.time()
            elif (step + 1) % log_every == 0:
                running = {}

            if (step + 1) % eval_every == 0 and is_main:
                metrics = evaluate(
                    core, val_loader, device, config["train"].get("eval_batches", 20)
                )
                metrics["step"] = step + 1
                print(
                    f"  eval  PSNR {metrics['psnr']:.2f}  SSIM {metrics['ssim']:.4f}  "
                    f"({metrics['clips']} clips)"
                )
                with open(os.path.join(out_dir, "eval.jsonl"), "a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                interval_start = time.time()

            if ((step + 1) % save_every == 0 or step + 1 == total_steps) and is_main:
                path = os.path.join(out_dir, f"ckpt_{step+1:06d}.pt")
                state = {
                    key: value for key, value in core.state_dict().items()
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
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()