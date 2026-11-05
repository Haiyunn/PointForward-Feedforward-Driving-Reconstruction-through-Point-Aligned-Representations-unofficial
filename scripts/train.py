"""Training entry point.

Reproduces the paper's schedule: AdamW, 45K iterations at batch 8, lr 5e-4 with
2K warmup steps followed by cosine decay, multi-resolution inputs, and the
L1 + 0.2*LPIPS rendering objective.

A "batch" is 8 clips; each clip is a full scene reconstruction, so they are
accumulated rather than stacked (instance counts differ per clip).

The paper trains on 8 GPUs at global batch 8, i.e. one clip per GPU per step.
Run it the same way with torchrun; a single process falls back to accumulating
the whole batch sequentially, which is correct but roughly 8x slower per step.

Usage:
    torchrun --nproc_per_node=8 -m scripts.train --config configs/waymo_ds.yaml
    python -m scripts.train --config configs/waymo_ds.yaml --depth constant --steps 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from datetime import timedelta

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from pointforward.data.argoverse import ARGOVERSE_RIGID_CLASSES, ArgoverseDriveStudioClips
from pointforward.data.mixed import MixedDataset, MixedDistributedSampler

from pointforward.data import (
    WaymoClips,
    WaymoDriveStudioClips,
    collate_clip,
    to_scene_input,
)
from pointforward.losses import RenderLoss, psnr, ssim
from pointforward.models import ConstantDepth, FrozenInfiniDepth, PointForward
from pointforward.variants import build_model as build_variant_model


def build_depth(cfg: dict) -> torch.nn.Module:
    """Instantiate the frozen depth prior named by the config."""
    kind = cfg["depth"]["kind"]
    if kind == "constant":
        # Pipeline dry-run only: no depth prior, so renders are meaningless.
        return ConstantDepth(cfg["depth"].get("value", 20.0))
    if kind == "infinidepth":
        return FrozenInfiniDepth(
            repo_root=cfg["depth"]["repo_root"],
            checkpoint=cfg["depth"]["checkpoint"],
            variant=cfg["depth"].get("variant", "InfiniDepth_DepthSensor"),
            encoder=cfg["depth"].get("encoder", "vitl16"),
        )
    raise ValueError(f"unknown depth kind: {kind}")


def build_dataset(cfg: dict, split: str):
    """Build the train or val clip dataset for the configured layout."""
    d = cfg["data"]
    if "sources" in d:
        if not d["sources"]:
            raise ValueError("data.sources must not be empty")
        shared = {key: value for key, value in d.items() if key not in (
            "sources", "root", "layout", "split_file", "train_scenes", "val_scenes"
        )}
        return MixedDataset([
            build_dataset({**cfg, "data": {**shared, **source}}, split)
            for source in d["sources"]
        ], [source["name"] for source in d["sources"]])

    # A split file (scene-level, produced by scripts/make_split.py) takes
    # precedence over inline scene lists.
    scenes = d.get(f"{split}_scenes")
    if d.get("split_file"):
        with open(d["split_file"], "r") as fh:
            scenes = json.load(fh)[split]

    common = dict(
        root=d["root"],
        scenes=scenes,
        cameras=tuple(d.get("cameras", (0, 1, 2))),
        num_context=d.get("num_context", 4),
        num_target=d.get("num_target", 4),
        frame_stride=d.get("frame_stride", 5),
        resolution=tuple(d.get("resolution", (576, 768))),
        # Multi-resolution sampling is a training-time augmentation only;
        # evaluation stays at a fixed size so metrics stay comparable.
        resolutions=(
            tuple(tuple(r) for r in d["resolutions"])
            if split == "train" and d.get("resolutions")
            else None
        ),
        max_instances=d.get("max_instances", 24),
        deterministic=(split != "train"),
        seed=cfg.get("seed", 0),
    )

    layout = d.get("layout", "drivestudio")
    if layout in ("drivestudio", "argoverse_drivestudio"):
        dataset_class = ArgoverseDriveStudioClips if layout == "argoverse_drivestudio" else WaymoDriveStudioClips
        rigid_classes = ARGOVERSE_RIGID_CLASSES if layout == "argoverse_drivestudio" else ("Vehicle", "Cyclist")
        return dataset_class(
            clip_length=d.get("clip_length", 20),
            load_lidar=d.get("load_lidar", True),
            target_lidar=d.get("target_lidar", False),
            require_lidar=d.get("require_lidar", False),
            require_instances=d.get("require_instances", False),
            require_contiguous=d.get("require_contiguous", False),
            rigid_classes=tuple(d.get("rigid_classes", rigid_classes)),
            apply_opencv2waymo=d.get("apply_opencv2waymo", True),
            clip_stride=d.get("clip_stride", 1),
            variable_views=d.get("variable_views", False),
            min_cameras=d.get("min_cameras", 2),
            **common,
        )
    if layout == "streetgaussian":
        return WaymoClips(**common)
    raise ValueError(f"unknown data.layout: {layout}")


def setup_distributed() -> tuple[int, int, int]:
    """Join the torchrun process group if there is one.

    Returns (rank, world_size, local_rank); (0, 1, 0) when running standalone.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=2))
    return rank, world_size, local_rank


def lr_at(step: int, cfg: dict) -> float:
    """Linear warmup then cosine decay, as a multiplier on the base lr."""
    warmup = cfg["train"]["warmup_steps"]
    total = cfg["train"]["steps"]
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate(model: PointForward, loader: DataLoader, device: torch.device, max_batches: int) -> dict:
    """Render held-out target views and report PSNR / SSIM."""
    if isinstance(loader.dataset, MixedDataset):
        results = {
            name: evaluate(model, DataLoader(dataset, batch_size=1, collate_fn=collate_clip), device, max_batches)
            for name, dataset in zip(loader.dataset.names, loader.dataset.datasets)
        }
        return {
            "psnr": sum(result["psnr"] for result in results.values()) / len(results),
            "ssim": sum(result["ssim"] for result in results.values()) / len(results),
            "clips": sum(result["clips"] for result in results.values()),
            "datasets": results,
        }
    model.eval()
    tot_psnr, tot_ssim, n = 0.0, 0.0, 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        for clip in batch:
            scene, targets, gt = to_scene_input(clip, device)
            out = model(scene, targets)
            tot_psnr += float(psnr(out["rgb"], gt))
            tot_ssim += float(ssim(out["rgb"], gt))
            n += 1

    model.train()
    if n == 0:
        return {"psnr": 0.0, "ssim": 0.0, "clips": 0}
    return {"psnr": tot_psnr / n, "ssim": tot_ssim / n, "clips": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--depth", default=None, help="override depth.kind")
    ap.add_argument("--steps", type=int, default=None, help="override train.steps")
    ap.add_argument("--out", default=None, help="override output dir")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)
    if args.depth:
        cfg["depth"]["kind"] = args.depth
    if args.steps:
        cfg["train"]["steps"] = args.steps
    if args.out:
        cfg["output"] = args.out

    out_dir = cfg["output"]

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    torch.manual_seed(cfg.get("seed", 0) + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    def log(msg: str) -> None:
        if is_main:
            print(msg, flush=True)

    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.yaml"), "w") as fh:
            yaml.safe_dump(cfg, fh)

    # ------------------------------------------------------------------ data
    train_set = build_dataset(cfg, "train")
    val_set = build_dataset(cfg, "val")
    log(f"world size {world_size}")
    log(f"train clips {len(train_set)} over {len(train_set.scenes)} scenes")
    log(f"val   clips {len(val_set)} over {len(val_set.scenes)} scenes")
    if isinstance(train_set, MixedDataset):
        for name, dataset in zip(train_set.names, train_set.datasets):
            log(f"source {name}: {len(dataset.scenes)} scenes, {len(dataset)} clips, probability {1 / len(train_set.names):.2f}")

    # The configured batch size is GLOBAL, as in the paper; split it across ranks
    # so 8 processes at global batch 8 means one clip each, matching their setup.
    global_batch = cfg["train"]["batch_size"]
    if global_batch % world_size:
        raise ValueError(
            f"batch_size {global_batch} must divide the world size {world_size}"
        )
    local_batch = global_batch // world_size
    log(f"global batch {global_batch} = {world_size} ranks x {local_batch} clips")

    sampler_class = MixedDistributedSampler if isinstance(train_set, MixedDataset) else DistributedSampler
    train_sampler = (
        sampler_class(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.get("seed", 0))
        if world_size > 1 or isinstance(train_set, MixedDataset)
        else None
    )
    train_loader = DataLoader(
        train_set,
        batch_size=local_batch,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg["train"].get("num_workers", 8),
        collate_fn=collate_clip,
        drop_last=True,
        persistent_workers=cfg["train"].get("num_workers", 8) > 0,
    )
    # Validation runs on rank 0 only, so it is never sharded.
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_clip,
    )

    # ----------------------------------------------------------------- model
    # V is fixed by the clip protocol; the fusion module needs it up front so
    # phi_v exists before the optimiser is built from model.parameters().
    num_views = cfg["data"].get("num_context", 4) * len(cfg["data"].get("cameras", (0, 1, 2)))
    if not cfg["data"].get("variable_views", False):
        assert train_set[0]["context"]["images"].shape[0] == num_views, (
            f"config implies V={num_views} but the loader produced "
            f"{train_set[0]['context']['images'].shape[0]} context views"
        )

    model = build_variant_model(cfg, build_depth(cfg)).to(device)

    criterion = RenderLoss(lpips_weight=cfg["train"].get("lpips_weight", 0.2), lpips_chunk=cfg["model"].get("lpips_chunk", 0)).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log(f"trainable params {n_train/1e6:.1f}M   frozen {n_frozen/1e6:.1f}M")

    core = model
    if world_size > 1:
        # Gradient checkpointing re-enters the graph, which the default reducer
        # cannot see, so DDP is told not to expect a static graph.
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    base_lr = cfg["train"]["lr"]
    opt = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=cfg["train"].get("weight_decay", 0.05))

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        core.load_state_dict(ckpt["model"], strict=False)
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        log(f"resumed from {args.resume} at step {start_step}")

    # Fixed, rank-independent key order for the logging all_reduce.
    loss_keys = ["l1", "loss"] + (["lpips"] if cfg["train"].get("lpips_weight", 0.2) > 0 else [])

    amp_dtype = torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float32
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    log_every = cfg["train"].get("log_every", 50)
    eval_every = cfg["train"].get("eval_every", 2000)
    save_every = cfg["train"].get("save_every", 2000)
    total_steps = cfg["train"]["steps"]

    # ------------------------------------------------------------------ loop
    model.train()
    log_path = os.path.join(out_dir, "log.jsonl")
    step = start_step
    running: dict[str, float] = {}
    t0 = time.time()
    running_steps = 0

    while step < total_steps:
        if train_sampler is not None:
            # Reshuffle differently each pass, or every epoch sees one order.
            train_sampler.set_epoch(step)
        for batch in train_loader:
            if step >= total_steps:
                break

            for group in opt.param_groups:
                group["lr"] = base_lr * lr_at(step, cfg)

            opt.zero_grad(set_to_none=True)
            accum: dict[str, float] = {}
            n_ok = 0

            # Each clip is one scene; gradients accumulate over the rank's share
            # of the global batch, and DDP all-reduces them at the last backward.
            for j, clip in enumerate(batch):
                if step == start_step:
                    print(f"rank {rank} first-step source {clip.get('dataset', 'waymo')} scene {clip['scene']} shape {tuple(clip['context']['images'].shape)}", flush=True)
                scene, targets, gt = to_scene_input(clip, device)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                    out = model(scene, targets)
                    losses = criterion(out["rgb"].float(), gt)
                loss = losses["loss"] / len(batch)

                finite = torch.isfinite(loss).to(torch.int32)
                if world_size > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not bool(finite):
                    raise FloatingPointError(f"step {step}: non-finite loss on at least one rank")

                loss.backward()
                n_ok += 1
                for k, v in losses.items():
                    accum[k] = accum.get(k, 0.0) + float(v)

            # Every rank must execute the same collectives in the same order,
            # whatever happened to its own clips.  A rank that bailed out here
            # (all its clips produced a non-finite loss) would leave the others
            # blocked in all_reduce forever, so the path below is unconditional
            # and the loss keys are fixed rather than read off this rank's dict.
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()

            if world_size > 1:
                stats = torch.tensor(
                    [accum.get(k, 0.0) for k in loss_keys] + [float(n_ok)],
                    device=device,
                    dtype=torch.float32,
                )
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                total_ok = max(float(stats[-1]), 1.0)
                accum = {k: float(stats[i]) / total_ok for i, k in enumerate(loss_keys)}
            elif n_ok:
                accum = {k: v / n_ok for k, v in accum.items()}
            else:
                accum = {}

            for k, v in accum.items():
                running[k] = running.get(k, 0.0) + v

            running_steps += 1
            if ((step + 1) % log_every == 0 or step == start_step) and is_main:
                rec = {k: v / running_steps for k, v in running.items()}
                rec.update(
                    step=step + 1,
                    lr=opt.param_groups[0]["lr"],
                    sec_per_step=(time.time() - t0) / running_steps,
                    peak_gb=round(torch.cuda.max_memory_allocated(device) / 1e9, 1),
                )
                print(
                    f"step {rec['step']:>6}/{total_steps}  "
                    + "  ".join(f"{k} {v:.4f}" for k, v in rec.items() if k in ("loss", "l1", "lpips"))
                    + f"  lr {rec['lr']:.2e}  {rec['sec_per_step']:.2f}s/it"
                )
                with open(log_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                running, t0 = {}, time.time()
                running_steps = 0
            elif (step + 1) % log_every == 0 or step == start_step:
                running, running_steps = {}, 0

            if (step + 1) % eval_every == 0 and is_main:
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                    metrics = evaluate(core, val_loader, device, cfg["train"].get("eval_batches", 20))
                metrics["step"] = step + 1
                print(f"  eval  PSNR {metrics['psnr']:.2f}  SSIM {metrics['ssim']:.4f}  ({metrics['clips']} clips)")
                with open(os.path.join(out_dir, "eval.jsonl"), "a") as fh:
                    fh.write(json.dumps(metrics) + "\n")
                t0 = time.time()

            if ((step + 1) % save_every == 0 or step + 1 == total_steps) and is_main:
                path = os.path.join(out_dir, f"ckpt_{step+1:06d}.pt")
                # The frozen depth prior is not saved: it is loaded from its own
                # released checkpoint and never updated.
                state = {
                    k: v for k, v in core.state_dict().items() if not k.startswith("depth_model.")
                }
                torch.save({"model": state, "optimizer": opt.state_dict(), "step": step, "config": cfg}, path)
                print(f"  saved {path}", flush=True)
                t0 = time.time()

            if world_size > 1 and ((step + 1) % eval_every == 0 or (step + 1) % save_every == 0):
                dist.barrier()
            step += 1

    log("training finished")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
