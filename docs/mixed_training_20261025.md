# Waymo + Argoverse Mixed Training

## Independent Runs

| Project | GPUs | Global batch | Local accumulated items | Steps | Initialization |
| --- | --- | --- | --- | --- | --- |
| PointForwardRec (fjn project) | 0,1,2,3 | 4 | 1 two-window item | 5000 | Original Waymo step-20000 base |
| PointForward (this project) | 4,5 | 8 | 4 clips | 45000 | Fresh task model with the original pretrained backbone and frozen depth prior |

GPU 7's existing service was not stopped. Both training processes are detached
from the terminal and use separate `flock` locks and torchrun rendezvous ports.
Do not relaunch a running job. A lock prevents duplicate launches using the same
lock path; it is not a machine-wide GPU reservation.

## Data And Configuration

- Config: `configs/waymo_argoverse_paper.yaml`.
- Model, depth, and training settings match `configs/waymo_ds_paper.yaml`.
- Global batch 8 is preserved with two ranks, each accumulating four clips.
- Waymo keeps `splits/waymo_ds.json`: 60 training scenes and 11 validation scenes.
- Argoverse uses `splits/argoverse_ds.json`: 45 training scenes and 5 validation scenes.
- Argoverse validation scenes: 008, 017, 026, 035, 044.
- Only standard 000-049 Argoverse scene directories are used; derived copies are excluded.
- Each dataset has probability 0.5, independent of its clip count. Sampling is with
  replacement; an individual batch need not contain an exact 50/50 split.
- Dataset identities are namespaced. A clip never crosses dataset or scene boundaries.
- Argoverse uses its own rigid classes, four-column LiDAR, OpenCV extrinsics, and
  principal-point crop with corresponding intrinsic adjustment.
- Evaluation applies the configured batch budget separately to each dataset and
  reports per-dataset metrics plus an equal-dataset mean.

## Outputs And Restart

- PointForward output: `runs/waymo_argoverse_paper_20261025/`.
- PointForward launcher log: `runs/waymo_argoverse_paper_20261025.launch.log`.
- PointForward lock: `runs/waymo_argoverse_paper_20261025.lock`.
- PointForwardRec output in the fjn project: `runs/waymo_argoverse_rec_20261025/`.
- PointForwardRec log in the fjn project: `logs/waymo_argoverse_rec_20261025.log`.

Checkpoints keep the existing format and cadence: PointForward every 2000 steps,
PointForwardRec every 500 steps, and both at completion. To recover an exited run,
use the same config and GPU set with `--resume` pointing to an existing checkpoint.
Do not use `--resume` for the fresh PointForward training requested here.

The runtime is `python (conda env with the dependencies)`. Launch environment:
`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, and `PYTHONPATH=.` from each project's directory.

## Verification

- PointForwardRec: four focused tests passed, including two-process DDP gradient
  averaging with an unused fusion layer in the first window.
- PointForward: eight tests passed (mixed data and regression tooling).
- Both projects: real 576x768 samples from each dataset had 12 context views,
  finite image/calibration tensors, nonempty LiDAR prompts, and disjoint scene splits.
- PointForwardRec: first real four-GPU optimizer step completed with finite loss.
- Required fixed video regression is blocked: `--ckpt auto` found no strict
  `ckpt_N.pt` checkpoint for variant `01_base`. The restored original 20k base was
  provided via `--base-ckpt`, but does not replace the missing current checkpoint.
  No full video regression pass is claimed.