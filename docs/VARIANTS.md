# Canonical PointForward Variants

The canonical project keeps feature-color, SH direct-RGB, incremental Rec, and
arbitrary-view behavior as explicit model classes. Configuration selects the
class and the train/inference module; checkpoint type is never inferred.

## Architecture Matrix

| Version | Config | Color path | Fusion | Data convention | Model class |
|---|---|---|---|---|---|
| 1 | `01_base.yaml` | Feature color + RGB refiner | Original fixed-view `phi_v(V)` | Waymo, frontal 3 | `PointForward` |
| 2 | `02_sh.yaml` | SH coefficients rendered directly to RGB | Original fixed-view `phi_v(V)` | Waymo, frontal 3 | `PointForwardSH` |
| 3.1 | `03_1_rec_cross_attention.yaml` | Feature color + RGB refiner | Fixed-view fusion + SceneFusion cross-attention | Consecutive Waymo windows | `PointForwardRec` |
| 3.2 | `03_2_rec_temporal_attention.yaml` | Feature color + RGB refiner | 3.1 plus explicit temporal attention | Consecutive Waymo windows | `PointForwardRec` |
| 4 | `04_crossdata_arbitrary_views.yaml` | SH direct RGB | Masked mean+max, fixed two-input `phi_v` | Five cameras, random subsets of 2 through 5 | `PointForwardVariableViewSH` |
| 5 | `05_argoverse_sh.yaml` | SH direct RGB | Masked mean+max, fixed two-input `phi_v` | Seven Argoverse cameras, random subsets | `PointForwardVariableViewSH` |

Rec is feature-color, not SH. Versions 4 and 5 are SH models even though their
dataset and camera policies differ.

## Commands

Run every version through the same dispatcher. Arguments after `--mode` are
forwarded as an argument list without a shell.

```bash
python -m scripts.run_variant --config configs/variants/01_base.yaml --mode train
python -m scripts.run_variant --config configs/variants/02_sh.yaml --mode train
python -m scripts.run_variant --config configs/variants/03_1_rec_cross_attention.yaml --mode train
python -m scripts.run_variant --config configs/variants/03_2_rec_temporal_attention.yaml --mode train
python -m scripts.run_variant --config configs/variants/04_crossdata_arbitrary_views.yaml --mode train
python -m scripts.run_variant --config configs/variants/05_argoverse_sh.yaml --mode train
```

Inference uses the same config and an explicit checkpoint:

```bash
python -m scripts.run_variant --config configs/variants/02_sh.yaml --mode infer --ckpt PATH/ckpt.pt
python -m scripts.run_variant --config configs/variants/03_1_rec_cross_attention.yaml --mode infer --ckpt PATH/ckpt.pt
python -m scripts.run_variant --config configs/variants/05_argoverse_sh.yaml --mode infer --ckpt PATH/ckpt.pt
```

## Checkpoint Compatibility

- Version 1 retains the original `SpatialTemporalFusion` and fixed-view
  `phi_v(V)` parameter shapes. Existing base checkpoints remain compatible.
- Version 2 accepts fixed-view SH checkpoints from `PointForward_SH` when model
  dimensions and SH degree match.
- Versions 3.1 and 3.2 can warm-start feature-color base/Rec weights with
  `strict=False`; SceneFusion and temporal modules are additional parameters.
- Version 4/5 checkpoints use the separate two-input variable-view `phi_v` and
  are not shape-compatible with fixed-view base or fixed-view SH fusion heads.
- Feature-color and SH decoder/refiner checkpoints are intentionally distinct.

## Data Behavior

The merged DriveStudio loader retains configurable `apply_opencv2waymo`,
configurable rigid classes, and Waymo `Nx14`/`Nx10` plus Argoverse `Nx4` LiDAR.
It additionally supports `clip_stride`, random `variable_views` bounded by
`min_cameras`, and `WaymoDriveStudioWindows` with `collate_windows` for Rec.
Argoverse uses OpenCV camera axes already, so its config sets
`apply_opencv2waymo: false` and declares Argoverse rigid vehicle classes.

## Legacy Renderer Warning

Legacy generic render scripts in the source projects constructed `PointForward`
unconditionally. They must not be used for SH, crossdata, or Argoverse SH
checkpoints. Use the canonical dispatcher or the explicit `infer_sh`,
`infer_crossdata`, and `infer_argoverse` modules so the checkpoint is loaded into
the configured SH class rather than silently into the feature-color base model.

## Deterministic Regression Sampling

After framework changes, validate the fixed ten-scene Waymo protocol without
launching GPU rendering:

```bash
python -m scripts.regression_sample preflight \
  --config configs/variants/01_base.yaml --ckpt auto
```

After model, data, rendering, fusion, attention, or checkpoint-loading changes,
run the complete regression on an available GPU:

```bash
python -m scripts.regression_sample run \
  --config configs/variants/01_base.yaml --ckpt auto --gpu 0
```

If the frozen version-1 20k checkpoint has been restored at another location,
add `--base-ckpt /path/to/ckpt_020000.pt`. The workflow validates the checkpoint
and keys the immutable baseline cache by its SHA-256 digest; it never silently
uses a different base training step.

The frozen baseline is stored under
`results/regression_sampling/baselines/<baseline-checkpoint-sha256>/`. Current
runs are stored under
`results/regression_sampling/runs/<variant>_step<step>_<checkpoint-sha-prefix>/`.
Each completed run contains `current_all_scenes.mp4`,
`comparison_two_column.mp4`, `run_manifest.json`, and atomic `status.json`;
the frozen baseline directory contains `baseline_all_scenes.mp4` and its
identity manifest. The two concatenated streams contain exactly 1800 frames
(180 seconds at 10 fps), and the comparison places baseline on the left.