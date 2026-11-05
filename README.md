# PointForward（非官方复现）

> **声明：本仓库是论文 PointForward 的非官方复现代码**，与原作者无关。全部代码根据论文描述从零实现，可能与官方实现存在差异。官方代码请以原作者发布为准。
>
> **Disclaimer: this is an UNOFFICIAL reimplementation**, written from the paper only and not affiliated with the authors.

论文：**PointForward: Feedforward Driving Reconstruction through Point-Aligned Representations**
Cheng Chi, Xianqi Wang, et al. · NeurIPS 2026 · [arXiv:2605.11594](https://arxiv.org/abs/2605.11594)

## 方法概览

`pointforward/models/pointforward.py` 按论文 Fig. 2 的五个阶段实现：

1. **Sec. 3.1** 冻结的单目深度（InfiniDepth）+ 相机几何 → 稠密的点对齐 3D query
2. **Sec. 3.2** OmniRe 风格的场景图 → 静态 / 逐实例节点，动态 query 变换到物体规范坐标系后稀疏化
3. **Sec. 3.3** 时空融合 → 融合后的 query 特征
4. **Sec. 3.4** 高斯解码 → 带特征的 3D 高斯
5. **Sec. 3.4** gsplat 光栅化 + UNet 精修 → 任意目标视角 / 时刻的 RGB

训练设置按论文：AdamW，45K 步，batch 8，lr 5e-4（2K warmup + cosine），L1 + 0.2·LPIPS。

## 目录结构

```
pointforward/
  models/      主模型、深度、query、场景图、融合、高斯头、渲染器，以及 SH / Rec / 可变视角变体
  data/        Waymo（DriveStudio / NOTR 预处理格式）、Argoverse 2、混合数据集
  losses.py  metrics_cvpr.py  variants.py
scripts/       训练 / 评测 / 推理 / 渲染脚本
configs/       主配置 + variants/ 变体配置
splits/        数据划分
tests/         单元测试
docs/          变体说明、实验记录
```

## 环境

Python 3.10+、PyTorch 2.x（CUDA）、[gsplat](https://github.com/nerfstudio-project/gsplat)、lpips、torchvision、numpy、Pillow、PyYAML。
深度先验使用 InfiniDepth，需单独下载权重。

## 使用

数据采用 [DriveStudio](https://github.com/ziyc/drivestudio) 预处理格式。

### 路径参数

所有读取配置的脚本都支持以下命令行参数（实现见 `pointforward/paths.py`），无需修改 YAML：

| 参数 | 作用 | 默认值（相对仓库根目录） |
|---|---|---|
| `--data-root` | Waymo 数据根目录（单数据源配置即该数据源） | `data/waymo/processed/training` |
| `--argoverse-root` | Argoverse 2 数据根目录（混合数据配置） | `data/argoverse/processed/training` |
| `--depth-repo` | InfiniDepth 代码目录 | `third_party/InfiniDepth` |
| `--depth-ckpt` | InfiniDepth 权重 | `third_party/InfiniDepth/checkpoints/depth/infinidepth_depthsensor.ckpt` |
| `--warm-start` | 热启动权重（Rec / 微调变体） | 无 |

也可以直接按默认目录放置数据和权重（或建软链接），这样不需要任何参数。

### 命令

```bash
# 训练（论文设置，8 卡）
torchrun --nproc_per_node=8 -m scripts.train --config configs/waymo_ds_paper.yaml \
    --data-root /path/to/waymo/processed/training \
    --depth-repo /path/to/InfiniDepth --depth-ckpt /path/to/infinidepth_depthsensor.ckpt
# Waymo + Argoverse 混合训练
torchrun --nproc_per_node=8 -m scripts.train --config configs/waymo_argoverse_paper.yaml \
    --data-root /path/to/waymo/processed/training --argoverse-root /path/to/argoverse/processed/training
# 快速冒烟测试（不需要深度权重）
python -m scripts.train --config configs/waymo_ds.yaml --depth constant --steps 4 --data-root /path/to/waymo/processed/training
# 评测
python -m scripts.eval --config configs/waymo_ds_paper.yaml --ckpt PATH/ckpt.pt --data-root /path/to/waymo/processed/training
# 各变体统一入口（见 docs/VARIANTS.md），其余参数会原样转发
python -m scripts.run_variant --config configs/variants/03_1_rec_cross_attention.yaml --mode train \
    --data-root /path/to/waymo/processed/training --warm-start PATH/base_ckpt.pt
# 单元测试
python -m pytest -q tests
```

## 复现结果（参考）

| 设置 | val PSNR |
|---|---|
| Waymo-only，20K 步，3 前视相机 | ≈ 29.8 |
| Waymo + Argoverse 混合训练，18K 步（40 clip） | 26.37（Waymo 28.00 / Argoverse 24.74） |

数据划分与训练规模与论文不同，数字不可与论文直接比较。

## 引用

请引用原论文：

```bibtex
@inproceedings{chi2026pointforward,
  title     = {PointForward: Feedforward Driving Reconstruction through Point-Aligned Representations},
  author    = {Chi, Cheng and Wang, Xianqi and others},
  booktitle = {NeurIPS},
  year      = {2026}
}
```
