# Decoder 对比实验

本文档是当前实验主线的定义：**输入协议和 frozen Galileo encoder 固定，实验变量只放在 decoder/head。**

```text
PASTIS paper-aligned input
  -> frozen Galileo encoder
  -> one shared feature cache
  -> different decoder/head
  -> 19-class semantic segmentation
```

single-layer DPT 是 baseline；multi-layer DPT 和 UPerNet-style decoder 都围绕它做对比。

## 固定实验条件

所有 decoder 必须保持：

- Galileo base 权重相同且完全冻结。
- train=`fold1/2/3`、val=`fold4`、test=`fold5`。
- 原始 PASTIS 时序聚合为12个月。
- `128x128` 原图切成4个 `64x64` 子块。
- Galileo `patch_size=4`。
- 原始 void label `19` 映射为 `-1` 并忽略。
- 有效类别 `0..18`，`num_classes=19`。
- 官方 `OURS + norm_no_clip + std_multiplier=2.0` 输入缩放。
- 相同 loss、optimizer 和模型选择规则。
- test 只在模型与超参数固定后运行。

配置中的固定字段：

```yaml
data:
  train_folds: [1, 2, 3]
  val_folds: [4]
  test_folds: [5]
  temporal_aggregation: monthly
  selected_timesteps: 12
  monthly_start_offset: 1
  source_image_size: 128
  tile_size: 64
  image_size: 64
  num_classes: 19
  void_label: 19
  ignore_index: -1
  normalization: galileo_norm_no_clip
  normalization_std_multiplier: 2.0

encoder:
  patch_size: 4
  freeze: true
  normalize: false
  spatial_token_strategy: spacetime_mean
```

月度 raw-data 重建细节和官方公开协议边界见项目 [README](../README.md)。

## Single-Layer DPT Baseline

baseline 配置：

```text
configs/galileo_single_layer_dpt_shared.yaml
```

数据流：

```text
Galileo final hidden sequence
  -> 按 [H_grid, W_grid, T, group] 聚合
  -> [B, 768, 16, 16]
  -> projection + 3 residual conv blocks
  -> bilinear upsample + smoothing
  -> [B, 19, 64, 64]
```

这里“一层”表示 decoder 只读取 Galileo 最终层，不表示 decoder 只有一个卷积层。它回答的是：在 encoder 完全冻结时，最终层特征能否被一个轻量空间 decoder 有效读出。

正式 baseline 不允许临时改变：

- PASTIS 月份窗口与子块方式
- Galileo patch size
- 输入缩放
- 类别和 void 处理
- decoder 容量
- loss 权重
- fold 划分

OOM 排查时可以临时减小 batch size。batch size 只影响吞吐，不改变已经生成的缓存特征。

## 共享缓存

decoder 对比统一使用：

```text
configs/galileo_shared_cache.yaml
```

它一次保存：

```text
features           # Galileo final layer
features_by_layer  # Galileo layers 3/6/9/12
```

默认目录：

```text
data/cache/galileo-base-patch8/monthly12_tile64_patch4_hl3-6-9-12_train/
data/cache/galileo-base-patch8/monthly12_tile64_patch4_hl3-6-9-12_val/
data/cache/galileo-base-patch8/monthly12_tile64_patch4_hl3-6-9-12_test/
```

旧的以下缓存均不兼容：

```text
t24_patch8_*
t24_patch8_hl3-6-9-12_*
```

原因不只是目录名不同；旧缓存使用了不同的时序、空间切块、patch size、归一化和20类标签，不能通过重命名复用。

缓存文件使用 `patch_id + tile origin` 命名。例如：

```text
10000_y0_x0.npz
10000_y0_x64.npz
10000_y64_x0.npz
10000_y64_x64.npz
```

这保证一张原始 PASTIS patch 的四个子块不会互相覆盖。缓存还记录 `aggregation_counts`，可检查每个月由多少原始 Sentinel-2 观测组成；值为0表示该月使用了相邻月插值。

## 运行顺序

环境和协议测试：

```bash
conda run -n presl python -B -m unittest discover -s tests -v
conda run -n presl python -B scripts/check_env.py --config configs/galileo_shared_cache.yaml --try-model
```

先做两样本缓存 smoke test：

```bash
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_shared_cache.yaml --split val --max-samples 2 --output-dir data/cache/paper_input_smoke
```

检查通过后生成正式共享缓存：

```bash
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_shared_cache.yaml --split train
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_shared_cache.yaml --split val
```

训练 single-layer DPT：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_single_layer_dpt_shared.yaml
```

训练 multi-layer DPT：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_multi_layer_dpt_shared.yaml
```

训练 UPerNet-style decoder：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_upernet_shared.yaml
```

模型和超参数固定后才生成 test：

```bash
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_shared_cache.yaml --split test
```

最终评估示例：

```bash
conda run -n presl python -B scripts/eval_cached.py --config configs/galileo_single_layer_dpt_shared.yaml --checkpoint checkpoints/galileo_single_layer_dpt_shared_paper_input_cached/best.pt --split test
```

## 记录要求

每组实验至少记录：

```text
git branch / commit
config 文件
cache 目录
GPU 与显存
seed
最佳 epoch
val_loss / val_mIoU / per-class IoU
test_loss / test_mIoU / per-class IoU
checkpoint 与 TensorBoard 路径
```

当前 `best.pt` 按最低 `val_loss` 保存。由于论文主要指标是 mIoU，正式批量实验应增加最高 `val_mIoU` checkpoint，并采用一致的 early-stopping 规则和多个 seed。

## 对比矩阵

| 实验 | Frozen encoder | 读取特征 | Decoder | 目的 |
| --- | --- | --- | --- | --- |
| single-layer DPT | Galileo base, patch4 | final `features` | single-layer DPT-style | baseline |
| multi-layer DPT | 相同 | layers 3/6/9/12 | multi-layer fusion DPT-style | 检查多层特征收益 |
| UPerNet-style | 相同 | layers 3/6/9/12 | PPM + FPN-style | 比较另一类分割 decoder |

推荐结果名：

```text
galileo_base_frozen_single_layer_dpt_monthly12_tile64_patch4_fold123
```

## Multi-Layer DPT

multi-layer DPT 不更换 encoder，也不重新生成另一套输入。它读取同一个共享缓存中的中间层：

```yaml
encoder:
  hidden_layers: [3, 6, 9, 12]

model:
  decoder: multi_layer_dpt
```

```text
Galileo layer 3/6/9/12
  -> each [B, 768, 16, 16]
  -> projection and progressive fusion
  -> [B, 19, 64, 64]
```

single-layer DPT 使用同一 `.npz` 的 `features`；multi-layer DPT 使用 `features_by_layer`。二者不需要分别跑 encoder。

## UPerNet-Style Decoder

老师所说的 “upper net” 应理解为 **UPerNet（Unified Perceptual Parsing Network）**。典型结构是：

```text
backbone multi-level features
  -> PPM / Pyramid Pooling Module
  -> FPN top-down fusion
  -> segmentation head
```

在本项目中它仍然读取同一套 Galileo layer 3/6/9/12 缓存，不允许引入新 encoder。

需要明确的限制：标准 UPerNet 通常接收 `1/4、1/8、1/16、1/32` 多分辨率 CNN 特征；Galileo 的不同 transformer layer 聚合后都是 `16x16`。因此本项目实现应称为 **UPerNet-style decoder**：复用 PPM 和 FPN 的多层融合思想，但输入不是标准多尺度金字塔。

实现位置：

```text
models/decoders/upernet.py
models/decoders/__init__.py
models/model.py
models/cached.py
configs/galileo_upernet_shared.yaml
```

共享缓存流程无需变化。

当前实现对最深的 layer 12 特征执行 `1/2/3/6` 四级 PPM，对 layer 3/6/9 执行 lateral projection，再由深到浅完成 FPN top-down 融合。由于四层 Galileo 特征均为 `16x16`，FPN 在这里融合的是 transformer 深度层级，而不是 CNN 的空间分辨率层级。
