# Decoder 对比实验

本文档是当前实验主线的唯一说明：**encoder 固定为同一个 frozen Galileo，实验变量只放在 decoder/head。**

因此，一层 DPT 不是另一个单独方向，而是本阶段的 baseline；multi-layer DPT 和后续 UPerNet-style decoder 都要围绕它做对比。

## 核心原则

```text
PASTIS S2 time series
  -> frozen Galileo encoder
  -> cached Galileo features
  -> different decoder/head
  -> semantic segmentation logits
```

实验变量只放在 decoder：

- 不改变 Galileo 权重。
- 不解冻 Galileo。
- 不改变 PASTIS split。
- 不改变 `selected_timesteps=24`、`patch_size=8`、`normalize=True`。
- 尽量复用同一批 cached features，避免每个 decoder 重新跑 encoder。
- test set 只在模型和超参数固定后做最终报告，不用于调参。

默认 split：

```text
train: fold1, fold2, fold3
val:   fold4
test:  fold5
```

PASTIS 样本约定：

```text
S2:     [T, 10, 128, 128]
target: [128, 128]  # TARGET_*.npy 第 0 通道
classes: 0..19      # num_classes = 20
```

## 一层 DPT Baseline

默认 baseline 是：

```yaml
config: configs/galileo_dpt.yaml

encoder:
  freeze: true
  selected_timesteps: 24
  patch_size: 8
  spatial_token_strategy: spacetime_mean
  hidden_layers: []

model:
  decoder: single_layer_dpt
```

含义：

```text
Galileo final hidden sequence
  -> 按 [H_grid, W_grid, T, group] 聚合
  -> [B, D, 16, 16]
  -> single-layer DPT-style decoder
  -> [B, 20, 128, 128]
```

它回答的是第一层问题：在不微调 Galileo 的情况下，Galileo 最终层 frozen feature 是否能被一个轻量空间 decoder 读出为有效的 PASTIS 作物分割结果。

正式 baseline 不要临时改变：

- `selected_timesteps`
- `patch_size`
- `num_classes`
- fold 划分
- decoder 容量
- loss 权重

如果为了排查 OOM 临时改小 `selected_timesteps` 或 decoder，请把结果标记为 debug，不要和正式 baseline 混在一起。

## 共享缓存

公平比较 decoder 时，推荐生成一套“对比实验共享缓存”：

```yaml
encoder:
  hidden_layers: [3, 6, 9, 12]
```

这套缓存会同时包含：

```text
features           # final-layer spatial feature grid
features_by_layer  # layer 3/6/9/12 spatial feature grids
```

single-layer DPT 只读取 `features`；multi-layer DPT 和 UPerNet-style decoder 读取 `features_by_layer`。这样三组实验可以共用同一批 encoder 前向结果，变量更集中在 decoder。

项目里提供了三份 shared config：

```text
configs/galileo_shared_cache.yaml              # 只用于生成共享缓存
configs/galileo_single_layer_dpt_shared.yaml   # 用共享缓存训练 single-layer DPT
configs/galileo_multi_layer_dpt_shared.yaml    # 用共享缓存训练 multi-layer DPT
```

默认 baseline `configs/galileo_dpt.yaml` 保持 `hidden_layers: []`，用于单层 DPT 的最小主线；decoder 对比时优先使用上面这组三个 config，避免把 baseline 和对比实验混在一起。

使用 `configs/galileo_shared_cache.yaml` 时，默认目录会自动带 hidden layer 后缀：

```text
data/cache/galileo-base-patch8/t24_patch8_hl3-6-9-12_train/
data/cache/galileo-base-patch8/t24_patch8_hl3-6-9-12_val/
data/cache/galileo-base-patch8/t24_patch8_hl3-6-9-12_test/
```

如果手动指定 `--output-dir`，务必确认它不是旧的 `hidden_layers: []` 缓存；否则没有 `features_by_layer`，不能给 multi-layer DPT 或 UPerNet-style decoder 用。

## 运行前检查

确认数据和权重存在：

```text
data/PASTIS/metadata.geojson
data/PASTIS/DATA_S2/S2_*.npy
data/PASTIS/ANNOTATIONS/TARGET_*.npy

pretrained/galileo-base-patch8/config.json
pretrained/galileo-base-patch8/model.safetensors
pretrained/galileo-base-patch8/processing_galileo.py
pretrained/galileo-base-patch8/modeling_galileo.py
```

环境检查：

```bash
conda run -n presl python -B scripts/check_env.py --config configs/galileo_dpt.yaml --try-model
```

one-batch smoke test：

```bash
conda run -n presl python -B scripts/train.py --config configs/galileo_dpt.yaml --batch-size 1 --epochs 1 --max-train-batches 1 --max-val-batches 1 --no-amp
```

成功标准：

- 没有 shape error。
- 没有 CUDA OOM。
- 输出 `train_loss`、`val_loss`、`val_miou`。
- 写出对应 checkpoint。

如果本地 8GB 显存不够完整跑 Galileo cache 或训练，优先本地做 smoke test，再交给更大显存机器运行完整缓存和训练。

## Baseline 运行

直接在线训练 baseline：

```bash
conda run -n presl python -B scripts/train.py --config configs/galileo_dpt.yaml --batch-size 1 --epochs 100 --no-amp
```

更推荐先缓存，再训练 decoder/head：

```bash
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_dpt.yaml --split train
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_dpt.yaml --split val
```

默认 baseline 缓存目录：

```text
data/cache/galileo-base-patch8/t24_patch8_train/
data/cache/galileo-base-patch8/t24_patch8_val/
```

训练 cached baseline：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_dpt.yaml --batch-size 4 --epochs 100 --no-amp
```

评估验证集：

```bash
conda run -n presl python -B scripts/eval_cached.py --config configs/galileo_dpt.yaml --checkpoint checkpoints/galileo_dpt_cached/best.pt --split val
```

最终测试集评估：

```bash
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_dpt.yaml --split test
conda run -n presl python -B scripts/eval_cached.py --config configs/galileo_dpt.yaml --checkpoint checkpoints/galileo_dpt_cached/best.pt --split test
```

需要记录：

```text
使用的 git commit
使用的 config 文件
GPU 型号和显存
cache_features train/val/test 是否完成
val_loss / val_miou / val per_class_iou
test_loss / test_miou / test per_class_iou
checkpoint 路径
TensorBoard 日志路径
```

## Decoder 对比流程

推荐顺序：

```text
1. single-layer DPT baseline
2. multi-layer DPT
3. UPerNet-style decoder
```

先生成共享缓存：

```bash
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_shared_cache.yaml --split train
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_shared_cache.yaml --split val
```

最终报告前再生成 test：

```bash
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_shared_cache.yaml --split test
```

用共享缓存训练一层 DPT baseline：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_single_layer_dpt_shared.yaml
```

用共享缓存训练 multi-layer DPT：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_multi_layer_dpt_shared.yaml
```

如果 decoder/head 训练 OOM，把 config 或命令里的 batch size 改为 `2` 或 `1`。这只影响训练吞吐，不改变缓存里的 Galileo features。

## 对比方案

| 实验名 | Encoder | Cached features | Decoder | 目的 |
| --- | --- | --- | --- | --- |
| `galileo_single_layer_dpt` | frozen Galileo base patch8 | final `features` | single-layer DPT | 本阶段 baseline |
| `galileo_multi_layer_dpt` | frozen Galileo base patch8 | `features_by_layer` from 3/6/9/12 | multi-layer DPT | 看多层特征是否有收益 |
| `galileo_upernet` | frozen Galileo base patch8 | `features_by_layer` from 3/6/9/12 | UPerNet-style | 比较另一类 segmentation decoder |

为保证公平，三组实验应保持：

- 相同 train/val/test fold。
- 相同 loss。
- 相同 optimizer 设置，除非专门做 optimizer 消融。
- 相同训练轮数或相同 early-stopping 规则。
- 相同 cached feature 版本。
- test set 只在最后报告一次。

推荐 baseline 结果命名：

```text
galileo_base_patch8_frozen_single_layer_dpt_t24_fold123
```

## Multi-Layer DPT

multi-layer DPT 的变量仍然是 decoder 侧如何使用特征。Galileo encoder 不变，只是额外读取若干中间层 hidden states：

```yaml
encoder:
  hidden_layers: [3, 6, 9, 12]

model:
  decoder: multi_layer_dpt
```

含义：

```text
Galileo layer 3/6/9/12 hidden states
  -> 每层都聚合成 [B, D, 16, 16]
  -> DPT-style multi-layer fusion decoder
  -> [B, 20, 128, 128]
```

注意：这里不是换 encoder。encoder 权重、输入和前向过程不变；变化的是 decoder 能读取 final layer 还是读取多层 frozen hidden states。

在 decoder 对比实验中，如果缓存是用 `hidden_layers: [3, 6, 9, 12]` 生成的，single-layer DPT 仍然可以只读取同一个 `.npz` 里的 `features` 字段，不需要重新缓存一份 `hidden_layers: []` 版本。

## UPerNet-Style Decoder

老师说的 “upper net” 很可能是 **UPerNet**，不是普通英文里的 upper net。UPerNet 通常指 **Unified Perceptual Parsing Network**。

UPerNet 是一种语义分割 decoder/head 设计，常见结构是：

```text
backbone multi-level features
  -> PPM / Pyramid Pooling Module
  -> FPN-style top-down feature fusion
  -> segmentation head
```

直观理解：

- DPT 更像 Transformer dense prediction 里的 reassemble + fusion decoder。
- UPerNet 更像语义分割里常用的 “PPM + FPN” decoder。
- PPM 负责在最高层特征上做多尺度上下文池化。
- FPN 负责把不同层级特征融合起来。

在本项目里，UPerNet 不应该引入新的 encoder。它应该读取同一套 cached Galileo hidden features：

```yaml
encoder:
  hidden_layers: [3, 6, 9, 12]

model:
  decoder: upernet
```

推荐理解为：

```text
Galileo layer 3/6/9/12 hidden states
  -> 每层聚合成 [B, D, 16, 16]
  -> UPerNet-style PPM + FPN decoder
  -> [B, 20, 128, 128]
```

一个重要限制：标准 UPerNet 通常接 CNN backbone 的多尺度特征，例如 `1/4, 1/8, 1/16, 1/32`。Galileo transformer 各层 hidden states 聚合后目前都是同一个 `16x16` 空间分辨率。因此本项目里的 UPerNet 更准确地说是 **UPerNet-style decoder**：复用 PPM 和 FPN 融合思想，但输入是 Galileo 多层同分辨率特征。论文或汇报里需要把这一点讲清楚。

## UPerNet 代码落点

后续实现 UPerNet 时，建议放在：

```text
models/decoders/upernet.py
```

并在这些位置接入：

```text
models/decoders/__init__.py
models/model.py
models/cached.py
configs/galileo_dpt.yaml
```

建议新增 decoder 名称：

```yaml
model:
  decoder: upernet
```

缓存流程不用变，但 UPerNet 需要多层特征，因此要先设置：

```yaml
encoder:
  hidden_layers: [3, 6, 9, 12]
```
