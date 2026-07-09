# Decoder 对比实验设计

本文档记录后续 decoder-only 对比实验。老师的意思可以理解为：**encoder 固定为同一个 frozen Galileo，不再比较不同 encoder 或额外 temporal fusion，只比较 decoder/head 如何读取同一套 Galileo 特征。**

## 核心原则

```text
PASTIS S2 time series
  -> frozen Galileo encoder
  -> cached Galileo features
  -> different decoder/head
  -> semantic segmentation logits
```

也就是说，实验变量只放在 decoder：

- 不改变 Galileo 权重。
- 不解冻 Galileo。
- 不改变 PASTIS split。
- 不改变 `selected_timesteps=24`、`patch_size=8`、`normalize=True`。
- 尽量复用同一批 cached features，避免每个 decoder 重新跑 encoder。

公平比较 decoder 时，建议生成一套“对比实验缓存”：

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

## 当前 baseline：single-layer DPT

当前默认配置是：

```yaml
encoder:
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

这是第一条基线。它只证明 Galileo 最终层特征是否能被一个轻量空间 decoder 读出。

在 decoder 对比实验中，如果缓存是用 `hidden_layers: [3, 6, 9, 12]` 生成的，single-layer DPT 仍然可以只读取同一个 `.npz` 里的 `features` 字段，不需要重新缓存一份 `hidden_layers: []` 版本。

## 对比一：multi-layer DPT

multi-layer DPT 的变量仍然是 decoder 侧如何使用特征。Galileo encoder 不变，只是额外缓存若干中间层 hidden states，例如：

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

## 对比二：UPerNet decoder

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

## 建议实验表

| 实验名 | Encoder | Cached features | Decoder | 目的 |
| --- | --- | --- | --- | --- |
| `galileo_single_layer_dpt` | frozen Galileo base patch8 | final `features` | single-layer DPT | 默认 baseline |
| `galileo_multi_layer_dpt` | frozen Galileo base patch8 | `features_by_layer` from 3/6/9/12 | multi-layer DPT | 看多层特征是否有收益 |
| `galileo_upernet` | frozen Galileo base patch8 | `features_by_layer` from 3/6/9/12 | UPerNet-style | 比较另一类 segmentation decoder |

为保证公平，三组实验应保持：

- 相同 train/val/test fold。
- 相同 loss。
- 相同 optimizer 设置，除非专门做 optimizer 消融。
- 相同训练轮数或相同 early-stopping 规则。
- 相同 cached feature 版本。
- test set 只在最后报告一次。

## 代码落点

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

再重新运行：

```bash
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_dpt.yaml --split train
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_dpt.yaml --split val
```

如果用同一个缓存目录，务必确认旧缓存不是 `hidden_layers: []` 生成的；否则不会包含 `features_by_layer`。
