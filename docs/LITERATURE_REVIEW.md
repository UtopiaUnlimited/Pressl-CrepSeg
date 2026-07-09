# Literature Review Notes

本文档整理 PreSSL-CropSeg 当前阶段最需要的论文支撑。重点不是堆文献，而是说明每类文献在本项目研究逻辑中的作用。

## 1. 研究问题的核心支撑

### Galileo: Learning Global and Local Features in Pretrained Remote Sensing Models

- 链接：https://arxiv.org/abs/2502.09356
- 作用：本项目的核心模型来源。
- 支撑点：
  - 遥感任务具有跨区域、跨模态、跨尺度的共性，适合使用预训练 foundation model。
  - Galileo 面向遥感多模态和时空输入设计，目标是学习可迁移的 global/local 表征。
  - 本项目可以把 Galileo 作为 frozen encoder，研究其表征是否能迁移到 PASTIS 作物分割。

本项目应避免把 Galileo 只是当成普通 backbone。更准确的说法是：

```text
We evaluate whether a frozen remote-sensing foundation model can provide transferable spatiotemporal representations for crop semantic segmentation.
```

## 2. PASTIS 与作物时序分割支撑

### Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks

- 链接：https://arxiv.org/abs/2107.07933
- 作用：PASTIS 数据集和 SITS 分割任务的核心引用。
- 支撑点：
  - PASTIS 是面向 Satellite Image Time Series 的公开数据集，包含农业地块 panoptic/semantic annotations。
  - 作物/地块分割不能只依赖单时相图像，作物物候和时间序列模式很重要。
  - 该工作使用 temporal self-attention 提取多尺度时空特征，是我们讨论“时序信息重要性”的直接依据。

对本项目的意义：

```text
PASTIS 证明了 crop segmentation 是一个时序遥感问题，而不是普通 RGB 单图分割问题。
```

### Satellite Image Time Series Classification with Pixel-Set Encoders and Temporal Self-Attention

- 链接：https://arxiv.org/abs/1911.07757
- 作用：支撑“作物识别依赖时间序列/注意力建模”的早期代表工作。
- 支撑点：
  - Satellite image time series 对大规模农业监测非常重要。
  - temporal self-attention 可以替代 RNN 类模型，更高效地提取时间特征。

对本项目的意义：

```text
已有作物分类研究已经说明时序自注意力对作物识别有效；本项目进一步关注 foundation model 预训练表征能否迁移到 dense segmentation。
```

### Lightweight Temporal Self-Attention for Classifying Satellite Image Time Series

- 链接：https://arxiv.org/abs/2007.00586
- 作用：补充 SITS temporal attention 的效率和参数量讨论。
- 支撑点：
  - 遥感时序模型需要兼顾精度和计算效率。
  - 轻量 temporal attention 可以降低参数和计算复杂度。

对本项目的意义：

```text
本地 8GB 显存限制不是纯工程问题，而是遥感时序模型普遍面对的效率问题。
```

## 3. 遥感自监督与基础模型支撑

### Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data

- 链接：https://arxiv.org/abs/2103.16607
- 作用：遥感 in-domain self-supervised pretraining 的早期代表。
- 支撑点：
  - ImageNet 预训练和遥感影像之间存在 domain gap。
  - 利用遥感自身的时间、地点和季节信息做自监督预训练，可以得到更适合下游遥感任务的表征。

对本项目的意义：

```text
为什么不能只用 ImageNet baseline？因为遥感和自然图像存在域差异，需要比较遥感自监督预训练的迁移价值。
```

### SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery

- 链接：https://arxiv.org/abs/2207.08051
- 作用：masked autoencoder 在多光谱/多时相卫星影像上的代表工作。
- 支撑点：
  - 卫星影像拥有天然的 temporal 和 multispectral structure。
  - 对多时相和多光谱数据做预训练可以提升下游分类、分割等任务。

对本项目的意义：

```text
Galileo 不是孤立方向；SatMAE 等工作已经说明 temporal/multispectral pretraining 是遥感下游迁移的重要路线。
```

### Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning

- 链接：https://arxiv.org/abs/2212.14532
- 作用：支撑遥感预训练需要考虑尺度问题。
- 支撑点：
  - 遥感影像的尺度信息不同于自然图像。
  - scale-aware pretraining 可以改善 geospatial representation transfer。

对本项目的意义：

```text
解释为什么不能简单照搬自然图像 ViT/DPT 经验；遥感影像的空间尺度、分辨率和地物尺度需要单独考虑。
```

### Foundation Models for Generalist Geospatial Artificial Intelligence

- 链接：https://arxiv.org/abs/2310.18660
- 作用：Prithvi / geospatial foundation model 方向代表。
- 支撑点：
  - foundation model 可以通过大规模自监督预训练服务多个 Earth observation 下游任务。
  - 预训练模型在少标注和多任务迁移场景中有潜力。

对本项目的意义：

```text
本项目属于 geospatial foundation model 下游适配研究，而不仅是一个单数据集分割工程。
```

### Vision Foundation Models in Remote Sensing: A Survey

- 链接：https://arxiv.org/abs/2408.03464
- 作用：综述型引用，放在 related work 开头很合适。
- 支撑点：
  - 系统总结遥感 foundation model 的架构、预训练数据和方法。
  - 说明 self-supervised learning、masked autoencoder、contrastive learning 在遥感 foundation model 中的重要性。

对本项目的意义：

```text
用于说明我们的研究问题处在 remote sensing foundation models 的主流发展方向中。
```

### Foundation Models for Remote Sensing and Earth Observation: A Survey

- 链接：https://arxiv.org/abs/2410.16602
- 作用：另一个更宽的 RSFM/EO foundation model 综述。
- 支撑点：
  - 遥感 foundation model 需要处理多模态、空间/光谱分辨率差异和 temporal dynamics。
  - 这些挑战正好对应 Galileo 和本项目关注的问题。

## 4. Dense prediction / decoder 支撑

### Vision Transformers for Dense Prediction

- 链接：https://arxiv.org/abs/2103.13413
- 作用：DPT-style decoder 的核心引用。
- 支撑点：
  - Transformer backbone 输出 token，需要重新组装为 image-like representations。
  - dense prediction 可以通过 reassemble + fusion + upsampling 的方式恢复空间分辨率。

对本项目的意义：

```text
我们把 Galileo token 输出接到 DPT-style decoder，有明确的 dense prediction 方法学来源。
```

需要注意：

```text
当前 baseline 是 single-layer feature decoder，不是完整 multi-layer DPT。
```

如果后续提取 Galileo 第 3/6/9/12 层 hidden states，再做多层 fusion，才更接近 DPT 原始思想。

## 5. 推荐阅读顺序

第一轮只读这些，建立项目主线：

1. Galileo
2. PASTIS / U-TAE
3. SatMAE
4. DPT
5. SeCo

第二轮用于 related work 扩展：

1. Scale-MAE
2. Prithvi / Generalist Geospatial AI
3. Remote sensing foundation model surveys
4. Pixel-Set Encoder / LTAE

## 6. 可以写进论文/开题的逻辑

本项目 related work 可以按三段写：

### Satellite Image Time Series for Crop Segmentation

PASTIS 和 U-TAE 说明农业地块分割依赖多时相 Sentinel-2 数据。作物类别的差异往往体现在物候变化和时间序列光谱模式中，因此 crop segmentation 应被视作 spatiotemporal remote sensing dense prediction，而不是普通单图语义分割。

### Self-Supervised and Foundation Models in Remote Sensing

SeCo、SatMAE、Scale-MAE、Prithvi 和 Galileo 表明，大规模未标注遥感数据可以通过自监督预训练学习可迁移表征。相比 ImageNet 预训练，遥感预训练能更好处理多光谱、时序、尺度和传感器域差异。

### Dense Prediction from Transformer Features

DPT 说明 Transformer token 可以通过 reassemble 和 decoder fusion 转换为 dense prediction 输出。本项目借鉴这一思想，将 Galileo 的时空 token 表征读出为空间分割图。当前 baseline 先使用单层 token feature，后续扩展到多层 hidden states。

## 7. 当前 baseline 需要引用哪些文献

第一版 baseline 最少引用：

```text
Galileo
PASTIS / U-TAE
DPT
SatMAE
SeCo
```

如果写开题或论文 related work，建议再补：

```text
Scale-MAE
Prithvi
Remote sensing foundation model survey
Pixel-Set Encoder / LTAE
```

## 8. Galileo token layout 核查结论

当前项目不能把 Galileo 的 `last_hidden_state: [B, N, D]` 简单理解为 `[B, H_grid * W_grid, D]`。

PASTIS 输入是多时间、多光谱序列：

```text
[B, T, C, H, W]
```

送入 Galileo processor 后接近：

```text
space_time_x: [B, H, W, T, C]
```

因此 Galileo 内部 token 数通常不只由空间 patch 数决定，还会包含时间维度、模态/通道组维度，以及可能的 time/static/space tokens。对于 `128x128`、`patch_size=8`、`T=24`，空间 grid 是 `16x16=256`，但 space-time token 数可能是：

```text
16 * 16 * 24 * group_count
```

所以当前 baseline 的安全做法不是任意取前 256 个 token，也不是按 `[group, spatial]` 直接平均，而是：

```text
hidden: [B, N, D]
if N == H_grid * W_grid * T * group_count:
  reshape -> [B, H_grid, W_grid, T, group_count, D]
  mean over T and group_count
  -> [B, H_grid, W_grid, D]
  permute -> [B, D, H_grid, W_grid]
else:
  fail loudly and inspect official token layout
```

这就是当前代码中 `spatial_token_strategy=spacetime_mean` 的含义。

这个策略仍然是一个有条件的 baseline 假设：它假设 Galileo 输出序列的主要部分按 `[H_grid, W_grid, T, group]` 展平。如果后续官方代码核查显示 token 顺序或输出内容不同，应改为直接从 Galileo collapse 前的结构化 hidden states 或官方 processor/model 中间输出构造空间特征。
