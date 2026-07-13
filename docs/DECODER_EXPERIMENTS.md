# Decoder 对比实验

本文档是当前实验主线的定义：**输入协议和 frozen Galileo encoder 固定，实验变量只放在 decoder/head。**

```text
PASTIS paper-aligned input
  -> frozen Galileo encoder
  -> early fusion: one shared spacetime-mean cache
  -> late fusion: online per-month hidden grids
  -> different decoder/head and temporal fusion position
  -> 19-class semantic segmentation
```

当前已经完成的 `single_layer_dpt` 和 `multi_layer_dpt` 是历史配置名：前者实际为**最终层卷积 decoder**，后者实际为**多层同尺度融合 decoder**。二者用于建立受控 baseline，均不是 DPT 原论文的完整复现。

组员正在实现的**多尺度 Galileo-DPT**才是当前 DPT 主方法：它需要将 layer 3/6/9/12 的同尺寸特征重组为多尺度金字塔，并进行深到浅渐进融合。其配置、参数量、结果和结论在实现与训练完成前保持待定。

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

Galileo 原论文的线性探测另列为“论文复现基线”。它保持相同 encoder、输入和 fold，但使用论文指定的线性 head、交叉熵、固定 50 epoch 与学习率搜索，不与方案一至五共用 decoder 容量、loss 或早停设置。

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
  spatial_token_strategy: spacetime_mean  # 方案一至四
```

方案五输入 Galileo 的数据完全相同，但设置 `preserve_temporal_features: true`，从同一组 token 中只平均 band-group 轴，不提前平均 T。

月度 raw-data 重建细节和官方公开协议边界见项目 [README](../README.md)。

## 最终层卷积 Baseline（历史名：Single-Layer DPT）

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

这里“一层”表示 decoder 只读取 Galileo 最终层，不表示 decoder 只有一个卷积层。它回答的是：在 encoder 完全冻结时，最终层特征能否被一个轻量空间 decoder 有效读出。该结构没有多层 reassemble 和渐进融合，因此科研表述中不再将它称为 DPT。

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

训练最终层卷积 baseline（沿用历史配置名）：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_single_layer_dpt_shared.yaml
```

训练多层同尺度融合 baseline（沿用历史配置名）：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_multi_layer_dpt_shared.yaml
```

训练 UPerNet-style decoder：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_upernet_shared.yaml
```

运行论文式线性探测的完整 16 学习率 × 5 seed 搜索：

```bash
conda run -n presl python -B scripts/sweep_linear_probe.py --config configs/galileo_linear_probe.yaml
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

方案一至五默认以 fold4 `val_mIoU` 早停：第 10 个 epoch 后，连续 12 个 epoch 没有至少 `0.001` 的提升便结束训练。训练同时保存最低 `val_loss` 的 `best_val_loss.pt` 与最高 `val_mIoU` 的 `best_val_miou.pt`；`best.pt` 保留为最低 loss checkpoint 的兼容名称。正式结果统一评估 `best_val_miou.pt` 并运行多个 seed。线性探测为复现论文固定训练 50 epoch，不启用早停，模型选择使用最后一轮。

## 对比矩阵

| 实验 | Frozen encoder | 读取特征 | Decoder | 目的 |
| --- | --- | --- | --- | --- |
| 最终层卷积 baseline | Galileo base, patch4 | final `features` | projection + residual conv + upsample | 检查空间细化读取能力 |
| 多层同尺度融合 baseline | 相同 | layers 3/6/9/12，均为16×16 | projection + additive residual fusion | 检查跨层信息收益 |
| 多尺度 Galileo-DPT | 相同 | layers 3/6/9/12，重组为多尺度 | reassemble + progressive fusion | 正在实现，结果待补 |
| UPerNet-style | 相同 | layers 3/6/9/12 | PPM + FPN-style | 比较另一类分割 decoder |
| 3D-Aware DPT | 相同，在线冻结 | layers 3/6/9/12 × T12 | 3D reassemble + 时空 attention + temporal query pooling | 比较晚期时间融合 |
| Galileo 论文线性探测 | 相同 | final `features` | 单个 Linear，逐 patch 输出 4×4 像素 logits | 复现论文 Table 17 probing 基线 |
| 同协议线性 head | 相同 | final `features` | 与论文探测相同的单个 Linear | 在统一训练协议下隔离 decoder 结构收益 |

## Galileo 论文线性探测

该实验使用 `configs/galileo_linear_probe.yaml`，结构与官方 `src/eval/linear_probe.py` 的 segmentation probe 对齐：

```text
Galileo final features [B, 768, 16, 16]
  -> reshape to 256 patch embeddings
  -> Linear(768, 19 * 4 * 4)
  -> rearrange patch pixels
  -> logits [B, 19, 64, 64]
```

训练设置为 AdamW、`weight_decay=0.01`、50 epoch、前 5 epoch 线性 warmup、随后 cosine 衰减至 `1e-5`，损失仅为 `CrossEntropy(ignore_index=-1)`。缓存读取时只加载最终层 `features`，因此已有共享 train/val/test 缓存可以直接复用。

论文附录 C.1 给出的学习率集合是：

```text
{1, 3, 4, 5} x 10^{-4,-3,-2,-1}
```

`scripts/sweep_linear_probe.py` 对每个 seed 训练全部 16 个候选，用该 seed 的 fold4 最终 `val_mIoU` 选择学习率，然后仅在 fold5 测试一次；默认运行 5 个连续 seed 并报告均值和总体标准差。测试集不参与学习率选择，也不参与早停。

参考：[Galileo 官方 linear_probe.py](https://github.com/nasaharvest/galileo/blob/main/src/eval/linear_probe.py)。

### 同协议线性 head

`configs/galileo_linear_decoder_shared.yaml` 复用完全相同的线性结构与共享缓存，但将训练部分对齐到最终层卷积 baseline：

```text
batch_size=16
loss=CE + 0.5 Dice
optimizer=Prodigy
epochs=100
early stopping=fold4 val_mIoU, patience 12
```

运行命令：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_linear_decoder_shared.yaml
```

因此它与论文复现回答不同问题：`galileo_linear_probe.yaml` 衡量能否复现 Galileo 的 probing protocol；`galileo_linear_decoder_shared.yaml` 衡量在完全相同训练策略下，线性 head 与卷积、多层融合等 decoder 的结构差距。

推荐结果名：

```text
galileo_base_frozen_single_layer_dpt_monthly12_tile64_patch4_fold123
```

## 多层同尺度融合 Baseline（历史名：Multi-Layer DPT）

该 baseline 不更换 encoder，也不重新生成另一套输入。它读取同一个共享缓存中的中间层：

```yaml
encoder:
  hidden_layers: [3, 6, 9, 12]

model:
  decoder: multi_layer_dpt
```

```text
Galileo layer 3/6/9/12
  -> each [B, 768, 16, 16]
  -> independent projection
  -> deep-to-shallow same-resolution additive fusion
  -> [B, 19, 64, 64]
```

最终层 baseline 使用同一 `.npz` 的 `features`；多层同尺度 baseline 使用 `features_by_layer`。二者不需要分别跑 encoder。由于所有输入层在运行时都是 `16×16`，这里没有形成 DPT 原论文的多尺度特征金字塔。

## 多尺度 Galileo-DPT（正在实现）

目标结构：

```text
Galileo layers 3/6/9/12，each [B, 768, 16, 16]
  -> independent projection / reassemble
  -> [B, C, 64, 64]
     [B, C, 32, 32]
     [B, C, 16, 16]
     [B, C,  8,  8]
  -> deep-to-shallow progressive fusion
  -> segmentation head
  -> [B, 19, 64, 64]
```

该结构应继续读取现有共享缓存，不重新训练 Galileo，不改变 PASTIS 输入和 fold。由于 Galileo 的 token 结构与图像 ViT 不完全相同，本项目复用已经验证的结构化空间聚合，不机械复制 DPT 的 class-token readout；因此准确表述是“适配 Galileo 的多尺度 DPT”，而不是逐行复现原始代码。

训练完成后补充：

| 项目 | 数值 |
| --- | --- |
| 配置与 commit | 待补 |
| 可训练参数 | 待补 |
| 峰值显存 / 训练时间 | 待补 |
| best val mIoU / epoch | 待补 |
| fold5 test loss / mIoU | 待补 |
| per-class IoU 与定性观察 | 待补 |

在数据产生前，不预写“优于 baseline”或“多尺度融合有效”等结论。

## 3D-Aware DPT 晚期融合（方案五）

方案五保持 PASTIS monthly-12 输入、Galileo 权重、冻结策略、fold、loss 和评测口径不变，但不读取已经执行 `spacetime_mean` 的共享缓存。它在线提取 Galileo 第 3/6/9/12 层 token，只平均 band-group 轴，保留：

```text
[B, 4, T=12, D=768, 16, 16]
```

完整 decoder 包含：

1. **3D Reassemble**：四个隐藏层投影到 256 通道，并重组为 `64/32/16/8` 四个空间尺度，时间长度保持 12。
2. **Global 3D bottleneck**：在最深 `8x8` 特征上执行全局时空自注意力。
3. **Divided space-time blocks**：高分辨率阶段交替执行逐像素全局时间注意力、偏移窗口空间注意力和分解式 3D depthwise convolution。
4. **Gated DPT fusion**：从深到浅逐级空间上采样，以门控方式融合 lateral feature，全程保留 T。
5. **Temporal query pooling**：在 `64x64` 特征上用多头查询注意力融合 12 个月，最后输出 `[B, 19, 64, 64]`。

训练入口：

```bash
conda run -n presl python -B scripts/train.py --config configs/galileo_3d_aware_dpt.yaml
```

该路线不生成额外 temporal cache。冻结 Galileo 每个 batch 在线前向，训练器通过梯度累积维持有效 batch；日志和 checkpoint 使用独立目录，不覆盖早期融合实验。

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
