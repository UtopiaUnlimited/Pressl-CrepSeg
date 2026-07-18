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

现已实现的 **Galileo-Adapted 2D DPT（方案四）**是当前 DPT 主方法：它将 layer 3/6/9/12 的同尺寸特征重组为多尺度金字塔，保留 final 原尺度旁路，并进行深到浅渐进融合。当前配置与参数量已经固定，训练和测试结果仍保持待定。

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

缓存现在分成两条互不覆盖的实验记录。

旧版 `spatial_v1` 继续使用：

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

新版 `temporal_v2` 使用：

```text
configs/galileo_temporal_shared_cache.yaml
```

每个 `.npz` 只保存：

```text
temporal_features_by_layer  # block3/block6/block9/encoder_final
                            # [4, 12, 768, 16, 16]
target 和少量元数据
```

它不重复保存 `features`、`features_by_layer` 或 `hidden_state`。最深一级使用 final-normalized encoder output，而不是额外重复保存 block 12 和 final 两份 T 特征。方案一至四在 loader 中沿 T 求均值，方案五直接读取 T。旧配置和旧命令默认仍指向 `spatial_v1`；对训练命令增加 `--cache-format temporal_v2 --temporal-dtype float16` 才切换到新缓存。

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

训练器支持按 fold4 `val_mIoU` 早停：第 10 个 epoch 后，连续 12 个 epoch 没有至少 `0.001` 的提升便结束训练；但当前 `configs/` 下所有配置均将该开关设为 `false`，由实验者手动决定何时终止。训练同时保存最低 `val_loss` 的 `best_val_loss.pt` 与最高 `val_mIoU` 的 `best_val_miou.pt`；`best.pt` 保留为最低 loss checkpoint 的兼容名称。正式结果统一评估 `best_val_miou.pt` 并运行多个 seed。线性探测为复现论文固定训练 50 epoch，模型选择使用最后一轮。

## 对比矩阵

| 实验 | Frozen encoder | 读取特征 | Decoder | 目的 |
| --- | --- | --- | --- | --- |
| 最终层卷积 baseline | Galileo base, patch4 | final `features` | projection + residual conv + upsample | 检查空间细化读取能力 |
| 多层同尺度融合 baseline | 相同 | layers 3/6/9/12，均为16×16 | projection + additive residual fusion | 检查跨层信息收益 |
| Galileo-Adapted 2D DPT | 相同 | layers 3/6/9/12，多尺度 + final 原尺度旁路 | learned reassemble + progressive fusion | 方案四已实现，结果待补 |
| UPerNet-style | 相同 | layers 3/6/9/12 | PPM + FPN-style | 比较另一类分割 decoder |
| 3D-Aware DPT | 相同，冻结并可缓存 | block 3/6/9/final × T12，final 时间旁路 | 3D reassemble + 时空 attention + temporal query pooling | 比较晚期时间融合 |
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

## Galileo-Adapted 2D DPT（方案四，已实现）

配置：

```text
configs/galileo_adapted_dpt_shared.yaml
```

该结构不机械复制 DPT 的图像 ViT readout，而是把共享缓存中已经完成 Galileo 时空上下文化和 `spacetime_mean` 的 layer 3/6/9/12 作为专用输入适配接口：

```text
Galileo layers 3/6/9/12，each [B, 768, 16, 16]
  -> independent 1x1 projection, GroupNorm, GELU
  -> learned reassemble
  -> [B, 256, 64, 64]  (layer 3,  ConvTranspose x4)
     [B, 256, 32, 32]  (layer 6,  ConvTranspose x2)
     [B, 256, 16, 16]  (layer 9,  identity refine)
     [B, 256,  8,  8]  (layer 12, stride-2 Conv)
  -> parallel native skip [B, 256, 16, 16] (layer 12/final)
  -> inject native skip into the 16x16 fusion stage
  -> deep-to-shallow progressive fusion
  -> 256-to-128 segmentation head
  -> [B, 19, 64, 64]
```

每一级融合先将深层特征双线性上采样到 lateral 尺寸，再使用带 GroupNorm/GELU 的 residual convolution unit 细化 lateral 和融合结果。最深层的 `8x8` 只作为低分辨率上下文分支；同一个 layer 12/final 输入还通过共享 projection/refine 保持 `16x16`，并与 layer 9 的 `16x16` lateral 一起进入融合。GroupNorm 不依赖 batch 统计，适合后续因显存调整 physical batch；四个 projection 相互独立，保留不同 Galileo 深度层级的分布差异。

它继续读取现有共享缓存，不重新训练 Galileo，不改变 PASTIS 输入、fold 或 `spacetime_mean`。因此方案二和方案四之间的主要变量是“同尺度相加”与“多尺度 reassemble + 渐进融合”。准确表述是“适配 Galileo 的 2D DPT decoder”，不是 DPT 原论文 encoder 的完整复现。

训练：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_adapted_dpt_shared.yaml
```

默认 physical/effective batch 均为 16，与方案一、二一致。OOM 时可临时使用 `--batch-size 4`，并在正式对比前将配置增加 `gradient_accumulation_steps: 4` 以保持 effective batch 16。

实现位置：

```text
models/decoders/galileo_dpt.py
models/cached.py
models/model.py
configs/galileo_adapted_dpt_shared.yaml
```

训练完成后补充：

| 项目 | 数值 |
| --- | --- |
| 配置与 commit | `configs/galileo_adapted_dpt_shared.yaml` / 待提交 |
| 可训练参数 | 18,337,427（约 18.34M） |
| 峰值显存 / 训练时间 | 待补 |
| best val mIoU / epoch | 待补 |
| fold5 test loss / mIoU | 待补 |
| per-class IoU 与定性观察 | 待补 |

该规模高于当前多层 DPT 的约 `9.06M`，因此比较结果时应同时报告参数量；若方案四更好，不能仅凭单次结果断言收益全部来自多尺度重组。

在数据产生前，不预写“优于 baseline”或“多尺度融合有效”等结论。

## 3D-Aware DPT 晚期融合（方案五）

方案五保持 PASTIS monthly-12 输入、Galileo 权重、冻结策略、fold、loss 和评测口径不变。它读取 `temporal_v2`，其中 Galileo 的 block 3/6/9 与经过最终归一化的 encoder output 只平均 band-group 轴，保留：

```text
[B, 4, T=12, D=768, 16, 16]
```

完整 decoder 包含：

1. **3D Reassemble**：四个隐藏层投影到 256 通道，并重组为 `64/32/16/8` 四个空间尺度，时间长度保持 12；final 同时保留 `[B,256,T,16,16]` 原尺度旁路。
2. **Global 3D bottleneck**：在最深 `8x8` 特征上执行全局时空自注意力。
3. **Native skip injection**：把 final 原尺度时间特征注入 `16x16` lateral，避免全局分支的 `16x16 -> 8x8` 成为唯一信息通路。
4. **Divided space-time blocks**：高分辨率阶段交替执行逐像素全局时间注意力、偏移窗口空间注意力和分解式 3D depthwise convolution。
5. **Gated DPT fusion**：从深到浅逐级空间上采样，以门控方式融合 lateral feature，全程保留 T。
6. **Temporal query pooling**：在 `64x64` 特征上用多头查询注意力融合 12 个月，最后输出 `[B, 19, 64, 64]`。

缓存训练入口：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_3d_aware_dpt.yaml --cache-format temporal_v2 --temporal-dtype float16
```

旧的在线入口 `scripts/train.py` 仍保留，用于复现已有实验。新路线只在缓存生成时运行一次冻结 Galileo；decoder 训练期间不再重复前向。两条路线使用独立日志和 checkpoint，结果不得混在同一次受控比较中。

### 方案五当前测试记录（待补完整审计信息）

组内已提供一组 fold5 测试输出：

```text
test_loss=1.03279
test_miou=0.59945
per_class_iou=0.73111,0.61606,0.82834,0.85413,0.73276,0.88384,0.56744,0.68865,0.49959,0.85782,0.36777,0.59229,0.36341,0.51119,0.36512,0.81174,0.40846,0.39818,0.31166
```

对应的逐类记录如下：

| Class ID | IoU |
| ---: | ---: |
| 0 | 73.111% |
| 1 | 61.606% |
| 2 | 82.834% |
| 3 | 85.413% |
| 4 | 73.276% |
| 5 | 88.384% |
| 6 | 56.744% |
| 7 | 68.865% |
| 8 | 49.959% |
| 9 | 85.782% |
| 10 | 36.777% |
| 11 | 59.229% |
| 12 | 36.341% |
| 13 | 51.119% |
| 14 | 36.512% |
| 15 | 81.174% |
| 16 | 40.846% |
| 17 | 39.818% |
| 18 | 31.166% |

这使它成为目前仓库中**已收到的最高 fold5 test mIoU 记录**：相对多层同尺度融合的 `48.005%` 高 `11.940` 个百分点。但在补齐以下信息前，只把它视为阶段性记录，不能据此完成最终模型排序或声称收益完全来自 3D decoder：配置文件、Git commit、seed、checkpoint 路径、best fold4 `val_mIoU` 与 epoch、训练/缓存目录，以及是否严格使用 `best_val_miou.pt`。

### CA-HPI M1 第一结果身份（待运行）

| 字段 | 固定值 |
| --- | --- |
| 实验 | M1：3D-Aware DPT + decoder 前 CA-HPI |
| Base config | `configs/galileo_3d_aware_dpt.yaml` |
| Prior overlay | `configs/prior_injection/ca_hpi_structured.yaml` |
| Prior version | `data/priors/pastis_ext_prior_v1.csv`（R1 v1） |
| Prior canonical SHA256 | `8BA07883D29A7112B16575C36A480C92D5DB232EE6BD8BE74ACF7C0A4BF6A0CC`（UTF-8/LF/单末尾换行） |
| Seed / folds | seed 42；train=fold1/2/3，val=fold4，test=fold5 |
| 训练上限 | 50 epoch；不根据 fold5 调整 |
| checkpoint | fold4 最高 mIoU 的 `best_val_miou.pt` |
| 首轮对照 | 上述 3D-Aware B0 `test_mIoU=0.59945`，暂作阶段性参考 |
| 状态 | 待服务器恢复、代码同步和正式训练 |

第一结果只要求填回：最佳 epoch、fold4 val 指标、fold5 一次性 test 指标、19 类 IoU/F1、诊断曲线和资源信息。FiLM、class-shuffled、多 seed 与文本先验均不阻塞本条结果。

### 空间重采样审计

- 方案一最终层卷积 baseline：在 decoder 内保持 `16x16`，只在输出前上采样到 `64x64`。
- 方案二多层同尺度 baseline：四层始终在 `16x16` 融合，只在输出前上采样。
- 方案三 UPerNet-style：PPM 的 `1/2/3/6` 池化是并行上下文分支，未池化的原始 `16x16` 特征作为第一个分支直接参与 concatenate。
- 方案四和方案五：保留 `8x8` 全局上下文分支，同时新增 final `16x16` 原尺度旁路；配置项为 `preserve_native_deep_skip: true`。

方案四、五的旁路复用已有 projection/refine，不增加参数量，也不新增 state-dict 键。旧 checkpoint 可以严格加载，但旧权重并未在新计算图下训练，正式比较应重新训练并使用带 `native_skip` 的独立日志目录；要复现旧输出，应把该开关设为 `false`。

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

它既可继续使用旧 `spatial_v1`，也可从 `temporal_v2` 现场求时间均值；后者需要重新训练并单独记录结果。

当前实现对最深的 layer 12 特征执行 `1/2/3/6` 四级 PPM，对 layer 3/6/9 执行 lateral projection，再由深到浅完成 FPN top-down 融合。PPM concatenate 的第一个分支就是未经池化的原始 layer 12 `16x16` 特征，所以池化上下文不会替换原始信息。由于四层 Galileo 特征均为 `16x16`，FPN 在这里融合的是 transformer 深度层级，而不是 CNN 的空间分辨率层级。

## 2026-07-17 Temporal Readout 与物候消融 fold5 test 记录

本节是项目正式实验数据记录。原始结果由组员汇总在 `test结果.md` 中，共包含四种 Temporal Readout 模型以及基于 Single-layer DPT Temporal Readout 的 P1/P2 物候消融。六次评估均保存了 `1984` 张 fold5 test tile 预测。

实验名称按配置和现有 checkpoint 命名核对如下：

| 简称 | 实验配置 | 先验设置 |
| --- | --- | --- |
| Single / P0 | `configs/galileo_single_layer_dpt_temporal_readout.yaml` | 无 |
| Multi | `configs/galileo_multi_layer_dpt_temporal_readout.yaml` | 无 |
| Adapted | `configs/galileo_adapted_dpt_temporal_readout.yaml` | 无 |
| UPerNet | `configs/galileo_upernet_temporal_readout.yaml` | 无 |
| P1 正确物候 | `configs/galileo_single_layer_dpt_temporal_readout.yaml` | `configs/phenology/external.yaml` |
| P2 类别置乱 | `configs/galileo_single_layer_dpt_temporal_readout.yaml` | `configs/phenology/external_class_shuffled.yaml` |

组内汇总确认，这批 P1/P2 的匹配无先验基线是 `Single / P0`。本地另有一个 `galileo_multi_layer_dpt_temporal_readout_seed42_phenology_external_cached` checkpoint 目录，但它不是这两条 test 结果的实验身份依据，不能据此改写组内汇总的模型映射。这批 Single 消融也不能与 3D-Aware DPT 记录混为一组。

### 总体指标

| 实验 | Test loss | Test mIoU | Test accuracy | Test macro F1 | Saved predictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single / P0 | 0.92940 | 0.54938 | 0.80589 | 0.68833 | 1984 |
| Multi | 0.95581 | **0.56154** | **0.81030** | **0.69712** | 1984 |
| Adapted | 0.92311 | 0.54637 | 0.80904 | 0.68433 | 1984 |
| UPerNet | **0.91416** | 0.54442 | 0.80342 | 0.68279 | 1984 |
| P1 正确物候 | 0.91485 | 0.55316 | 0.80698 | 0.69155 | 1984 |
| P2 类别置乱物候 | 0.92023 | 0.54784 | 0.80462 | 0.68703 | 1984 |

四种无先验 Temporal Readout 中，Multi 的 mIoU 和 macro F1 最高。物候消融的总体差值为：

| 比较 | Δ loss | Δ mIoU | Δ accuracy | Δ macro F1 |
| --- | ---: | ---: | ---: | ---: |
| P1 正确物候 − Single/P0 | -0.01455 | **+0.00378** | +0.00109 | +0.00322 |
| P2 类别置乱 − Single/P0 | -0.00917 | **-0.00154** | -0.00127 | -0.00130 |
| P1 正确物候 − P2 类别置乱 | -0.00538 | **+0.00532** | +0.00236 | +0.00452 |

这里的 mIoU 差值使用绝对比例；例如 `+0.00532` 等于 `+0.532` 个百分点。结果表明，P1 比匹配的 Single/P0 高 `0.378` 个百分点，并比 P2 高 `0.532` 个百分点；P2 则比 Single/P0 低 `0.154` 个百分点。当前单 seed test 可记录为：**正确物候在 Single Temporal Readout 上出现弱正信号，且正确类别对应优于置乱对应。** 但在补齐 fold4 validation、checkpoint 和多 seed 证据前，不能据此声称已有稳定增益。loss 与 mIoU 的排序并不一致，因此不能以较低 test loss 替代主要分割指标。

### 19 类 per-class IoU

下表保持 `data/pastis.py` 中 `PASTIS_CLASS_NAMES` 的类别顺序，数值为原始 IoU 比例。

| ID | 类别 | Single/P0 | Multi | Adapted | UPerNet | P1 正确 | P2 置乱 | P1−P0（pp） | P1−P2（pp） |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Background | 0.72552 | 0.73349 | 0.73197 | 0.72278 | 0.72721 | 0.72537 | +0.17 | +0.18 |
| 1 | Meadow | 0.62581 | 0.61716 | 0.62643 | 0.60311 | 0.62384 | 0.62686 | -0.20 | -0.30 |
| 2 | Soft winter wheat | 0.79749 | 0.79116 | 0.80363 | 0.80027 | 0.79247 | 0.78443 | -0.50 | +0.80 |
| 3 | Corn | 0.81835 | 0.82172 | 0.82114 | 0.81671 | 0.81623 | 0.81510 | -0.21 | +0.11 |
| 4 | Winter barley | 0.66343 | 0.68536 | 0.65876 | 0.66274 | 0.64757 | 0.64828 | -1.59 | -0.07 |
| 5 | Winter rapeseed | 0.83313 | 0.83483 | 0.83560 | 0.84032 | 0.84024 | 0.84143 | +0.71 | -0.12 |
| 6 | Spring barley | 0.48458 | 0.48010 | 0.46404 | 0.40404 | 0.47047 | 0.45114 | -1.41 | +1.93 |
| 7 | Sunflower | 0.54621 | 0.64480 | 0.58988 | 0.60207 | 0.59033 | 0.57793 | +4.41 | +1.24 |
| 8 | Grapevine | 0.55059 | 0.53505 | 0.51970 | 0.49182 | 0.52512 | 0.52259 | -2.55 | +0.25 |
| 9 | Beet | 0.81486 | 0.84216 | 0.81397 | 0.80618 | 0.82610 | 0.81975 | +1.12 | +0.63 |
| 10 | Winter triticale | 0.29716 | 0.28534 | 0.31062 | 0.27798 | 0.30914 | 0.28703 | +1.20 | +2.21 |
| 11 | Winter durum wheat | 0.53259 | 0.56191 | 0.54753 | 0.54905 | 0.54430 | 0.52884 | +1.17 | +1.55 |
| 12 | Fruits, vegetables, flowers | 0.35398 | 0.36837 | 0.31025 | 0.32491 | 0.35563 | 0.34311 | +0.16 | +1.25 |
| 13 | Potatoes | 0.42217 | 0.48306 | 0.37168 | 0.42169 | 0.45052 | 0.45818 | +2.84 | -0.77 |
| 14 | Leguminous fodder | 0.29995 | 0.30259 | 0.29801 | 0.27541 | 0.28730 | 0.29316 | -1.27 | -0.59 |
| 15 | Soybeans | 0.70842 | 0.75544 | 0.72655 | 0.74501 | 0.74097 | 0.72060 | +3.26 | +2.04 |
| 16 | Orchard | 0.42105 | 0.39981 | 0.40103 | 0.41266 | 0.38138 | 0.38782 | -3.97 | -0.64 |
| 17 | Mixed cereal | 0.29736 | 0.28819 | 0.30475 | 0.32317 | 0.32631 | 0.30757 | +2.90 | +1.87 |
| 18 | Sorghum | 0.24560 | 0.23871 | 0.24540 | 0.26403 | 0.25491 | 0.26982 | +0.93 | -1.49 |

P1 相对 Single/P0 在 `11/19` 类上升。提升最大的是 Sunflower（`+4.41 pp`）、Soybeans（`+3.26 pp`）、Mixed cereal（`+2.90 pp`）和 Potatoes（`+2.84 pp`）；下降最大的是 Orchard（`-3.97 pp`）、Grapevine（`-2.55 pp`）和 Winter barley（`-1.59 pp`）。

P1 相对 P2 在 `12/19` 类上升，其中 Winter triticale（`+2.21 pp`）、Soybeans（`+2.04 pp`）、Spring barley（`+1.93 pp`）和 Mixed cereal（`+1.87 pp`）差值最大；Sorghum（`-1.49 pp`）下降最大。这说明正确/置乱的知识内容会影响分类结果，但作用具有明显类别差异，仍需匹配验证集与多 seed 证据判断其稳定性。

### 19 类 per-class F1

| ID | 类别 | Single/P0 | Multi | Adapted | UPerNet | P1 正确 | P2 置乱 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Background | 0.84093 | 0.84626 | 0.84525 | 0.83909 | 0.84207 | 0.84083 |
| 1 | Meadow | 0.76985 | 0.76326 | 0.77032 | 0.75242 | 0.76835 | 0.77064 |
| 2 | Soft winter wheat | 0.88734 | 0.88341 | 0.89113 | 0.88905 | 0.88422 | 0.87919 |
| 3 | Corn | 0.90010 | 0.90213 | 0.90178 | 0.89911 | 0.89882 | 0.89813 |
| 4 | Winter barley | 0.79767 | 0.81331 | 0.79428 | 0.79717 | 0.78609 | 0.78661 |
| 5 | Winter rapeseed | 0.90897 | 0.90998 | 0.91044 | 0.91323 | 0.91318 | 0.91389 |
| 6 | Spring barley | 0.65282 | 0.64874 | 0.63391 | 0.57554 | 0.63989 | 0.62177 |
| 7 | Sunflower | 0.70651 | 0.78405 | 0.74204 | 0.75162 | 0.74240 | 0.73252 |
| 8 | Grapevine | 0.71017 | 0.69711 | 0.68395 | 0.65935 | 0.68863 | 0.68645 |
| 9 | Beet | 0.89799 | 0.91432 | 0.89744 | 0.89269 | 0.90477 | 0.90095 |
| 10 | Winter triticale | 0.45817 | 0.44399 | 0.47400 | 0.43503 | 0.47228 | 0.44604 |
| 11 | Winter durum wheat | 0.69502 | 0.71952 | 0.70762 | 0.70889 | 0.70492 | 0.69182 |
| 12 | Fruits, vegetables, flowers | 0.52287 | 0.53840 | 0.47357 | 0.49046 | 0.52467 | 0.51092 |
| 13 | Potatoes | 0.59369 | 0.65144 | 0.54194 | 0.59322 | 0.62119 | 0.62843 |
| 14 | Leguminous fodder | 0.46148 | 0.46460 | 0.45918 | 0.43187 | 0.44636 | 0.45340 |
| 15 | Soybeans | 0.82933 | 0.86069 | 0.84162 | 0.85387 | 0.85121 | 0.83762 |
| 16 | Orchard | 0.59259 | 0.57123 | 0.57248 | 0.58423 | 0.55217 | 0.55889 |
| 17 | Mixed cereal | 0.45840 | 0.44743 | 0.46714 | 0.48848 | 0.49205 | 0.47044 |
| 18 | Sorghum | 0.39435 | 0.38541 | 0.39410 | 0.41776 | 0.40627 | 0.42497 |

### 服务器预测输出目录

| 实验 | Output directory |
| --- | --- |
| Single / P0 | `/exstorage/gis2026/pressl-cropseg-mirror/output/test_single_layer` |
| Multi | `/exstorage/gis2026/pressl-cropseg-mirror/output/test_multi_layer` |
| Adapted | `/exstorage/gis2026/pressl-cropseg-mirror/output/test_adapted_dpt` |
| UPerNet | `/exstorage/gis2026/pressl-cropseg-mirror/output/test_upernet` |
| P1 正确物候 | `/exstorage/gis2026/pressl-cropseg-mirror/output/test_phenology_external` |
| P2 类别置乱物候 | `/exstorage/gis2026/pressl-cropseg-mirror/output/test_phenology_class_shuffled` |

### 审计状态与待补字段

| 字段 | 当前状态 |
| --- | --- |
| fold5 test 总体指标 | 已记录 |
| 19 类 IoU/F1 | 已记录 |
| 每组预测数量与输出目录 | 已记录 |
| P1/P2 匹配骨干 | 组内确认是 Single-layer DPT Temporal Readout |
| Seed | 配置和运行名称指向 `42`，正式归档仍应保存服务器命令 |
| 精确 checkpoint 路径 | 原始结果文件未附带；需从服务器 test 命令或日志补齐 |
| `best_val_miou.pt` 对应 epoch 与 fold4 val mIoU | 未提供 |
| Git commit | 未提供 |
| test cache 完整目录/manifest | 已知为 `temporal_v2` test cache，但原始结果文件未附完整命令 |

在上述审计字段补齐前，这批结果可以用于项目阶段汇总和方向判断，但不应被描述为完整可复现的最终论文表格。后续模型选择仍只能依据 fold4 validation，不能根据本节 fold5 test 排名反向调参。
