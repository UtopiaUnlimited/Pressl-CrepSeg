# PreSSL-CropSeg

本项目研究 **frozen Galileo 遥感自监督表征在 PASTIS 作物语义分割上的迁移能力**。

> 当前后半阶段已固定 3D-Aware DPT 为视觉基线，核心目标转为 V4 类别-时序原型记忆先验。请先阅读 [项目文档导航](docs/README.md) 和 [当前唯一执行规划](docs/NEXT_STAGE_V4_CLASS_TEMPORAL_PROTOTYPE_PLAN_2026-07-21.md)，不要从旧讲稿或旧 Global Add 手册生成后续任务。

V2 CA-HPI 与 V3 SA-SFiLM 保留为已实现对照：现有 M2/M3/M4 的 patch/区域级属性未显示稳定增益，不再继续增加外部环境数据。V4 将只从训练折构建类别-月份原型记忆，并以交叉建库、类别置乱和月份偏移对照检验其有效性。

当前实验固定同一个 Galileo encoder、输入协议和冻结策略，对比早期与晚期融合 decoder：

```text
PASTIS Sentinel-2 time series
  -> Galileo 论文输入协议
  -> frozen Galileo encoder
  -> spatial_v1: encoder 输出端固定 time mean（历史对照）
  -> temporal_v2: 保留四层逐月上下文化特征
  -> learned temporal readout / full 3D temporal decoder
  -> convolution / multi-layer / UPerNet / Galileo-DPT / 3D-Aware DPT
  -> 19-class semantic segmentation logits
```

现有 `single_layer_dpt` 与 `multi_layer_dpt` 是历史配置名，实际分别为最终层卷积 baseline 和多层同尺度融合 baseline，均不是完整 DPT。旧版方案一至四在 decoder 前对时间 token 固定求均值；新增版本从 `temporal_v2` 学习逐层、逐空间位置的月份权重，再进入相同二维空间 decoder。方案五 `3d_aware_dpt` 则在完整 3D DPT 多尺度融合后才消除时间维。

## 技术路线图

![PreSSL-CropSeg 技术路线图](pictures/project_technical_route.svg)

可编辑 SVG 源码：[pictures/project_technical_route.svg](pictures/project_technical_route.svg)。图中没有已经撤销的 paper-faithful DPT；第五列是阶段二新增的 3D-Aware DPT 晚期融合方案。

## 当前实现

已经支持：

- Galileo 论文 PASTIS split 与输入协议
- 最终层卷积 decoder（历史配置名 `single_layer_dpt`）
- 多层同尺度融合 decoder（历史配置名 `multi_layer_dpt`）
- Galileo-Adapted 2D DPT（四级 learned reassemble + 渐进融合）
- UPerNet-style decoder（PPM + FPN）
- 3D-Aware DPT：3D Reassemble、final 原尺度时间旁路、全局/分解时空注意力、门控多尺度融合与时间查询池化
- 方案一至四的月份感知可学习时间读出，以及对应的 `temporal_v2` 独立配置
- 通用 `PriorBatch`、structured phenology token encoder 与 CSV confidence 映射
- decoder 前共享 CA-HPI：内容感知 cross-attention、mask/confidence 和逐层零初始化门控
- 多层 Galileo 特征共享缓存
- 冻结 Galileo 在线训练与梯度累积
- cached feature 训练与评估
- Galileo 论文式线性探测与学习率搜索
- 与 decoder 实验同训练协议的线性 head 对照
- 基于 fold4 `val_mIoU` 的可配置早停
- TensorBoard loss / val mIoU 日志

当前研究状态：

- decoder 对比和四种 Temporal Readout 的阶段性 test 已收口，正式数值与审计缺口统一见实验台账；
- 3D-Aware DPT 固定为后续视觉基线，不再以新增 decoder 刷分为主线；
- decoder 前 CA-HPI 最小实现与单元测试已经完成，尚未产生正式训练结果。

详细实验定义见 [docs/DECODER_EXPERIMENTS.md](docs/DECODER_EXPERIMENTS.md)。两个既有 baseline 的首轮结果见 [docs/PROGRESS_REPORT_2026-07-11.md](docs/PROGRESS_REPORT_2026-07-11.md)；加入线性 head 并统一按最高 val mIoU 重测后的报告见 [docs/PROGRESS_REPORT_2026-07-13_LINEAR_COMPARISON.md](docs/PROGRESS_REPORT_2026-07-13_LINEAR_COMPARISON.md)。

## 数据与权重

PASTIS 放在：

```text
data/PASTIS/
  metadata.geojson
  DATA_S2/S2_*.npy
  ANNOTATIONS/TARGET_*.npy
```

Galileo Hugging Face 权重放在：

```text
pretrained/galileo-base-patch8/
  config.json
  model.safetensors
  modeling_galileo.py
  processing_galileo.py
  preprocessor_config.json
```

以下大文件目录保持 Git 忽略，不会上传：

```text
data/PASTIS/
data/cache/
pretrained/
.hf_cache/
logs/
checkpoints/
__pycache__/
```

## 论文输入协议

Galileo 官方论文和仓库中的 PASTIS 设置为：

```text
train: fold1 + fold2 + fold3
val:   fold4
test:  fold5

原图:          [T, 10, 128, 128]
月度聚合后:    [12, 10, 128, 128]
空间切块后:    [12, 10, 64, 64]，每张原图生成4个样本
Galileo patch: 4
特征图:        [D, 16, 16]
有效类别:      0..18，共19类
void label:    原始19 -> -1，在 loss 和 mIoU 中忽略
```

19 个有效类别是 1 个非农业背景类和 18 个农业/作物类别，逐类中英文名称及含义见 [`docs/next.md`](docs/next.md#pastis-的19个有效类别)。

当前 raw PASTIS 转换规则：

1. 去掉首尾不完整月份，取 `2018-10` 到 `2019-09` 的12个月作物年。
2. 同月多个 Sentinel-2 观测逐像素取均值。
3. 个别区域缺少 `2018-12`，使用相邻月线性插值。
4. 每个 `128x128` 样本按行优先切成四个 `64x64` 子块。
5. 使用 Galileo 官方 probing 的 `OURS + norm_no_clip + std_multiplier=2.0` 缩放。
6. 数据已完成官方缩放，因此 HF processor 设置 `encoder.normalize=false`，避免二次归一化。

官方评测数据类读取预生成的 `(N, 12, 13, 64, 64)` 张量。13波段包含为其他 baseline 补齐的 B1/B9/B10；Galileo 官方 wrapper 实际只选择 PASTIS 原生10波段，因此本项目直接保留原生10波段。

论文 Table 17 还会在验证集上搜索多种归一化统计量、缩放倍数和学习率，并运行5次。因此当前输入已经对齐论文公开协议，但要严格复现 `39.2% mIoU`，仍需补做论文的 normalization/LR sweep 和多次运行。

参考：

- [Galileo 官方论文](https://openreview.net/pdf?id=gqZO3eSZRy)
- [Galileo 官方仓库](https://github.com/nasaharvest/galileo)
- [官方 PASTIS 数据类](https://github.com/nasaharvest/galileo/blob/main/src/eval/datasets/pastis.py)

## 配置文件

常规配置与历史训练命令见下表；旧 `Global Add` overlay 的复现说明见 [LEGACY PHENOLOGY_RUNBOOK.md](docs/PHENOLOGY_RUNBOOK.md)，它不再是当前 CA-HPI 方法的操作入口。

| 配置 | 用途 |
| --- | --- |
| `configs/galileo_dpt.yaml` | 最终层卷积 baseline 的旧配置，`hidden_layers: []` |
| `configs/galileo_shared_cache.yaml` | 生成旧版 `spatial_v1` 共享缓存 |
| `configs/galileo_temporal_shared_cache.yaml` | 生成只保存四级完整 T 的 `temporal_v2` 共享缓存 |
| `configs/galileo_single_layer_dpt_shared.yaml` | 使用共享缓存训练最终层卷积 baseline |
| `configs/galileo_multi_layer_dpt_shared.yaml` | 使用共享缓存训练多层同尺度融合 baseline |
| `configs/galileo_upernet_shared.yaml` | 使用共享缓存训练 UPerNet-style decoder |
| `configs/galileo_adapted_dpt_shared.yaml` | 使用共享缓存训练带 final 原尺度旁路的方案四 Galileo-Adapted 2D DPT |
| `configs/galileo_single_layer_dpt_temporal_readout.yaml` | 使用完整 T 缓存训练带月份读出的方案一 |
| `configs/galileo_multi_layer_dpt_temporal_readout.yaml` | 使用完整 T 缓存训练带月份读出的方案二 |
| `configs/galileo_upernet_temporal_readout.yaml` | 使用完整 T 缓存训练带月份读出的方案三 |
| `configs/galileo_adapted_dpt_temporal_readout.yaml` | 使用完整 T 缓存训练带月份读出的方案四 |
| `configs/galileo_3d_aware_dpt.yaml` | 3D-Aware DPT 内部晚期融合配置；当前先验方法的固定视觉开发骨干 |
| `configs/prior_injection/ca_hpi_structured.yaml` | decoder 前 CA-HPI + 冻结 structured prior v1；通过 `--prior-config` 组合 |
| `configs/prior_injection/ca_hpi_m4_geography.yaml` | M4 地理上下文单来源对照；每个 patch 生成 1 个经纬度 token |
| `configs/prior_injection/ca_hpi_m1_m2_m3_balanced.yaml` | M1+M2+M3 的来源数量平衡匹配基线 |
| `configs/prior_injection/ca_hpi_m1_m2_m3_m4.yaml` | M1+M2+M3+M4，并启用来源数量平衡 |
| `configs/prior_injection/sa_spatial_film_m1_m2_m3_m4.yaml` | **当前主方法**：来源分层注意力 + decoder 前空间—通道 FiLM，使用冻结 M1/M2/M3/M4 |
| `configs/prior_injection/class_temporal_prototype_k1.yaml` | V4-K1：每个训练折类别—月份组一个原型，最终层时空 token 检索后门控写回 |
| `configs/prior_injection/class_temporal_prototype_k4.yaml` | V4-K4：每个训练折类别—月份组四个子原型，处理同类作物的类内多样性 |
| `configs/galileo_linear_probe.yaml` | 使用最终层共享特征复现 Galileo 论文的 PASTIS 线性探测 |
| `configs/galileo_linear_decoder_shared.yaml` | 保留相同线性结构，但使用 decoder 对比实验的统一训练协议 |

当前主方法缓存训练入口：

```bash
conda run -n presl python -B scripts/train_cached.py \
  --config configs/galileo_3d_aware_dpt.yaml \
  --prior-config configs/prior_injection/sa_spatial_film_m1_m2_m3_m4.yaml \
  --cache-format temporal_v2 \
  --temporal-dtype float16 \
  --device cuda
```

评估 checkpoint 时必须传入相同的 `--config` 与 `--prior-config`。旧 Global Add 继续使用 `--phenology-config`，两种参数禁止同时传入。

当前 overlay 默认启用轻量 CA-HPI 诊断。训练会把每层、按 train/val 分开的标量写入 TensorBoard 的 `prior/...`，并额外生成 `prior_diagnostics_history.json` 和 `prior_diagnostics_history.csv`。其中：

- `strength=tanh(raw_strength)` 是实际残差系数；
- `gate_mean/std` 与高低饱和比例描述内容门控；
- `attention_entropy` 是按有效 prior token 数归一化到 `[0,1]` 的熵，越接近 1 越均匀；
- `candidate_residual_ratio` 是门控候选残差相对视觉特征的范数；
- `applied_residual_ratio` 额外乘以 `abs(strength)`，才表示真正进入 decoder 的注入比例；
- `attended_confidence` 可与 `valid_confidence_mean` 比较，判断 attention 是否偏向高置信度知识。
- 多源平衡配置还按 overlay 中的 `name` 记录 `<source_name>/attention_mass` 与 `<source_name>/valid_token_fraction`。
- SA-SFiLM 额外记录 `film_scale_abs_mean`、`film_scale_std` 与 `film_shift_abs_mean`；多源 `<source_name>/attention_mass` 表示视觉内容决定的最终来源权重。

M4 的数据边界、通用 `patch_numeric_table` 接口及 `source_balance_bias_scale` 的定义见
[`docs/M4_GEOGRAPHIC_PRIOR_AND_SOURCE_BALANCING.md`](docs/M4_GEOGRAPHIC_PRIOR_AND_SOURCE_BALANCING.md)。

所有配置都固定 PASTIS 协议和 Galileo 权重；方案五额外设置 `preserve_temporal_features: true`：

```yaml
data:
  train_folds: [1, 2, 3]
  val_folds: [4]
  test_folds: [5]
  temporal_aggregation: monthly
  selected_timesteps: 12
  tile_size: 64
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

## 环境检查

```bash
conda run -n presl python -B -m unittest discover -s tests -v
conda run -n presl python -B scripts/check_env.py --config configs/galileo_shared_cache.yaml --try-model
```

预期数据检查结果：

```text
dataset_train_samples=5820
s2_shape=(12, 10, 64, 64)
months_minmax=0,11
target labels=-1..18
```

## 生成共享缓存

### 旧版 spatial_v1

旧的 `t24_patch8_*` 缓存不符合论文协议，不能在本分支继续使用。新目录名包含完整协议，不会覆盖旧缓存。

先生成 train 和 val：

```bash
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_shared_cache.yaml --split train
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_shared_cache.yaml --split val
```

默认目录：

```text
data/cache/galileo-base-patch8/monthly12_tile64_patch4_hl3-6-9-12_train/
data/cache/galileo-base-patch8/monthly12_tile64_patch4_hl3-6-9-12_val/
```

每个文件按原始 patch 和子块位置命名：

```text
10000_y0_x0.npz
10000_y0_x64.npz
10000_y64_x0.npz
10000_y64_x64.npz
```

每个缓存包含：

```text
features             # final layer [D, 16, 16]
features_by_layer    # layers 3/6/9/12 [4, D, 16, 16]
target               # [64, 64]，void 为 -1
patch_id / sample_id / tile_id / tile_y / tile_x
dates / months / aggregation_counts
```

缓存默认不保存巨大的完整 `hidden_state`。只有调试 token 时才显式添加 `--save-hidden-state`。

8GB 显存建议从共享配置的 `data.batch_size: 2` 开始；OOM 时命令行加 `--batch-size 1`。本协议的 `64x64 + T12 + patch4` 仍输出 `16x16` grid，但时序比旧缓存更短。

已有 `spatial_v1` 不需要删除。原来的方案一至四训练命令仍默认读取这些目录，可以马上继续训练和评估。

### 新版 temporal_v2

为了让方案一至五共享同一次 Galileo 编码，新缓存保留四个隐藏层完整的时间轴：

```text
temporal_features_by_layer  # block3/block6/block9/encoder_final
                            # [L=4, T=12, D=768, 16, 16]
target                      # [64, 64]，void 为 -1
patch_id / sample_id / tile_id / tile_y / tile_x
dates / months / aggregation_counts
```

`temporal_v2` **不再重复保存** `features`、`features_by_layer` 或 `hidden_state`。最深一级使用经过 Galileo final normalization 的 `encoder_final`，其余三级为 block 3/6/9。旧版方案一至四配置读取这种缓存时仍可现场求时间均值，用作固定读出对照；新增的四份 `temporal_readout` 配置则把完整 T 交给可学习月份读出模块。方案五直接让完整 T 进入 3D decoder。这里“不压缩 T”是指不做时间平均；文件仍使用 `np.savez_compressed` 做无损磁盘压缩。

生成命令：

```bash
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_temporal_shared_cache.yaml --split train
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_temporal_shared_cache.yaml --split val
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_temporal_shared_cache.yaml --split test
```

默认目录：

```text
data/cache/galileo-base-patch8/monthly12_tile64_patch4_hl3-6-9-12_temporal-v2_tfp16_train/
data/cache/galileo-base-patch8/monthly12_tile64_patch4_hl3-6-9-12_temporal-v2_tfp16_val/
data/cache/galileo-base-patch8/monthly12_tile64_patch4_hl3-6-9-12_temporal-v2_tfp16_test/
```

默认将 temporal feature 保存为 `float16`。单个样本的四级 T 特征原始大小约 18 MiB；真实 smoke 文件约 16.6 MiB，按全部 9,732 个子块估算约 157 GiB，实际总大小会随特征压缩率变化。若改用 `float32`，体积约翻倍；本地磁盘不足时应把 `data/cache` 链接到 Linux 服务器的大容量磁盘。

因为新缓存不保存时间均值，读取时必须解压完整 T 特征，I/O 会比旧缓存慢。已有空间 decoder 仍可优先使用旧缓存复现实验；新一轮方案一至五则统一使用 `temporal_v2` 做受控比较。

## 训练 Decoder

最终层卷积 baseline（历史配置名）：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_single_layer_dpt_shared.yaml
```

多层同尺度融合 baseline（历史配置名）：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_multi_layer_dpt_shared.yaml
```

UPerNet-style decoder：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_upernet_shared.yaml
```

方案四 Galileo-Adapted 2D DPT：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_adapted_dpt_shared.yaml
```

该方案直接读取共享缓存的 layer 3/6/9/12，将四个 `[B,768,16,16]` 特征 learned reassemble 为 `64/32/16/8` 四级金字塔。最深特征降到 `8x8` 只用于上下文分支；同一套投影与细化权重还会保留一份 final `16x16` 原尺度旁路，并把它注入 `16x16` 融合级，再由深到浅逐级恢复。它不重新运行 Galileo，也不覆盖方案一、二的日志和 checkpoint。

上述旧命令默认读取 `spatial_v1`。如需验证“同一新缓存现场均值是否能恢复旧接口”，可在原命令末尾添加：

```text
--cache-format temporal_v2 --temporal-dtype float16
```

例如方案四的固定均值对照：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_adapted_dpt_shared.yaml --cache-format temporal_v2 --temporal-dtype float16
```

### 方案一至四：可学习月份读出

新增的四个接口不再固定执行 `mean(T)`。Galileo 已经完成跨月份上下文化，月份读出模块只负责学习“当前空间位置更应该读取哪些月份”：

```text
[B, L, T, D, 16, 16]
  -> channel LayerNorm + month embedding + layer embedding
  -> 每层、每个空间位置预测 T 个分数
  -> softmax(T) 后对逐月特征加权求和
[B, L, D, 16, 16]
  -> 原方案一 / 二 / 三 / 四的二维空间 decoder
```

时间打分器的末层采用全零初始化，因此训练开始时12个月权重均为 `1/12`，数学上等价于旧的时间均值；训练后才允许不同隐藏层、不同地块位置学习非均匀月份权重。方案一只读出 `encoder_final`，方案二至四分别读出四层后再进入原空间结构。该模块不重复执行完整时间 Transformer，也不改变 Galileo 权重。

训练命令：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_single_layer_dpt_temporal_readout.yaml
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_multi_layer_dpt_temporal_readout.yaml
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_upernet_temporal_readout.yaml
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_adapted_dpt_temporal_readout.yaml
```

四份配置已经固定 `cache.format: temporal_v2` 和独立的日志、checkpoint 目录，不需要附加 `--cache-format`。默认 batch size 与各自旧配置一致；显存不足时可先在命令末尾添加 `--batch-size 2`，再按需要增加 `train.gradient_accumulation_steps` 维持有效 batch。

3D-Aware DPT 使用 `temporal_v2` 后不再在线重复运行 Galileo：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_3d_aware_dpt.yaml --cache-format temporal_v2 --temporal-dtype float16
```

### 方案五：时间维处理位置

当前方案五是**末端晚期融合**，不是“3D 模块处理一部分时间后，在 decoder 中段转成标准 2D DPT”。时间维在完整的四级 3D 金字塔中始终保留：

```text
[B, 4, T, 768, 16, 16]
  -> 3D Reassemble: [B, 256, T, 64/32/16/8]
  -> 8x8 Global Space-Time Attention
  -> 各尺度 Temporal Attention + Spatial Window Attention + Local 3D Conv
  -> Gated Cross-Scale Fusion: 8 -> 16 -> 32 -> 64（全程保留 T）
  -> Temporal Query Pool: [B, 256, T, 64, 64] -> [B, 256, 64, 64]
  -> 19-class Head: [B, 19, 64, 64]
```

最后的 `Temporal Query Pool` 不是固定均值池化。它在每个空间位置上把 12 个月组织为 `[T,256]` 序列，以“时间均值特征 + 可学习 query”为查询执行 8 头注意力，再生成一个内容相关的时间汇聚结果。月份嵌入、深层全局时空注意力、各尺度时间注意力、3D 局部卷积和跨尺度门控融合都发生在这一步之前，因此最终读出并不是方案五唯一的时间建模模块。

单张语义分割结果不含时间轴，所以 `T` 最终必须在某处消除；当前代码选择在完整 3D 金字塔融合后、二维分类头前消除。若后续改为“深层 3D 建模 -> 中段时间读出 -> 标准 2D DPT 恢复”，应作为独立的中段融合实验，使用新的配置、日志和 checkpoint，不能继续加载当前方案五权重。

> **历史物候实验说明：** 已完成的 P0/P1/P2 使用 Single-layer Temporal Readout 与旧 `Global Add` 旁路，并不是 3D-Aware DPT 结果。旧 overlay 命令和旁路架构图只用于复现历史实验，见 [LEGACY 运行手册](docs/PHENOLOGY_RUNBOOK.md) 与 [旧方案摘要](docs/PHENOLOGY_PRIOR_INJECTION_PLAN.md)。当前 CA-HPI 保留 decoder 前公共位置，但以逐视觉 token cross-attention 和置信度门控取代全局广播残差。

方案五同样只把 final 从 `16x16` 降到 `8x8` 用于全局时空注意力，同时保留 `[B,256,T,16,16]` 原尺度时间旁路并注入 `16x16` 融合级。方案四、五都通过 `model.preserve_native_deep_skip: true` 开启该行为，且复用已有层，不增加参数量或 state-dict 键。两份配置使用带 `native_skip` 的新日志与 checkpoint 目录，避免覆盖旧实验；复现旧结构时可将开关设为 `false`。

重采样审计结果：方案一和方案二在最终输出上采样前始终保持 `16x16`；方案三 UPerNet 的 PPM 是并行上下文分支，原始 `16x16` 特征仍直接参与拼接。这三种结构不存在“唯一特征先降到 `8x8`”的问题，因此不需要修改计算图。

如需复现实验五原来的在线冻结 Galileo 路线，旧命令仍保留：

```bash
conda run -n presl python -B scripts/train.py --config configs/galileo_3d_aware_dpt.yaml
```

该配置默认 physical batch 为 2、梯度累积 8 次，有效 batch 为 16。缓存训练和在线训练都只更新 3D-Aware DPT decoder，但缓存训练不再为每个 epoch 重复计算 Galileo。

旧缓存和新缓存属于两条明确的实验记录。旧 checkpoint 继续配旧 `spatial_v1` 评估；使用 `temporal_v2` 和可学习月份读出的方案一至四必须重新训练，不把新旧缓存结果混在一次受控对比中。

## 线性探测复现

线性探测是对 Galileo 原论文 probing protocol 的独立复现基线，不属于方案一至五的 decoder 设计对比。它读取共享缓存中的最终层 `features`，不加载无用的 `features_by_layer`：

```text
[B, 768, 16, 16]
  -> 每个 patch 一个 Linear(768, 19 * 4 * 4)
  -> 每个 patch 还原为 4x4 像素 logits
  -> [B, 19, 64, 64]
```

先跑固定学习率的单次实验：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_linear_probe.yaml
```

严格按论文附录 C.1 搜索 `{1, 3, 4, 5} x 10^{-4,-3,-2,-1}` 共 16 个学习率，并运行 5 个 seed：

```bash
conda run -n presl python -B scripts/sweep_linear_probe.py --config configs/galileo_linear_probe.yaml
```

每个候选固定训练 50 epoch；每个 seed 只用 fold4 最终 `val_mIoU` 选择学习率，随后在 fold5 测试一次。完整运行共训练 80 个候选模型，汇总写入 `outputs/linear_probe_sweep/results.json`。为保持与官方固定轮数协议一致，线性探测关闭早停，并保存 `last.pt`。

参考实现：[Galileo 官方 linear_probe.py](https://github.com/nasaharvest/galileo/blob/main/src/eval/linear_probe.py)。

### 同协议线性 head 对照

为了单独比较 decoder 结构，另提供一组只保留上述线性 head、其余设置与最终层卷积 baseline 相同的实验：batch size 16、CE+Dice、Prodigy 和 100 epoch。当前与其他配置一样关闭自动早停。

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_linear_decoder_shared.yaml
```

该实验应与方案一至四一起评估 `best_val_miou.pt`。它不是论文原始 linear probing 数值的复现；论文复现仍使用 `galileo_linear_probe.yaml` 和固定 50 epoch。

## 早停与模型保存

当前 `configs/` 下所有配置都关闭自动早停，训练会运行到 `train.epochs`，除非手动用 `Ctrl+C` 终止。早停机制仍保留；需要恢复时将对应配置改为：

```yaml
train:
  early_stopping:
    enabled: true
    monitor: val_miou
    mode: max
    patience: 12
    min_delta: 0.001
    start_epoch: 10
```

早停只减少无有效提升的后期训练，不改变最佳 checkpoint 的保存逻辑；它绝不能监控 fold5 test。已经启动的 Python 进程不会读取后来修改的配置，需要重启训练才会应用新开关。`Ctrl+C` 是终止进程而不是可恢复暂停：已经完成验证的最佳 `best_val_loss.pt` 和 `best_val_miou.pt` 会保留，但当前未完成 epoch、`last.pt` 和最终曲线文件不保证写出。

TensorBoard：

```powershell
conda run -n presl tensorboard --logdir logs
```

浏览器访问 `http://localhost:6006`。如果已经激活 `presl` 环境，也可以直接运行 `tensorboard --logdir logs`。

每轮验证还会从整个 fold4 的累计混淆矩阵计算像素准确率 `val_acc` 和宏平均 `val_f1`。其中 void/ignore 像素不参与统计，宏平均 F1 只平均在真值或预测中出现过的类别。训练控制台、TensorBoard 和 checkpoint 都会记录这两个指标。

正常训练结束（或重新启用后触发早停）后，不依赖 TensorBoard，程序会在该实验的 `train.log_dir` 下自动写出原始 epoch 数据和曲线：

- `training_history.json`、`training_history.csv`
- `train_loss.png`、`val_loss.png`
- `val_miou.png`、`val_acc.png`、`val_f1.png`
- `training_curves.png`（五项指标总览）

曲线不做平滑；mIoU、Acc 和 F1 的纵轴固定为 `[0, 1]`，loss 纵轴从 `0` 开始，避免自动缩放夸大较小的后期变化。

### 自选方案、指标和 epoch 绘图

`scripts/plot_training_curves.py` 可以读取 TensorBoard event；即使训练被手动停止、尚未生成 `training_history.json`，也能按当前已经完成的 epoch 画图。先列出 `logs/` 中可用的运行：

```powershell
conda run -n presl python -B scripts/plot_training_curves.py --list-runs
```

例如对比方案一、二在前 60 个 epoch 的 train loss、val loss 和 val mIoU：

```powershell
conda run -n presl python -B scripts/plot_training_curves.py --runs 1 2 --metrics train_loss val_loss val_miou --max-epoch 60 --output output/scheme1_vs_scheme2_epoch60.png
```

只画方案五第 10 至 50 个 epoch 的 val mIoU，并自动放大纵轴：

```powershell
conda run -n presl python -B scripts/plot_training_curves.py --runs 5 --metrics val_miou --min-epoch 10 --max-epoch 50 --auto-y --output output/scheme5_val_miou.png
```

`--runs` 支持方案别名 `1..5`、`scheme1..scheme5`、日志目录名或 `标签=日志目录名`；`--metrics` 可选 `train_loss`、`val_loss`、`val_miou`、`val_acc`、`val_f1` 和 `lr`。多项指标默认纵向排列，可用 `--columns 2` 改为两列；`--smoothing-window 5` 表示 5 个 epoch 的尾随滑动平均，默认值 `1` 保留原始曲线。需要固定纵轴时可传入如 `--y-limit val_miou=0.3:0.7`。

训练会同时保存最低 `val_loss` 的 `best_val_loss.pt` 和最高 `val_mIoU` 的 `best_val_miou.pt`。为兼容已有命令，`best.pt` 与 `best_val_loss.pt` 含义相同。最终报告 mIoU 时建议评估 `best_val_miou.pt`，并运行多个 seed；论文式线性探测固定使用最后一轮 `last.pt`。

## 最终评估

调参只使用 fold4。模型、学习率和训练轮数固定后，再生成 fold5 test 缓存：

```bash
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_shared_cache.yaml --split test
```

上面生成的是旧实验使用的 `spatial_v1` test 缓存。方案一至五的新时间保留实验必须生成或下载 `temporal_v2` test 缓存：

```bash
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_temporal_shared_cache.yaml --split test
```

评估 single-layer DPT：

```bash
conda run -n presl python -B scripts/eval_cached.py --config configs/galileo_single_layer_dpt_shared.yaml --checkpoint checkpoints/galileo_single_layer_dpt_shared_paper_input_bs16_rerun_seed42_cached/best_val_miou.pt --split test
```

评估会输出 `test_loss`、`test_miou`、`test_acc`、`test_f1`、逐类别 IoU 和逐类别 F1。当前 fold5 包含 496 个原始 `128x128` patch，每个 patch 被切为 4 个 `64x64` tile，因此 `scripts/eval.py` 和 `scripts/eval_cached.py` 默认都会遍历并保存全部 **1984 个 test tile**。每个 tile 在项目根目录 `output/` 中生成一张 PNG，从左到右依次为 RGB 合成图、真值和预测，文件名形如 `test_10002_y0_x64.png`。

当前评估脚本不会自动把四个 tile 拼回一张 `128x128` 原始 patch 图，也没有“只计算指标但不保存图片”的开关。可用 `--output-dir` 修改目录、用 `--panel-size` 修改单个面板尺寸；若只需要少量定性结果，应使用后面的 `visualize_predictions.py --num-samples N`。

使用 `temporal_v2` 评估 3D-Aware DPT：

```bash
conda run -n presl python -B scripts/eval_cached.py --config configs/galileo_3d_aware_dpt.yaml --checkpoint checkpoints/galileo_3d_aware_dpt_native_skip_late_fusion_seed42_cached/best_val_miou.pt --split test --cache-format temporal_v2 --temporal-dtype float16
```

评估旧的在线方案五 checkpoint 时，应先在配置中把 `model.preserve_native_deep_skip` 设为 `false`，再使用：

```bash
conda run -n presl python -B scripts/eval.py --config configs/galileo_3d_aware_dpt.yaml --checkpoint checkpoints/galileo_3d_aware_dpt_late_fusion_seed42/best_val_miou.pt --split test
```

如果只想定性查看 3 个有代表性的 test tile（不需要先生成完整 test 缓存）：

```powershell
conda run -n presl python -B scripts/visualize_predictions.py --config configs/galileo_single_layer_dpt_shared.yaml --checkpoint checkpoints/galileo_single_layer_dpt_shared_paper_input_bs16_rerun_seed42_cached/best_val_miou.pt --split test --num-samples 3
```

显式传入 `--num-samples 3` 时，脚本只在线编码自动选出的 3 个样本；省略该参数时默认输出完整 split。结果默认写入 `output/`。每行从左到右依次为 12 个月 Sentinel-2 B4/B3/B2 中位数合成、真实标签和网络预测；真实标签与预测使用同一套类别颜色，void 区域显示为灰色。自动选图只依据真实标签的有效比例和类别丰富度，不查看模型预测；需要固定样本时可传入 `--sample-ids 10002_y0_x0 10002_y0_x64`。

## 实验原则

- decoder 对比只改变 `model.decoder` 及其专属结构参数。
- 旧实验中方案一至四共用 `spatial_v1`；新实验中方案一至五共用只保存 T 特征的 `temporal_v2`。
- 不改变 fold、输入缩放、月份、patch size、loss 或评测口径。
- test fold 只用于最终报告，不用于 early stopping 或选超参数。
- `data/PASTIS`、缓存、权重、日志、checkpoint 和 Python cache 都不提交到 Git。

## Linux 运行

训练代码使用相对路径和 `pathlib`，核心训练、缓存和评估脚本不依赖 Windows API，复制到 Linux 后通常不需要改 Python 代码。`.npz` 缓存和 PyTorch checkpoint 也可以在 Windows 与 Linux 之间读取。

需要注意：

1. Linux 文件名区分大小写，数据目录必须保持 `DATA_S2`、`ANNOTATIONS`、`INSTANCE_ANNOTATIONS` 和 `metadata.geojson` 的原始大小写。
2. `data.root` 和 `encoder.checkpoint` 默认是相对仓库根目录的路径；从仓库根目录执行命令即可。
3. 根据服务器 CUDA 版本安装对应的 PyTorch，再执行 `pip install -r requirements.txt`。不要直接复制 Windows conda 环境目录。
4. `.hf_cache`、`logs`、`checkpoints` 和 `data/cache` 必须位于有写权限的磁盘。
5. Linux 的 DataLoader 通常可以把 `num_workers` 调高，但先从配置值开始，确认共享内存 `/dev/shm` 足够后再增加。

大缓存建议放到服务器数据盘并建立软链接：

```bash
mkdir -p /mnt/large_disk/Pressl-CrepSeg-cache
mkdir -p data
ln -s /mnt/large_disk/Pressl-CrepSeg-cache data/cache
```

环境和路径检查：

```bash
conda activate presl
python -B scripts/check_env.py --config configs/galileo_temporal_shared_cache.yaml --try-model
python -B -m unittest discover -s tests -v
```

仓库中 `presentation/**/build` 的少量历史报告可能记录 Windows 绝对路径，但它们只是生成物，不会被训练代码读取。
