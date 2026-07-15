# PreSSL-CropSeg

本项目研究 **frozen Galileo 遥感自监督表征在 PASTIS 作物语义分割上的迁移能力**。

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
- 多层 Galileo 特征共享缓存
- 冻结 Galileo 在线训练与梯度累积
- cached feature 训练与评估
- Galileo 论文式线性探测与学习率搜索
- 与 decoder 实验同训练协议的线性 head 对照
- 基于 fold4 `val_mIoU` 的可配置早停
- TensorBoard loss / val mIoU 日志

待完成实验：

- Galileo-Adapted 2D DPT、UPerNet-style 与 3D-Aware DPT 的正式训练、测试和多 seed 复验
- 结果和研究结论只在训练完成后补充

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
| `configs/galileo_3d_aware_dpt.yaml` | 使用 T=12 缓存或在线冻结 Galileo，训练带时间旁路的 3D-Aware DPT |
| `configs/galileo_linear_probe.yaml` | 使用最终层共享特征复现 Galileo 论文的 PASTIS 线性探测 |
| `configs/galileo_linear_decoder_shared.yaml` | 保留相同线性结构，但使用 decoder 对比实验的统一训练协议 |

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

物候先验旁路消融使用同一份 `temporal_v2` 缓存：

```bash
# 无先验 baseline
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_3d_aware_dpt.yaml --cache-format temporal_v2 --temporal-dtype float16

# 启用外部物候先验旁路
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_3d_aware_dpt_phenology_ext.yaml --cache-format temporal_v2 --temporal-dtype float16
```

两份配置的 encoder、输入时间序列和 temporal cache 相同，区别只有 decoder 是否构造 `PhenologyPriorAdapter`。旁路在每层 `Reassemble3D` 后、时间注意力和跨层融合前注入；修改 `phenology.path` 可切换到训练集 `P_data`。

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

训练会同时保存最低 `val_loss` 的 `best_val_loss.pt` 和最高 `val_mIoU` 的 `best_val_miou.pt`。为兼容已有命令，`best.pt` 与 `best_val_loss.pt` 含义相同。最终报告 mIoU 时建议评估 `best_val_miou.pt`，并运行多个 seed；论文式线性探测固定使用最后一轮 `last.pt`。

## 最终评估

调参只使用 fold4。模型、学习率和训练轮数固定后，再生成 fold5 test 缓存：

```bash
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_shared_cache.yaml --split test
```

评估 single-layer DPT：

```bash
conda run -n presl python -B scripts/eval_cached.py --config configs/galileo_single_layer_dpt_shared.yaml --checkpoint checkpoints/galileo_single_layer_dpt_shared_paper_input_bs16_rerun_seed42_cached/best_val_miou.pt --split test
```

评估会输出 `test_loss`、`test_miou`、`test_acc`、`test_f1`、逐类别 IoU 和逐类别 F1，同时把 test split 的每一个样本保存到项目根目录 `output/`。每张 PNG 从左到右依次为真实 RGB 合成图、真值和预测。可用 `--output-dir` 修改目录、用 `--panel-size` 修改单个面板尺寸；在线评估 `scripts/eval.py` 与缓存评估 `scripts/eval_cached.py` 的行为一致。

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
