# PreSSL-CropSeg

本项目研究 **遥感自监督预训练模型 Galileo 在 PASTIS 作物/耕地语义分割任务中的表征迁移能力**。

我们关心的不是单纯把一个模型跑通，而是回答一个科研问题：

> Galileo 在大规模遥感自监督预训练中学到的时空表征，是否能够迁移到 PASTIS 作物语义分割？如果有效，增益来自时序/物候表征、光谱表征，还是 decoder 设计本身？

因此，当前项目的核心主线是：

```text
PASTIS Sentinel-2 time series
  -> frozen Galileo SSL encoder
  -> final Galileo space-time feature grid
  -> single-layer DPT-style decoder/head
  -> 20-class semantic segmentation logits
```

## 研究定位

PASTIS 作物分割依赖多时相 Sentinel-2 影像。不同作物类别往往不能只靠单时相纹理区分，而需要利用全年光谱变化、物候节律和空间结构。

Galileo 是遥感时空自监督模型，预训练阶段已经见过大规模多模态、多时间遥感数据。本项目希望验证：冻结 Galileo encoder 后，仅训练轻量 decoder/head，是否仍能在 PASTIS 下游分割任务中提供有竞争力的表征。

当前版本不再沿用旧项目中的：

```text
temporal_chunk_size=4 -> temporal fusion -> DPT
```

旧路线可以跑通，但研究对象不够干净：如果下游再手写 temporal fusion，就难以区分性能来自 Galileo 本身，还是来自我们额外设计的时序融合模块。

新的默认假设是：

```text
Galileo 负责时空建模
DPT-style decoder 负责空间 dense prediction
```

## 当前实现状态

当前默认实现是一条尽量干净的 baseline：

```text
frozen Galileo final feature grid + single-layer DPT-style decoder
```

也就是说，当前代码使用 Galileo 最终输出的 space-time token，将其按 Galileo 的 `[H_grid, W_grid, T, group]` 结构聚合为 `16x16` 空间特征图，再接 lightweight DPT-style decoder 和 segmentation head。

多层 hidden states 接 DPT 已经保留了实验开关，但默认 baseline 不启用。也就是说，`model.decoder: single_layer_dpt` 和 `encoder.hidden_layers: []` 是当前主实验；如果后续改成 `multi_layer_dpt` 并设置 hidden layers，需要单独标记为多层 DPT 实验，不能和默认 baseline 混在一起。

## 数据约定

PASTIS 数据应放在：

```text
data/PASTIS/
  metadata.geojson
  NORM_S2_patch.json
  DATA_S2/
    S2_*.npy
  ANNOTATIONS/
    TARGET_*.npy
```

单个样本约定：

```text
S2:     [T, 10, 128, 128]
dates:  [T]
target: [128, 128]  # 由 TARGET_*.npy 的第 0 通道取得
```

关键约束：

- 保持 PASTIS 原始 `128x128`，不 resize，不 crop。
- `T > 24` 时按时间顺序 uniform sample 到 24 个时相。
- `T <= 24` 时保留全部时相，不做空间 resize/crop。当前 wrapper 会逐样本调用 Galileo encoder，因此默认仍建议 `batch_size=1` 起步排查。
- S2 从 `[T, 10, H, W]` 转为 Galileo processor 所需的 `[H, W, T, 10]`。
- 月份使用 `0..11` 索引，例如 January = 0。
- 本地 PASTIS 标签值为 `0..19`，所以 `num_classes=20`，不要改成 19。
- 如果使用 Galileo processor 的 `normalize=True`，不要再对 S2 做 PASTIS norm，避免双重归一化。

默认数据划分：

```text
train: fold1, fold2, fold3
val:   fold4
test:  fold5
```

如果为了本地 smoke test 或显存排查临时只用 fold3，必须把结果标记为 debug，不要和正式 baseline 混在一起。

当前默认优化器：

```text
optimizer: Prodigy
lr: 1.0
weight_decay: 0.1
decouple: true
slice_p: 11
scheduler: none
```

test set 只用于最终报告，不用于 early stopping、调参或模型选择。

## Galileo 权重

本项目使用 Hugging Face 格式 Galileo 权重：

```text
https://huggingface.co/BiliSakura/GALILEO-transformers
```

本地路径约定：

```text
pretrained/galileo-base-patch8/
  config.json
  model.safetensors
  modeling_galileo.py
  processing_galileo.py
  pipeline_galileo.py
  preprocessor_config.json
```

如果下载后出现：

```text
pretrained/galileo-base-patch8/galileo-base-patch8/
```

需要把内层 checkpoint 文件移动到 `pretrained/galileo-base-patch8/`。

## 工程结构

```text
configs/
  galileo_dpt.yaml

data/
  pastis.py
  transforms.py
  collate.py
  cached_features.py

models/
  encoders/
    galileo_hf.py
  decoders/
    dpt.py
  heads/
    segmentation.py
  model.py
  cached.py

losses/
metrics/
train/
scripts/
utils/
```

设计原则：

- dataset、model、loss、metric、trainer 分离。
- encoder 默认冻结，研究 Galileo 表征迁移，而不是端到端微调大模型。
- decoder/head 是 readout，不应承担复杂时序建模。
- `data/PASTIS/`、`pretrained/`、`data/cache/`、`logs/`、`checkpoints/` 不提交到 Git。

## 快速检查

第一条 baseline 的完整执行流程见：

```text
docs/BASELINE_RUNBOOK.md
```

Decoder-only 对比实验设计见：

```text
docs/DECODER_EXPERIMENTS.md
```

环境检查：

```bash
conda run -n presl python -B scripts/check_env.py --config configs/galileo_dpt.yaml
```

one-batch smoke test：

```bash
conda run -n presl python -B scripts/train.py ^
  --config configs/galileo_dpt.yaml ^
  --batch-size 1 ^
  --epochs 1 ^
  --max-train-batches 1 ^
  --max-val-batches 1 ^
  --no-amp
```

正式训练示例：

```bash
conda run -n presl python -B scripts/train.py ^
  --config configs/galileo_dpt.yaml ^
  --batch-size 1 ^
  --epochs 100 ^
  --no-amp
```

## 特征缓存

因为 Galileo encoder 默认冻结，后续建议优先使用特征缓存加速研究迭代：

```text
PASTIS sample
  -> Galileo forward once
  -> save .npz features
  -> train decoder/head from cached features
```

缓存 train / val：

```bash
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_dpt.yaml --split train
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_dpt.yaml --split val
```

默认缓存目录形如：

```text
data/cache/galileo-base-patch8/t24_patch8_train/
data/cache/galileo-base-patch8/t24_patch8_val/
```

注意：默认目录名没有包含 fold 列表。当前正式 train split 是 fold1/2/3；如果之前用 fold3-only 缓存过同名目录，请先清空旧 train cache，或用 `--output-dir` 指定新目录，避免不同 split 的缓存混在一起。

缓存文件以 PASTIS `patch_id` 命名。因为每个 fold 的 patch_id 本来就是离散分布，所以看到 `10002 -> 10004 -> 10008` 这类跳号是正常现象，不代表 DataLoader shuffle 或缓存漏样本。

使用缓存训练：

```bash
conda run -n presl python -B scripts/train_cached.py ^
  --config configs/galileo_dpt.yaml ^
  --batch-size 4 ^
  --epochs 100 ^
  --no-amp
```

使用缓存评估：

```bash
conda run -n presl python -B scripts/eval_cached.py ^
  --config configs/galileo_dpt.yaml ^
  --checkpoint checkpoints/galileo_dpt_cached/best.pt ^
  --split val
```

如果 `encoder.hidden_layers` 为空，缓存只保存默认 `features`，对应 single-layer decoder。如果设置了 hidden layers，缓存会额外保存 `features_by_layer` 和 `hidden_layers`，对应 multi-layer DPT 实验。

做 decoder-only 对比时，建议用 `hidden_layers: [3, 6, 9, 12]` 生成一套共享缓存：single-layer DPT 只读取其中的 `features`，multi-layer DPT 和 UPerNet-style decoder 读取 `features_by_layer`。这样 encoder 前向结果一致，变量更集中在 decoder。

## 显存与 batch size

Galileo 对 PASTIS `128x128`、`T=24`、`patch_size=8` 的单样本输入并不只是 `16x16` 个 token。S2 会被拆成多个 space-time group，默认 `spacetime_mean` 会在 Galileo 输出后按 `[H_grid, W_grid, T, group]` 聚合为空间特征图。

当前 wrapper 对官方 attention 做了一个保守优化：当所有保留下来的 token 都有效时，不再把全 True mask 展开成 `[B, heads, N, N]`，而是传 `None` 给 PyTorch SDPA。这个优化不改变有效 token 的注意力语义，只避免了冗余 mask 占用显存。

当前实现会对 batch 内样本逐个调用 Galileo encoder，所以增大 `batch_size` 不会让 Galileo encoder 的峰值显存严格按 batch size 翻倍，但运行时间会接近线性增加；decoder/head、loss 和 metrics 的显存仍会随 batch size 增大。正式训练和调参优先走特征缓存。

## 短期任务安排

短期目标不是追求最高指标，而是把研究装置固定下来，确保后续结果可信。

1. 明确并冻结主线
   - 默认路线固定为 `frozen Galileo encoder + spatial DPT-style decoder/head`。
   - 不再把 early/late/decision temporal fusion 作为当前主线。
   - README、汇报和代码注释统一使用同一套表述。

2. 验证数据和输入处理
   - 检查 PASTIS `S2`、`dates`、`target` shape。
   - 确认 `num_classes=20`。
   - 确认 `T>24` 的 uniform sampling 正确。
   - 确认月份为 `0..11`。
   - 确认没有空间 resize/crop。
   - 确认没有 PASTIS norm 和 Galileo processor normalize 双重归一化。

3. 跑通可信 baseline
   - 完成 one-batch smoke test。
   - 完成 fold1/2/3 -> fold4 的短训练。
   - 记录 train loss、val loss、mIoU、显存和训练时间。
   - 明确当前结果属于 `single-layer final feature + DPT baseline`。

4. 建立对照实验
   - 按老师建议，优先固定 Galileo encoder，只比较 decoder 设计。
   - 第一组对比：single-layer DPT vs multi-layer DPT。
   - 第二组对比：DPT-style decoder vs UPerNet-style decoder。
   - 保持相同 fold、loss、训练轮数和 cached feature 版本。

5. 加入特征缓存流程
   - 缓存 fold1/2/3 / fold4 的 Galileo features。
   - 验证缓存文件的 patch_id、fold、shape、target 与非缓存数据路径一致。
   - 后续以缓存训练作为主要调试路径。

## 长期任务安排

长期目标是从“能跑的 baseline”推进到“论文级证据链”。

1. 多层 Galileo hidden states + DPT
   - 从 Galileo 中间层提取 hidden states。
   - 尝试第 3/6/9/12 层等组合。
   - 构建更合理的 DPT-style 多层 decoder。
   - 对比 single-layer decoder 与 multi-layer decoder 的差异。

2. Decoder-only 对比实验
   - 保持 frozen Galileo encoder 不变。
   - Galileo single-layer DPT vs Galileo multi-layer DPT。
   - Galileo multi-layer DPT vs Galileo UPerNet-style decoder。
   - 使用相同 cached features，确保变量集中在 decoder。

3. 完整消融实验
   - Galileo frozen vs ImageNet frozen。
   - T=1 / T=8 / T=16 / T=24。
   - processor normalize 策略对比。
   - fold3-only 协议与标准 fold1/2/3 协议对比。

4. 类别级分析
   - 输出 per-class IoU。
   - 分析哪些作物类别受益于 Galileo。
   - 观察提升是否集中在物候差异明显的类别。
   - 生成 confusion matrix 和代表性可视化样本。

5. 训练策略拓展
   - 比较 AdamW 与 Prodigy。
   - 评估 AMP / fp32 / bf16 稳定性。
   - 在显存允许时尝试部分解冻 Galileo 后几层。
   - 记录训练成本和性能收益。

6. 形成科研叙事
   - 明确研究问题、假设、方法、实验协议、结果和局限。
   - 将结果组织为表格和图。
   - 区分工程 smoke test、正式验证集结果、最终测试集结果。
   - 准备论文/开题/汇报材料中的方法图和实验表。


## 项目一句话

本项目通过 PASTIS 作物语义分割任务，研究 Galileo 遥感自监督时空表征的下游迁移价值，并逐步构建以 frozen Galileo encoder 和 DPT-style spatial decoder 为核心的可解释分割框架。
