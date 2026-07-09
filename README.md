# PreSSL-CropSeg

本项目研究 **遥感自监督预训练模型 Galileo 在 PASTIS 作物/耕地语义分割任务中的表征迁移能力**。

我们关心的不是单纯把一个模型跑通，而是回答一个科研问题：

> Galileo 在大规模遥感自监督预训练中学到的时空表征，是否能够迁移到 PASTIS 作物语义分割？如果有效，增益来自时序/物候表征、光谱表征，还是 decoder 设计本身？

因此，当前项目的核心主线是：

```text
PASTIS Sentinel-2 time series
  -> frozen Galileo SSL encoder
  -> spatial DPT-style decoder/head
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

第一版实现是一个清晰的 baseline：

```text
frozen Galileo single-layer feature decoder
```

也就是说，当前代码使用 Galileo 的单层 token 输出，将空间 token 重排为特征图，再接 lightweight DPT-style decoder 和 segmentation head。它不是完整的多层 DPT，也不应被表述为 Galileo 原生多尺度特征金字塔。

后续更完整的方向是从 Galileo 多层 hidden states 中取若干中间层，例如第 3/6/9/12 层，再构造更接近 DPT 的多层 decoder。

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
target: [128, 128]
```

关键约束：

- 保持 PASTIS 原始 `128x128`，不 resize，不 crop。
- `T > 24` 时按时间顺序 uniform sample 到 24 个时相。
- `T <= 24` 时保留全部时相，当前优先用 `batch_size=1` 简化处理。
- S2 从 `[T, 10, H, W]` 转为 Galileo processor 所需的 `[H, W, T, 10]`。
- 月份使用 `0..11` 索引，例如 January = 0。
- 本地 PASTIS 标签值为 `0..19`，所以 `num_classes=20`，不要改成 19。
- 如果使用 Galileo processor 的 `normalize=True`，不要再对 S2 做 PASTIS norm，避免双重归一化。

默认短期数据划分：

```text
train: fold3
val:   fold4
test:  fold5
```

标准参考划分可作为后续补充：

```text
train: fold1, fold2, fold3
val:   fold4
test:  fold5
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

使用缓存训练：

```bash
conda run -n presl python -B scripts/train_cached.py ^
  --config configs/galileo_dpt.yaml ^
  --batch-size 4 ^
  --epochs 100 ^
  --no-amp
```

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
   - 完成 fold3/fold4 的短训练。
   - 记录 train loss、val loss、mIoU、显存和训练时间。
   - 明确当前结果属于 `single-layer feature decoder baseline`。

4. 建立对照实验
   - 增加或整理 ImageNet / non-Galileo baseline。
   - 保持相同 fold、decoder 容量、训练轮数和 loss。
   - 对比 Galileo frozen features 是否真正带来增益。

5. 加入特征缓存流程
   - 缓存 fold3 / fold4 的 Galileo features。
   - 验证缓存训练和非缓存训练结果一致。
   - 后续以缓存训练作为主要调试路径。

## 长期任务安排

长期目标是从“能跑的 baseline”推进到“论文级证据链”。

1. 多层 Galileo hidden states + DPT
   - 从 Galileo 中间层提取 hidden states。
   - 尝试第 3/6/9/12 层等组合。
   - 构建更合理的 DPT-style 多层 decoder。
   - 对比 single-layer decoder 与 multi-layer decoder 的差异。

2. 完整消融实验
   - Galileo frozen vs ImageNet frozen。
   - Galileo single-layer vs Galileo multi-layer。
   - T=1 / T=8 / T=16 / T=24。
   - processor normalize 策略对比。
   - fold3-only 协议与标准 fold1/2/3 协议对比。

3. 类别级分析
   - 输出 per-class IoU。
   - 分析哪些作物类别受益于 Galileo。
   - 观察提升是否集中在物候差异明显的类别。
   - 生成 confusion matrix 和代表性可视化样本。

4. 训练策略拓展
   - 比较 AdamW 与 Prodigy。
   - 评估 AMP / fp32 / bf16 稳定性。
   - 在显存允许时尝试部分解冻 Galileo 后几层。
   - 记录训练成本和性能收益。

5. 形成科研叙事
   - 明确研究问题、假设、方法、实验协议、结果和局限。
   - 将结果组织为表格和图。
   - 区分工程 smoke test、正式验证集结果、最终测试集结果。
   - 准备论文/开题/汇报材料中的方法图和实验表。


## 项目一句话

本项目通过 PASTIS 作物语义分割任务，研究 Galileo 遥感自监督时空表征的下游迁移价值，并逐步构建以 frozen Galileo encoder 和 DPT-style spatial decoder 为核心的可解释分割框架。
