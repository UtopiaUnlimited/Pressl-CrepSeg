# PreSSL-CropSeg

本项目研究 **frozen Galileo 遥感自监督表征在 PASTIS 作物语义分割上的迁移能力**。

当前主线固定为：

```text
PASTIS Sentinel-2 time series
  -> frozen Galileo SSL encoder
  -> cached / online Galileo spatial features
  -> decoder/head
  -> 20-class semantic segmentation logits
```

核心原则：**encoder 不变，只比较 decoder 如何读取同一套 Galileo 特征。**

## 当前状态

默认 baseline：

```text
frozen Galileo final feature grid
  -> single-layer DPT-style decoder/head
```

当前代码已经支持：

- single-layer DPT decoder
- multi-layer DPT decoder
- Galileo feature cache
- cached feature training / evaluation
- decoder-only 对比实验配置

尚未实现：

- UPerNet-style decoder 代码

UPerNet 的实验设计已经写在 [docs/DECODER_EXPERIMENTS.md](G:/Pressl-CrepSeg/docs/DECODER_EXPERIMENTS.md)。

## 重要文档

```text
docs/DECODER_EXPERIMENTS.md   # baseline 与 decoder-only 对比实验流程
docs/LITERATURE_REVIEW.md     # 相关文献和写作素材
```

## 数据与权重

PASTIS 数据放在：

```text
data/PASTIS/
  metadata.geojson
  NORM_S2_patch.json
  DATA_S2/S2_*.npy
  ANNOTATIONS/TARGET_*.npy
```

Galileo 权重放在：

```text
pretrained/galileo-base-patch8/
  config.json
  model.safetensors
  modeling_galileo.py
  processing_galileo.py
  pipeline_galileo.py
  preprocessor_config.json
```

这些目录不会提交到 Git：

```text
data/PASTIS/
pretrained/
data/cache/
logs/
checkpoints/
.hf_cache/
```

## 数据协议

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

关键约束：

- 不 resize、不 crop，保持 `128x128`。
- `T > 24` 时 uniform sample 到 24 个时相。
- 月份使用 `0..11`，不是 `1..12`。
- 使用 Galileo processor `normalize=True` 时，不再做 PASTIS norm。
- test set 只用于最终报告，不用于 early stopping 或调参。

## 配置文件

| 配置 | 用途 |
| --- | --- |
| `configs/galileo_dpt.yaml` | 默认 baseline：single-layer DPT，`hidden_layers: []` |
| `configs/galileo_shared_cache.yaml` | 生成 decoder 对比共享缓存，`hidden_layers: [3, 6, 9, 12]` |
| `configs/galileo_single_layer_dpt_shared.yaml` | 用共享缓存训练 single-layer DPT |
| `configs/galileo_multi_layer_dpt_shared.yaml` | 用共享缓存训练 multi-layer DPT |

`configs/galileo_dpt.yaml` 保持最小 baseline，不要为了 decoder 对比随手改乱。decoder 对比优先使用 shared 系列配置。

## 快速检查

```bash
conda run -n presl python -B scripts/check_env.py --config configs/galileo_dpt.yaml --try-model
```

one-batch smoke test：

```bash
conda run -n presl python -B scripts/train.py --config configs/galileo_dpt.yaml --batch-size 1 --epochs 1 --max-train-batches 1 --max-val-batches 1 --no-amp
```

## 默认 baseline

直接在线训练 baseline：

```bash
conda run -n presl python -B scripts/train.py --config configs/galileo_dpt.yaml --batch-size 1 --epochs 100 --no-amp
```

更推荐先缓存，再训练：

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

评估 cached baseline：

```bash
conda run -n presl python -B scripts/eval_cached.py --config configs/galileo_dpt.yaml --checkpoint checkpoints/galileo_dpt_cached/best.pt --split val
```

## Decoder 对比

老师建议当前阶段固定 encoder，只比较 decoder：

```text
1. single-layer DPT baseline
2. multi-layer DPT
3. UPerNet-style decoder  # 设计已写，代码未实现
```

公平做法是先生成一套共享缓存：

```bash
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_shared_cache.yaml --split train
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_shared_cache.yaml --split val
```

共享缓存目录会自动带 hidden layer 后缀：

```text
data/cache/galileo-base-patch8/t24_patch8_hl3-6-9-12_train/
data/cache/galileo-base-patch8/t24_patch8_hl3-6-9-12_val/
```

这套缓存同时包含：

```text
features           # final-layer spatial feature grid
features_by_layer  # layer 3/6/9/12 spatial feature grids
```

训练 single-layer DPT：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_single_layer_dpt_shared.yaml
```

训练 multi-layer DPT：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_multi_layer_dpt_shared.yaml
```

注意：如果旧缓存是用 `hidden_layers: []` 生成的，它没有 `features_by_layer`，不能给 multi-layer DPT 或 UPerNet-style decoder 用。

## 特征与编号说明

缓存文件按 PASTIS `patch_id` 命名。每个 fold 的 patch_id 本来就是离散的，所以看到这种跳号是正常的：

```text
fold4/val: 10002, 10004, 10008, 10011, ...
```

这不是 DataLoader shuffle，也不是缓存漏样本。

## 显存说明

Galileo 对 `128x128`、`T=24`、`patch_size=8` 的输入并不只是 `16x16` 个空间 token。S2 会被拆成多个 space-time group，默认 `spacetime_mean` 会按 `[H_grid, W_grid, T, group]` 聚合为空间特征图。

当前 wrapper 对官方 attention 做了一个保守优化：当有效 token mask 全为 True 时，不再把 mask 展开成 `[B, heads, N, N]`，而是传 `None` 给 PyTorch SDPA。这个优化不改变注意力语义，只避免冗余显存占用。

当前 wrapper 会逐样本调用 Galileo encoder，因此增大 `batch_size` 不会让 Galileo encoder 峰值显存严格乘以 batch size，但运行时间会接近线性增加；decoder/head、loss 和 metrics 的显存仍会随 batch size 增大。

## 一句话

本项目用 PASTIS 作物语义分割任务，研究 frozen Galileo 时空表征能否被不同 decoder 有效读出；当前优先固定 encoder，通过特征缓存公平比较 single-layer DPT、multi-layer DPT 和后续 UPerNet-style decoder。
