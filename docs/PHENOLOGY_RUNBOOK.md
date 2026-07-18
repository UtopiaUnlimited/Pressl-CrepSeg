# [LEGACY] Global Add 物候先验训练运行手册

> **历史手册（2026-07-18 标记）：** 本文件只用于复现旧 `Global Add` overlay，不再是当前方法的操作入口。已经完成的 P0/P1/P2 基于 **Single-layer Temporal Readout**；文中的 3D-Aware 命令是当时计划的通用模板，不代表已有 3D-Aware 物候结果。当前唯一执行规划见 [NEXT_STAGE_HETEROGENEOUS_PRIOR_INJECTION_PLAN_2026-07-17.md](NEXT_STAGE_HETEROGENEOUS_PRIOR_INJECTION_PLAN_2026-07-17.md)。

旧方法研究摘要见 [PHENOLOGY_RESEARCH_LOGIC.md](PHENOLOGY_RESEARCH_LOGIC.md)；以下内容保留其运行、切换和评估命令，以便审计历史 checkpoint。

## 1. 先确认三件事

1. 在项目根目录拉取共享分支的最新提交：

   ```powershell
   git pull origin feat/paper-aligned-pastis-input
   ```

2. 使用装有 PyTorch 的 Conda 环境，并确认训练脚本识别公共先验参数：

   ```powershell
   conda run -n llm python -B scripts/train_cached.py --help
   ```

   输出中必须包含 `--phenology-config`。

3. 训练必须使用时间保留的 `temporal_v2` 缓存，不能使用已经平均时间的旧 `spatial_v1` 缓存。

每个缓存 `.npz` 至少应包含：

```text
temporal_features_by_layer: [4, 12, 768, 16, 16]
months: [12]
target: [64, 64]
```

## 2. 固定接口：只换两个变量

训练命令由两部分组成：

```text
--config <decoder route>
--phenology-config <prior overlay>   # P0 时省略
```

先验模块在 decoder 调用前统一处理 `[B,L,T,768,H,W]` 的 Galileo 特征；它不知道后面接的是哪一种 decoder。

### 可选 decoder 路线

| `--config` | 时间融合方式 |
| --- | --- |
| `configs/galileo_single_layer_dpt_temporal_readout.yaml` | Temporal Readout + Single-layer DPT |
| `configs/galileo_multi_layer_dpt_temporal_readout.yaml` | Temporal Readout + Multi-layer DPT |
| `configs/galileo_upernet_temporal_readout.yaml` | Temporal Readout + UPerNet |
| `configs/galileo_adapted_dpt_temporal_readout.yaml` | Temporal Readout + Galileo-Adapted DPT |
| `configs/galileo_3d_aware_dpt.yaml` | 3D-Aware DPT 自己进行内部晚期时间融合；不接 Temporal Readout |

不要把 `*_shared.yaml` 或其他已经将时间平均掉的旧 2D 配置用于物候实验。

### 可选先验状态

| 实验 | 先验参数 |
| --- | --- |
| P0：无先验 | 不传 `--phenology-config` |
| P1：正确外部先验 | `--phenology-config configs/phenology/external.yaml` |
| P2：类别置乱先验 | `--phenology-config configs/phenology/external_class_shuffled.yaml` |

P1/P2 overlay 会自动在日志和 checkpoint 目录名后增加 `_phenology_external` 或 `_phenology_external_class_shuffled`。缓存训练还会在最后增加 `_cached`，因此不同运行不会互相覆盖。

## 3. 设置本机变量

在**缓存所在机器的项目根目录**执行：

```powershell
$envName = "llm"
$trainCache = "D:\path\to\monthly12_tile64_patch4_hl3-6-9-12_temporal-v2_tfp16_train"
$valCache = "D:\path\to\monthly12_tile64_patch4_hl3-6-9-12_temporal-v2_tfp16_val"
$testCache = "D:\path\to\monthly12_tile64_patch4_hl3-6-9-12_temporal-v2_tfp16_test"

# 随时可替换的后续 decoder
$decoder = "configs/galileo_3d_aware_dpt.yaml"
```

例如，要切换到 UPerNet，只改一行：

```powershell
$decoder = "configs/galileo_upernet_temporal_readout.yaml"
```

无论换哪条路线，P0/P1/P2 必须使用同一份 train/val cache、相同 seed、batch size、loss、优化器和 early-stopping 设置。

## 4. 先做一次 P0 冒烟测试

冒烟测试不产生正式结果，只检查缓存键、张量形状、DataLoader 和 CUDA 是否匹配：

```powershell
conda run -n $envName python -B scripts/train_cached.py `
  --config $decoder `
  --train-cache-dir $trainCache `
  --val-cache-dir $valCache `
  --cache-format temporal_v2 `
  --temporal-dtype float16 `
  --max-train-batches 2 `
  --max-val-batches 2 `
  --epochs 1 `
  --device cuda
```

只有冒烟测试正常结束后，才开始正式训练。显存不足时可在这里和所有正式对照中统一加入 `--batch-size N`；不要只修改其中一组。

## 5. 正式训练：P0、P1、P2

### P0：无先验

```powershell
conda run -n $envName python -B scripts/train_cached.py `
  --config $decoder `
  --train-cache-dir $trainCache `
  --val-cache-dir $valCache `
  --cache-format temporal_v2 `
  --temporal-dtype float16 `
  --device cuda
```

### P1：正确外部先验

```powershell
conda run -n $envName python -B scripts/train_cached.py `
  --config $decoder `
  --phenology-config configs/phenology/external.yaml `
  --train-cache-dir $trainCache `
  --val-cache-dir $valCache `
  --cache-format temporal_v2 `
  --temporal-dtype float16 `
  --device cuda
```

### P2：类别置乱先验

```powershell
conda run -n $envName python -B scripts/train_cached.py `
  --config $decoder `
  --phenology-config configs/phenology/external_class_shuffled.yaml `
  --train-cache-dir $trainCache `
  --val-cache-dir $valCache `
  --cache-format temporal_v2 `
  --temporal-dtype float16 `
  --device cuda
```

这段命令当时曾计划以 `$decoder = "configs/galileo_3d_aware_dpt.yaml"` 跑 P0/P1/P2，但该计划没有形成当前台账中的 3D-Aware 物候结果。已经完成并记录的 P0/P1/P2 使用 Single-layer Temporal Readout。新 CA-HPI 不沿用本节的实验顺序。

## 6. 选择 checkpoint 与评估 test

训练过程会保存：

```text
best_val_miou.pt  # 正式模型选择使用它
best_val_loss.pt  # 仅作辅助诊断
best.pt           # 与旧流程兼容，指向最佳 val loss
```

因此应根据验证集 mIoU 选择 `best_val_miou.pt`，不要根据 test 结果挑 checkpoint。test 只在训练方案和 checkpoint 已固定后运行一次。

以下是旧计划中以 3D-Aware P1 为例的**目录模板**，不能据此推断服务器上存在对应 checkpoint：

```text
checkpoints/galileo_3d_aware_dpt_native_skip_late_fusion_seed42_phenology_external_cached/
```

评估必须复用**完全相同的** `$decoder` 与 `--phenology-config`，否则 checkpoint 的先验模块参数无法正确加载：

```powershell
$checkpoint = "checkpoints/galileo_3d_aware_dpt_native_skip_late_fusion_seed42_phenology_external_cached/best_val_miou.pt"

conda run -n $envName python -B scripts/eval_cached.py `
  --config $decoder `
  --phenology-config configs/phenology/external.yaml `
  --checkpoint $checkpoint `
  --cache-dir $testCache `
  --split test `
  --cache-format temporal_v2 `
  --temporal-dtype float16 `
  --device cuda `
  --output-dir output/phenology_p1_3d
```

P0 评估时删除 `--phenology-config ...`；P2 评估时替换为 `external_class_shuffled.yaml`，并使用对应的 checkpoint 目录。

## 7. 每次运行必须记录

在实验记录中保存以下项目：

```text
运行编号（P0/P1/P2）
decoder config 路径
phenology overlay 路径或 none
train / val / test cache 绝对路径
git commit
seed、batch size、梯度累积、AMP
best_val_miou.pt 的 epoch、val_loss、val_miou
一次最终 test 的 loss、mIoU、per-class IoU
```

不要根据 test mIoU 重新选择配置、epoch、先验曲线或 checkpoint。这样 P0/P1/P2 才是可解释的消融对照。
