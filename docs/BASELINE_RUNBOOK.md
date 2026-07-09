# Baseline Runbook

本文档用于执行第一条可信 baseline：

```text
PASTIS fold3/fold4/fold5
  -> frozen Galileo base patch8 encoder
  -> single-layer spatial feature grid
  -> lightweight DPT-style decoder/head
  -> 20-class semantic segmentation
```

这条 baseline 的研究含义是：

> 在不微调 Galileo、也不额外手写 temporal fusion 的情况下，Galileo 单层时空表征能否被一个轻量空间 decoder 读出为有效的 PASTIS 作物分割结果。

它不是最终方法，也不是完整多层 DPT。它是后续所有改进和消融的参照点。

## 设备策略

优先使用本地 RTX 4060 Laptop 8GB 做环境检查、one-batch smoke test 和特征缓存小规模验证。

如果本地 8GB 显存无法完成完整 Galileo feature cache，则租用更大显存机器或交给组员运行。baseline 的命令和输出目录保持一致，便于不同机器之间合并结果。

建议策略：

```text
本地 8GB:
  1. check_env
  2. one-batch smoke test
  3. cache_features --split val 小规模观察

更大显存 / 组员机器:
  1. cache_features train/val/test
  2. train_cached
  3. eval_cached val/test
```

## 固定实验协议

默认配置：

```text
config: configs/galileo_dpt.yaml
encoder: pretrained/galileo-base-patch8
encoder.freeze: true
selected_timesteps: 24
patch_size: 8
image_size: keep 128x128
num_classes: 20
train: fold3
val: fold4
test: fold5
```

正式 baseline 不要临时改变：

- `selected_timesteps`
- `patch_size`
- `num_classes`
- fold 划分
- decoder 容量
- loss 权重

如果为了排查 OOM 临时改小 `selected_timesteps` 或 decoder，请把结果标记为 debug，不要和正式 baseline 混在一起。

## 运行前检查

确认数据和权重存在：

```text
data/PASTIS/metadata.geojson
data/PASTIS/DATA_S2/S2_*.npy
data/PASTIS/ANNOTATIONS/TARGET_*.npy

pretrained/galileo-base-patch8/config.json
pretrained/galileo-base-patch8/model.safetensors
pretrained/galileo-base-patch8/processing_galileo.py
pretrained/galileo-base-patch8/modeling_galileo.py
```

环境检查：

```bash
conda run -n presl python -B scripts/check_env.py --config configs/galileo_dpt.yaml
```

如果这里失败，先不要训练。优先修：

- PASTIS 路径
- Galileo checkpoint 路径
- `transformers` / `safetensors` / `torch` 依赖
- CUDA 是否可用

## 本地 8GB smoke test

先跑 one-batch，确认完整链路能 forward/backward：

```bash
conda run -n presl python -B scripts/train.py ^
  --config configs/galileo_dpt.yaml ^
  --batch-size 1 ^
  --epochs 1 ^
  --max-train-batches 1 ^
  --max-val-batches 1 ^
  --no-amp
```

成功标准：

- 没有 shape error。
- 没有 CUDA OOM。
- 输出 `train_loss`、`val_loss`、`val_miou`。
- `checkpoints/galileo_dpt/best.pt` 被写出。

如果 OOM：

1. 确认没有其他程序占用显存。
2. 确认 batch size 是 1。
3. 确认没有开启 AMP 造成额外不稳定；当前 baseline 默认 `--no-amp`。
4. 停止本地完整训练，转为更大显存机器跑 Galileo cache。

## 推荐正式路径：先缓存，再训练

因为 Galileo encoder 冻结，正式 baseline 推荐先缓存 features，再训练 decoder/head。

缓存 train：

```bash
conda run -n presl python -B scripts/cache_features.py ^
  --config configs/galileo_dpt.yaml ^
  --split train
```

缓存 val：

```bash
conda run -n presl python -B scripts/cache_features.py ^
  --config configs/galileo_dpt.yaml ^
  --split val
```

缓存 test，只在最终评估前做：

```bash
conda run -n presl python -B scripts/cache_features.py ^
  --config configs/galileo_dpt.yaml ^
  --split test
```

默认缓存目录：

```text
data/cache/galileo-base-patch8/t24_patch8_train/
data/cache/galileo-base-patch8/t24_patch8_val/
data/cache/galileo-base-patch8/t24_patch8_test/
```

缓存完成后训练 decoder/head：

```bash
conda run -n presl python -B scripts/train_cached.py ^
  --config configs/galileo_dpt.yaml ^
  --batch-size 4 ^
  --epochs 100 ^
  --no-amp
```

如果 decoder/head 训练也 OOM，把 `--batch-size 4` 改为 `2` 或 `1`。这只影响训练吞吐，不改变缓存的 Galileo features。

## 评估

缓存训练得到的 checkpoint 默认在：

```text
checkpoints/galileo_dpt_cached/best.pt
```

验证集评估：

```bash
conda run -n presl python -B scripts/eval_cached.py ^
  --config configs/galileo_dpt.yaml ^
  --checkpoint checkpoints/galileo_dpt_cached/best.pt ^
  --split val
```

测试集评估，只在模型和超参数全部固定后执行：

```bash
conda run -n presl python -B scripts/eval_cached.py ^
  --config configs/galileo_dpt.yaml ^
  --checkpoint checkpoints/galileo_dpt_cached/best.pt ^
  --split test
```

需要记录：

```text
val_loss
val_miou
val per_class_iou
test_loss
test_miou
test per_class_iou
```

## 组员或租算力交付清单

让组员跑训练时，只需要让对方回传这些内容：

```text
1. 使用的 git commit
2. 使用的 config 文件
3. GPU 型号和显存
4. cache_features train/val/test 是否完成
5. checkpoints/galileo_dpt_cached/best.pt
6. eval_cached val 输出
7. eval_cached test 输出
8. logs/galileo_dpt_cached/ 中的 TensorBoard 日志
```

不要只回传一句“跑通了”。baseline 的价值在于可复现的协议和可比较的指标。

## 结果命名建议

建议在记录表中使用这个实验名：

```text
galileo_base_patch8_frozen_single_layer_dpt_t24_fold3
```

含义：

- `galileo_base_patch8`: encoder 权重
- `frozen`: encoder 不训练
- `single_layer_dpt`: 当前 decoder 只读单层 feature grid
- `t24`: 最多 24 个时相
- `fold3`: train fold3, val fold4, test fold5

## 当前 baseline 的局限

这条 baseline 只回答第一层问题：

> Galileo 单层 frozen feature 是否有可读出的 PASTIS 分割信息？

它暂时不能回答：

- 多层 hidden states 是否更好。
- Galileo 是否显著优于 ImageNet baseline。
- 性能提升是否来自时间维建模。
- 哪些类别真正受益。

这些问题放到 baseline 稳定之后再做。
