# PreSSL-CropSeg

本项目研究 **frozen Galileo 遥感自监督表征在 PASTIS 作物语义分割上的迁移能力**。

当前实验固定同一个 Galileo encoder 和同一套缓存，只比较 decoder：

```text
PASTIS Sentinel-2 time series
  -> Galileo 论文输入协议
  -> frozen Galileo encoder
  -> shared cached features
  -> single-layer DPT / multi-layer DPT / UPerNet-style decoder
  -> 19-class semantic segmentation logits
```

single-layer DPT 是当前 baseline。multi-layer DPT 和后续 UPerNet-style decoder 都必须读取同一版缓存，避免把输入变化误认为 decoder 收益。

## 当前实现

已经支持：

- Galileo 论文 PASTIS split 与输入协议
- single-layer DPT-style decoder
- multi-layer DPT-style decoder
- 多层 Galileo 特征共享缓存
- cached feature 训练与评估
- TensorBoard loss / val mIoU 日志

尚未实现：

- UPerNet-style decoder 代码

详细实验定义见 [docs/DECODER_EXPERIMENTS.md](docs/DECODER_EXPERIMENTS.md)。

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
| `configs/galileo_dpt.yaml` | 最小 single-layer DPT baseline，`hidden_layers: []` |
| `configs/galileo_shared_cache.yaml` | 生成 layer 3/6/9/12 共享缓存 |
| `configs/galileo_single_layer_dpt_shared.yaml` | 使用共享缓存训练 single-layer DPT |
| `configs/galileo_multi_layer_dpt_shared.yaml` | 使用共享缓存训练 multi-layer DPT |

四份配置都固定：

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

## 训练 Decoder

single-layer DPT baseline：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_single_layer_dpt_shared.yaml
```

multi-layer DPT：

```bash
conda run -n presl python -B scripts/train_cached.py --config configs/galileo_multi_layer_dpt_shared.yaml
```

TensorBoard（在项目根目录的另一个 PowerShell 终端运行；若已有旧服务，先在其终端按 `Ctrl+C`）：

```powershell
conda run -n presl tensorboard --logdir .\logs\galileo_single_layer_dpt_shared_paper_input_bs16_cached --port 6006 --reload_interval 1 --reload_multifile true --load_fast false
```

浏览器访问 `http://localhost:6006`。`--reload_interval 1` 让后端每秒扫描一次新日志；网页通常按自身周期刷新，也可以点击右上角圆形刷新按钮。右上角的 `INACTIVE` 是未启用面板菜单，不表示自动刷新已停止。`train_cached.py` 会在配置中的 `log_dir` 后自动追加 `_cached`；更换实验名后，TensorBoard 路径也要同步更换，避免混合不同实验曲线。

当前 checkpoint 仍按最低 `val_loss` 保存为 `best.pt`。正式论文结果还应同时保存最高 `val_mIoU` checkpoint 并运行多个 seed。

## 最终评估

调参只使用 fold4。模型、学习率和训练轮数固定后，再生成 fold5 test 缓存：

```bash
conda run -n presl python -B scripts/cache_features.py --config configs/galileo_shared_cache.yaml --split test
```

评估 single-layer DPT：

```bash
conda run -n presl python -B scripts/eval_cached.py --config configs/galileo_single_layer_dpt_shared.yaml --checkpoint checkpoints/galileo_single_layer_dpt_shared_paper_input_bs16_cached/best.pt --split test
```

定性查看 3 个 test tile（不需要先生成完整 test 缓存）：

```powershell
conda run -n presl python -B scripts/visualize_predictions.py --config configs/galileo_single_layer_dpt_shared.yaml --checkpoint checkpoints/galileo_single_layer_dpt_shared_paper_input_bs16_cached/best.pt --split test --num-samples 3
```

脚本只在线编码自动选出的 3 个样本，输出到 `outputs/test_predictions/`。每行从左到右依次为 12 个月 Sentinel-2 B4/B3/B2 中位数合成、真实标签和网络预测；真实标签与预测使用同一套类别颜色，void 区域显示为灰色。自动选图只依据真实标签的有效比例和类别丰富度，不查看模型预测；需要固定样本时可传入 `--sample-ids 10002_y0_x0 10002_y0_x64`。

## 实验原则

- decoder 对比只改变 `model.decoder` 及其专属结构参数。
- 所有 decoder 共用同一批 `monthly12_tile64_patch4_hl3-6-9-12` 缓存。
- 不改变 fold、输入缩放、月份、patch size、loss 或评测口径。
- test fold 只用于最终报告，不用于 early stopping 或选超参数。
- `data/PASTIS`、缓存、权重、日志、checkpoint 和 Python cache 都不提交到 Git。
