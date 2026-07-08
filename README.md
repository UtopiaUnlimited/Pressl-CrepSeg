# PreSSL-CropSeg

这是一个基于 PASTIS 的作物/耕地语义分割实验工程。当前重建版本的主线是：

```text
PASTIS Sentinel-2 时间序列
  -> 冻结的 Galileo SSL encoder
  -> 空间 DPT-style segmentation decoder/head
  -> 20 类语义分割 logits
```

本项目不再沿用旧的 `temporal_chunk_size=4 -> temporal fusion -> DPT` 路线。时间维度交给 Galileo encoder 内部建模，DPT 只负责空间 dense prediction。

## 当前实现

第一版实现使用 Galileo 的单层 token 输出，将空间 token 重排成特征图，再接 lightweight DPT-style decoder。它是 single-layer feature decoder，不是完整的多层 DPT。后续可以在 `models/encoders/galileo_hf.py` 中扩展多层 hidden states，再在 decoder 侧融合第 3/6/9/12 层等中间特征。

关键约束已经写进代码：

- PASTIS S2 输入保持原始 `128x128`，不 resize、不 crop。
- S2 从 `[T, 10, H, W]` 转为 Galileo processor 需要的 `[H, W, T, 10]`。
- `T > 24` 时按时间顺序 uniform sample 到 24 个时相。
- 月份使用 `0..11` 索引，例如 January=0。
- `TARGET_*.npy` 默认取第 0 通道作为语义标签，类别值为 `0..19`，即 `num_classes=20`。
- Galileo encoder 默认冻结，只训练 decoder/head。

## 工程结构

```text
configs/
  galileo_dpt.yaml

data/
  pastis.py
  transforms.py
  collate.py

models/
  encoders/
    galileo_hf.py
  decoders/
    dpt.py
  heads/
    segmentation.py
  model.py

losses/
metrics/
train/
scripts/
utils/
```

## 数据与权重

PASTIS 数据应放在：

```text
data/PASTIS/
  metadata.geojson
  DATA_S2/
    S2_*.npy
  ANNOTATIONS/
    TARGET_*.npy
```

Galileo Hugging Face 格式权重应放在：

```text
pretrained/galileo-base-patch8/
  config.json
  modeling_galileo.py
  processing_galileo.py
  pipeline_galileo.py
  model.safetensors
```

注意：`data/PASTIS/`、`pretrained/`、`logs/`、`checkpoints/`、`__pycache__/` 和 `*.pyc` 已加入 `.gitignore`，不会被普通 `git add` 提交。

## 下载 Galileo 权重

本项目使用指南中给出的 Hugging Face 仓库：

```text
https://huggingface.co/BiliSakura/GALILEO-transformers
```

可用 `huggingface_hub` 下载：

```bash
conda run -n presl python -B -c "import os; os.environ['HF_HOME']=r'G:\Pressl-CrepSeg\.hf_cache'; from huggingface_hub import snapshot_download; snapshot_download(repo_id='BiliSakura/GALILEO-transformers', local_dir=r'G:\Pressl-CrepSeg\pretrained\galileo-base-patch8')"
```

这个仓库会下载 `galileo-base-patch8`、`galileo-tiny-patch8`、`galileo-nano-patch8` 三个子目录。如果下载后 base checkpoint 位于：

```text
pretrained/galileo-base-patch8/galileo-base-patch8/
```

请把其中这些文件移动到上一层：

```text
pretrained/galileo-base-patch8/
  config.json
  model.safetensors
  modeling_galileo.py
  pipeline_galileo.py
  preprocessor_config.json
  processing_galileo.py
```

## 环境检查

```bash
conda run -n presl python -B scripts/check_env.py --config configs/galileo_dpt.yaml
```

如果 `pretrained/galileo-base-patch8` 已经准备好，可以做一轮 one-batch smoke test：

```bash
conda run -n presl python -B scripts/train.py --config configs/galileo_dpt.yaml --batch-size 1 --epochs 1 --max-train-batches 1 --max-val-batches 1 --no-amp
```

正式训练示例：

```bash
conda run -n presl python -B scripts/train.py --config configs/galileo_dpt.yaml --batch-size 1 --epochs 100 --no-amp
```

## 显存说明

在 RTX 4060 Laptop 8GB 上，Galileo base 甚至 nano 在 `128x128`、`T=24`、`patch_size=8` 下都会因为 Galileo 内部 attention 申请数十 GB 显存而 OOM。当前代码和权重加载已经验证通过，但完整 smoke train 需要更大显存、降低时空 token 数、使用特征缓存，或后续改成更省显存的 Galileo 调用策略。
