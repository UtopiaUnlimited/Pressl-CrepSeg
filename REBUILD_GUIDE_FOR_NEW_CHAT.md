# PreSSL-CropSeg 从零重建说明

本文档用于开启新对话或新文件夹时，向新的代码助手说明项目背景、已经确认的技术判断，以及新的实现方向。不要直接沿用当前旧仓库的模型结构；旧仓库里存在若干为了先跑通而做的临时设计，尤其是 `temporal_chunk_size=4 -> temporal fusion -> DPT` 这条线，现在已经不是主方案。

## 1. 项目目标

我们要做的是基于遥感自监督模型 Galileo 的 PASTIS 作物/耕地语义分割实验。核心目标是：

```text
PASTIS Sentinel-2 time series
  -> frozen Galileo SSL encoder
  -> normal DPT segmentation decoder/head
  -> semantic mask logits
```

新的主线不是早期融合、晚期融合、决策融合三方案比较，而是：

```text
Galileo 内部负责时序建模
DPT 只负责空间 dense prediction
```

也就是说，时间维度 `T` 应该尽量在 Galileo encoder 内部消融，不再在下游手写 temporal fusion。

## 2. 必须先查官方实现

新项目开始时，第一步请查看 Galileo 官方 GitHub / 官方 Hugging Face 代码，重点找：

- 是否有官方 PASTIS downstream / linear probing / segmentation evaluation 代码。
- 是否有官方 PASTIS 数据预处理脚本。
- 是否有官方对 `s2`, `months`, `patch_size`, `mask` 的构造方式。

如果官方仓库中已经有 PASTIS 标准处理，请优先直接复用或严格对齐它，避免我们重新造一套输入处理。若找不到 PASTIS 专用代码，则以 Hugging Face 权重目录中的 `processing_galileo.py` / `GalileoProcessor` 为准。

已有本地 HF 格式权重：

- Hugging Face: <https://huggingface.co/BiliSakura/GALILEO-transformers>
- 本地子目录通常是：`pretrained/galileo-base-patch8`
- 其中包含：
  - `config.json`
  - `modeling_galileo.py`
  - `processing_galileo.py`
  - `pipeline_galileo.py`
  - `model.safetensors`

Galileo 论文说明其输入 tokenization 支持空间、时间、模态组的联合 token 序列；预训练样本是 `24` 个 monthly timesteps、`96x96` 像素、`10m/pixel`。但本项目不要把 PASTIS 图像 resize/crop 到 `96x96`，保持原始 `128x128`。

论文链接：<https://arxiv.org/abs/2502.09356>

## 3. 数据约定

PASTIS 本地数据结构：

```text
data/PASTIS/
  metadata.geojson
  NORM_S2_patch.json
  DATA_S2/
    S2_*.npy
  ANNOTATIONS/
    TARGET_*.npy
```

单个样本：

```text
S2:     [T, 10, 128, 128]
dates:  [T]
target: [128, 128]
```

类别数：

```text
num_classes = 20
```

因为本地 PASTIS target 标签值是 `0..19`。不要误改成 19 类。

建议实验划分：

```text
短学期统一对比：
train: fold3
val:   fold4
test:  fold5

标准参考划分：
train: fold1, fold2, fold3
val:   fold4
test:  fold5
```

训练/收敛判断只能看 train loss 和 val loss；test set 只用于最后一次报告，不能用于 early stopping 或调参。

## 4. Galileo 官方输入处理原则

旧项目里手写过 `_build_galileo_inputs`，但新项目应优先使用官方 processor：

```python
from transformers import AutoModel, AutoProcessor

processor = AutoProcessor.from_pretrained(
    "pretrained/galileo-base-patch8",
    trust_remote_code=True,
)

model = AutoModel.from_pretrained(
    "pretrained/galileo-base-patch8",
    trust_remote_code=True,
)
```

PASTIS 的 S2 输入要从：

```text
[T, 10, 128, 128]
```

转成官方 processor 需要的：

```text
[H, W, T, 10]
= [128, 128, T, 10]
```

示意：

```python
s2 = sample["S2"]                  # [T, 10, H, W]
s2 = s2.permute(2, 3, 0, 1)        # [H, W, T, 10]

inputs = processor(
    s2=s2,
    months=months_0_to_11,
    normalize=True,
    patch_size=8,
    return_tensors="pt",
)
```

官方 processor 会构造：

```text
space_time_x:    [B, H, W, T, 13]
space_time_mask: [B, H, W, T, 7]
space_x:         [B, H, W, 16]
time_x:          [B, T, 6]
static_x:        [B, 18]
months:          [B, T]
patch_size:      8
```

PASTIS 只有 S2，所以其它模态如 S1、ERA5、SRTM、static 等应通过 mask 标记为缺失。

重要细节：

- 不要对图像做 resize。
- 保持 `128x128` 输入。
- `patch_size=8` 时 token grid 是 `16x16`。
- 如果尝试 `patch_size=4`，token grid 是 `32x32`，但需要确认权重和显存。
- PASTIS dataloader 不建议先做自己的 `NORM_S2_patch.json` 标准化；如果使用官方 processor 的 `normalize=True`，应让 processor 做 Galileo 预训练统计量归一化，避免双重标准化。
- `months` 应该是 `0..11` 的月份索引，而不是 `1..12`。例如 January=0，December=11。

注意：本地 HF `processing_galileo.py` 定义了 NDVI 模态，但 `processor(s2=...)` 默认不会自动根据 S2 计算 NDVI。严格对齐 HF processor 时，NDVI 会被 mask 为缺失。若官方 GitHub 的 PASTIS 处理会额外计算 NDVI，则以官方 PASTIS 处理为准。

## 5. 时间维度设计

旧设计是：

```text
T 很长
-> 每 4 张切一个 chunk
-> Galileo 分别提 chunk 特征
-> 下游 temporal fusion 再融合
```

新设计推翻这个方向。新的主实验是：

```text
T selected = 24
-> 一次性输入 Galileo
-> Galileo 内部完成时空 attention
-> DPT 不再做 temporal fusion
```

原因：

- Galileo 本身就是时空多模态 SSL encoder。
- 如果后面再做 temporal fusion，会变成我们自己在 Galileo 后又建了一个时序模型，主线不干净。
- 老师建议不要根据 SSL 预训练图像大小去 resize 图像；但时间长度对 Galileo 是模型位置编码/预训练设定问题，可以按官方 `T=24` 对齐。

PASTIS 的 `T` 常见大于 24，例如 38、43、46、61。建议：

```text
如果 T > 24：按时间顺序 uniform sample 24 个时相
如果 T <= 24：保留全部，必要时 pad 并用 mask 处理，或直接 batch_size=1 不 pad
```

第一版为了简单稳妥：

```text
batch_size = 1
每个样本独立处理
T > 24 时 uniform sample 到 24
T <= 24 时直接输入
```

不要把 `128x128` 裁剪或缩放到 Galileo 预训练的 `96x96`。ViT/patch embedding 可以处理不同空间 token grid。

## 6. 模型结构主线

新项目只做：

```text
frozen Galileo SSL encoder + normal DPT segmentation decoder
```

不要再实现旧的三种方案：

- early fusion
- late fusion
- decision fusion

这些旧方案可以留作文档背景，但不要作为新项目默认代码。

### Galileo 输出怎么给 DPT

Galileo 原始输出不是四张特征图，而是 token sequence：

```text
last_hidden_state: [B, N_tokens, D]
```

DPT 理论上可以只用一层特征，也可以用多层 transformer hidden states。当前建议如下：

1. 优先尝试多层 hidden states 接 DPT。
2. 取 Galileo transformer 的若干中间层，例如第 `3, 6, 9, 12` 层。
3. 每层 token shape 本质上是相同的序列结构，可以 reshape / reassemble 成同一空间 grid。
4. DPT decoder 再对这些层级特征做 projection、fusion、upsampling。

这比旧代码里把同一个 `last_hidden_state` adaptive pooling 成 `F1..F4` 更合理。旧做法能跑，但学术表述不干净，因为那 4 个特征并不是 Galileo 原生多层输出。

如果短期实现多层 hidden states 太麻烦，可以先做单层版本：

```text
Galileo last_hidden_state
-> 取 S2/spatial tokens
-> reshape 为 [B, D, h, w]
-> lightweight DPT/head
-> [B, 20, 128, 128]
```

但文档里要清楚写明：这是 single-layer feature decoder，不是完整多层 DPT。

## 7. 工程结构建议

新项目结构参照 `segmentation_models.pytorch` 的解耦思想：

<https://github.com/qubvel-org/segmentation_models.pytorch>

不要把数据、模型、训练、loss、metrics 全塞进一个脚本。建议拆分：

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
  ce.py
  dice.py
  combined.py

metrics/
  miou.py
  confusion_matrix.py

train/
  trainer.py
  optim.py
  scheduler.py

scripts/
  train.py
  eval.py
  cache_features.py
  check_env.py
```

核心原则：

- pipeline / model / train / loss / metric 分离。
- encoder 冻结，默认 `requires_grad=False`。
- Galileo 输入处理放在 encoder wrapper 或 dataset adapter 中，但优先复用官方 processor。
- 不要把 PASTIS 大数据提交到 GitHub。

`.gitignore` 必须包含：

```text
data/PASTIS/
pretrained/
logs/
checkpoints/
__pycache__/
*.pyc
```

## 8. 训练设置

默认训练：

```text
encoder: Galileo base patch8
encoder.freeze: true
batch_size: 1
selected_timesteps: 24
image_size: keep 128x128
num_classes: 20
train: fold3
val: fold4
test: fold5
```

建议先关闭 AMP 做 smoke test：

```bash
python scripts/train.py --config configs/galileo_dpt.yaml --batch-size 1 --epochs 1 --max-train-batches 1 --max-val-batches 1 --no-amp
```

再跑正式训练：

```bash
python scripts/train.py --config configs/galileo_dpt.yaml --batch-size 1 --epochs 100 --no-amp
```

监控：

- TensorBoard 记录 train loss、val loss、mIoU、learning rate、GPU memory。
- 不用 test loss 判断收敛。
- 时间允许时训练到 train/val loss 基本不动为止。

优化器：

- 默认可先用 AdamW。
- 后续可加 Prodigy，对应建议配置：
  - `lr=1`
  - `weight_decay=0.1`
  - `decouple=True`
  - `slice_p=11`
- Prodigy 项目：<https://github.com/konstmish/prodigy>

## 9. 特征缓存

因为 Galileo encoder 冻结，建议后续加入 feature caching：

```text
PASTIS sample
-> Galileo forward once
-> save .npz
-> train DPT/head from cached features
```

建议缓存字段：

```text
patch_id
fold
dates
selected_indices
months
target
features or hidden_states
encoder_name
encoder_checkpoint
patch_size
selected_timesteps
normalization
```

缓存路径示例：

```text
data/cache/galileo-base-patch8/t24_patch8_fold3/
```

注意缓存目录也不要提交到 GitHub。

## 10. 新对话可直接使用的提示词

可以把下面这段直接发给新的代码助手：

```text
我们要在一个新文件夹从零重建 PreSSL-CropSeg。请不要沿用旧项目里 early/late/decision temporal fusion 的设计。新的主线是 frozen Galileo SSL encoder + normal DPT segmentation decoder，用 Galileo 自己的时空输入消融时间维度，DPT 只做空间 dense prediction。

请先查看 Galileo 官方 GitHub 和 Hugging Face 代码，重点找是否有官方 PASTIS 处理。如果有，优先复用官方 PASTIS preprocessing；如果没有，就用 Hugging Face 权重目录里的 GalileoProcessor / processing_galileo.py。PASTIS 的 S2 shape 是 [T,10,128,128]，要转为 processor 需要的 [H,W,T,10]。不要 resize 或 crop 图像到 96x96，保持 PASTIS 原始 128x128。Galileo 预训练使用 T=24，所以当 PASTIS T>24 时按时间 uniform sample 到 24；T<=24 时先 batch_size=1 直接处理或后续用 pad+mask。

encoder 冻结，只训练 DPT decoder/head。Galileo 原始输出是 token sequence，不是四张 F 特征图。DPT 可以先从 Galileo 的多层 hidden states 取第 3/6/9/12 层重排为空间特征；如果实现复杂，先做 single-layer last_hidden_state decoder，但要在文档里说明不是完整多层 DPT。

工程结构参考 segmentation_models.pytorch 的解耦思想：pipeline、model、train、loss、metrics 分开。数据用 PASTIS，num_classes=20，短学期默认 train fold3、val fold4、test fold5。test set 只用于最终评估，不能用于 early stopping。data/PASTIS、pretrained、logs、checkpoints 都要加入 .gitignore。
```

## 11. 旧项目中需要避免继承的问题

- 不要继续把 Galileo 输出说成原生 `F1..F4` 多尺度特征图。
- 不要把同一个 `last_hidden_state` adaptive pooling 成四个尺度再称为 Galileo 多尺度输出。
- 不要默认 `temporal_chunk_size=4` 后再接 temporal fusion；新主线应是 `T=24` 一次输入 Galileo。
- 不要对 PASTIS 图像做空间 resize。
- 不要对 S2 做 PASTIS norm 后又让 Galileo processor `normalize=True`，避免双重归一化。
- 不要把月份传成 `1..12`；应对齐官方 month embedding 的 `0..11`。
- 不要把 `num_classes` 改成 19；本地 target 是 `0..19`。

