# 物候先验研究逻辑：一条主线，不是多套配置

## 1. 我们到底要回答什么问题

Galileo 已经通过自监督学习看到了多时相、多光谱序列，但下游分割任务未必能充分利用作物的物候规律。我们的研究问题不是“给某个 decoder 多加一层会不会变好”，而是：

> 在冻结 Galileo 且输入时间序列不变的条件下，人工整理的“类别-月份”物候知识，能否作为额外条件信息改善作物分割；不同的时间融合和空间 decoder 如何利用同一份信息？

所以必须把 **先验**、**时间融合**、**空间解码** 分开。它们是三个不同变量，不能在配置文件里揉成一个看不清来源的模型名字。

## 2. 唯一正确的数据流

### 2.1 固定输入：冻结 Galileo 的时间保留缓存

所有新实验读取同一份 `temporal_v2` 缓存：

```text
F[l, t] : [B, L, T, 768, 16, 16]
L = 4（Galileo 第 3、6、9、12 层）
T = 12（自然月）
```

`months[b, t]` 保存该时间步真实对应的自然月。这个缓存是所有实验的共同起点；不得让某些模型读取已经池化的 `[B, D, H, W]`，另一些模型读取保留 `T` 的特征。

### 2.2 公共先验旁路：先于任何 decoder

外部表为：

```text
P_ext[c, m] : [19, 12]
```

它描述“第 `c` 类作物在自然月 `m` 的相对辨识价值”。旁路根据当前月取出完整类别向量 `P_ext[:, m_t]`，经可学习适配器得到：

```text
q[t] : [B, T, 768, 1, 1]
F'[l, t] = F[l, t] + q[t]
```

这里注入的是 **Galileo 的 768 维特征空间**，不是 DPT 内部的 256 维空间。`q[t]` 在空间位置上广播，并共享给每个选中的 Galileo 层；随后各层和各 decoder 自己决定如何使用它。

这个旁路不读取像素真值类别，也不直接“筛掉”某类特征。它给模型的是该月的全局类别物候上下文；模型仍必须从图像特征中判断每个像素属于什么作物。

### 2.3 时间融合：与空间 decoder 分开

`temporal_readout` 和后面的空间 decoder 本来就是两个模块：

```text
F' [B,L,T,768,H,W]
  -> Temporal Readout R（沿 T 学习权重）
  -> G [B,L,768,H,W]
  -> Spatial Decoder S（DPT / UPerNet / Adapted DPT）
  -> segmentation logits
```

因此：

- `temporal_readout_upernet` = `R + UPerNet`；
- `temporal_readout_multi_layer_dpt` = `R + Multi-layer DPT`；
- `temporal_readout_single_layer_dpt` = `R + Single-layer DPT`；
- `temporal_readout_galileo_dpt` = `R + Adapted DPT`。

它们不是一个不可拆分的新 decoder，只是“时间模块 + 空间模块”的组合名称。

`3D-Aware DPT` 是另一种时间融合路线：它在 decoder 内保留 `T`，经过时空和跨层处理后才进行自己的时间聚合，因此属于**晚期融合**；它不使用独立的 `Temporal Readout`。但它仍然读取同样的 `F'`，所以先验旁路可以与它公平组合：

```text
F' -> 3D-Aware DPT -> segmentation logits
```

## 3. 配置应该如何组合

基础 decoder 配置只描述时间融合和空间解码，例如：

```text
configs/galileo_upernet_temporal_readout.yaml
configs/galileo_multi_layer_dpt_temporal_readout.yaml
configs/galileo_3d_aware_dpt.yaml
```

先验配置只描述先验表：

```text
configs/phenology/external.yaml
configs/phenology/external_class_shuffled.yaml
```

组合关系是：

```text
base decoder config + optional phenology overlay
```

因此同一份 `external.yaml` 应能接到任何保留 `T` 的路线，不需要再复制出 `phenology_upernet.yaml`、`phenology_dpt.yaml` 等完整配置。早期的 `galileo_3d_aware_dpt_phenology_*.yaml` 已删除，避免有人误把 3D-DPT 专用配置当成公共接口。

## 4. 实验不需要一次全部跑完

### 第一阶段：证明先验本身是否有信息

固定一个时间保留 decoder。当前选 `3D-Aware DPT`，因为它是现阶段验证集表现最好的候选；此时 decoder 不变，只改变先验：

| 实验 | 输入特征 | 先验 | 要回答的问题 |
| --- | --- | --- | --- |
| P0 | `F` | 无 | 固定 decoder 的严格基线 |
| P1 | `F'` | 正确 `P_ext` | 正确物候知识是否有增益 |
| P2 | `F'` | 固定 class-shuffled `P_ext` | 增益是否来自正确类别-月份对应，而非额外参数 |

若 P1 没有同时优于 P0 和 P2，就不能声称外部物候知识有效，也不值得立刻把矩阵扩到全部 decoder。

### 第二阶段：证明它不是 3D-DPT 的偶然现象

只有 P1 有可信信号后，选择一个 temporal-readout 组合（优先 `Temporal Readout + UPerNet` 或 `+ Multi-layer DPT`）重复 P0/P1。此时研究问题才变为：同一份先验是否能跨时间融合/空间解码方式工作。

不是把十几组实验一次堆上去，而是先验证因果链，再扩展泛化性。

## 5. 三个不能再混淆的点

1. `768` 是 Galileo 缓存特征维度，也是旁路注入维度；`256` 只是许多 decoder 的内部工作维度。
2. “公共旁路”指先验在 decoder 调用前注入，所有时间保留 decoder 都能复用；不等于所有 decoder 必须共享同一种时间融合方式。
3. `Temporal Readout` 与空间 decoder 是分开的。前者决定“12 个月怎样合成二维表示”，后者决定“怎样在空间上分割”。

## 6. 当前执行原则

先验训练只在新的“基础配置 + 独立先验 overlay”接口验证完成后启动。每次训练记录基础 decoder 配置、先验 overlay、cache 路径、seed、最佳验证 checkpoint 和最终 test 结果；不要再使用名称相近但架构边界不同的旧完整配置混跑。

### 通用运行模板

在缓存所在机器的项目根目录设置：

```powershell
$decoder = "configs/galileo_upernet_temporal_readout.yaml"
$prior = "configs/phenology/external.yaml"
$trainCache = "D:\path\to\temporal-v2_train"
$valCache = "D:\path\to\temporal-v2_val"

conda run -n llm python -B scripts/train_cached.py `
  --config $decoder `
  --phenology-config $prior `
  --train-cache-dir $trainCache `
  --val-cache-dir $valCache `
  --device cuda
```

只改 `$decoder` 就切换后续路线；只改 `$prior` 就切换先验。P0 不传 `--phenology-config`，P1 使用 `external.yaml`，P2 使用 `external_class_shuffled.yaml`。每个 overlay 会自动给日志和 checkpoint 目录追加后缀，避免不同先验运行互相覆盖。
