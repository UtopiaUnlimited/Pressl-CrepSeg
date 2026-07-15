# 多层 Galileo 特征中的物候先验处理

## 1. 先把三个维度分开

在多层时序特征中，推荐使用下面的记号：

```text
F[l, t] = Galileo 第 l 个中间层、自然月份 t 的空间特征
P[c, m] = 类别 c 在自然月份 m 的物候先验
```

- `l` 是网络深度，例如第 3、6、9、12 层；
- `t` 是样本序列中的时间位置；
- `m` 是 `months[:, t]` 对应的自然月份；
- `c` 是 PASTIS 类别；
- `P_ext` 的定义只能是 `类别 × 自然月份`，不应该变成 `层 × 类别 × 月份`。

中间层不是四套不同的物候知识。它们看到的是同一组月份，只是语义抽象程度和空间上下文不同。

## 2. 当前代码中的两条分支

### 2.1 早期时间融合：普通多层 DPT/UPerNet

`multi_layer_dpt`、UPerNet 以及相近的早期融合配置读取：

```text
features_by_layer[l] -> [B, D, H_l, W_l]
```

这里的时间维已经在 encoder 的空间特征重组阶段被聚合。此时无法再对月份 `m_t` 做严格的逐月先验注入，因为月度对应关系已经丢失。

因此有两个选择：

1. 把这类模型作为“无显式物候先验”的 decoder 对照；
2. 重新生成 `temporal_v2` 缓存，让每一层保留 `[B, T, D, H_l, W_l]`，再接带时间处理的 decoder。

不能在 `[B,D,H,W]` 上强行复制一份 `P_ext`，然后声称模型使用了逐月物候知识；那只会变成一个没有时间定位的类别偏置。

### 2.2 保留时间：`3d_aware_dpt`

当前 `3D-Aware DPT` 读取：

```text
temporal_features_by_layer[l] -> [B, T, D, H_l, W_l]
months                     -> [B, T]
```

它目前已经加入了：

- `month_embedding`：自然月份编码；
- `layer_embedding`：第几层的深度编码；
- 每一层的时空重组、跨尺度融合和最后的 temporal pooling。

因此物候先验应在保留 `T` 的分支中加入，且对每个层使用同一份 `P_ext[:, month_t]`。层之间只允许有可学习的投影或调制参数，不允许手工指定不同的物候月份。

## 3. 推荐的最小注入位置

对每个时间步先取：

```text
p_t = P_ext[:, month_t]       # [B, K]
q_t = MetaEncoder(month_t, p_t) # [B, C]
```

其中 `K=19`，`C` 是 decoder 的通道数。不能把 `[B,K]` 直接和 `[B,D,H,W]` 相加，需要一个可学习的 `MetaEncoder` 投影到 decoder 通道。

在每个 Galileo 层重组后、时空 block 之前注入：

```text
F_l,t = Reassemble_l(F_l,t)
F_l,t = F_l,t + lambda_l * q_t[:, :, None, None]
F_l,t = F_l,t + month_embedding(month_t)
F_l,t = F_l,t + layer_embedding(l)
```

更强但更复杂的版本是 FiLM/AdaGN：

```text
F'_l,t = gamma_l(q_t) * Norm(F_l,t) + beta_l(q_t)
```

建议第一版使用共享 `MetaEncoder` 加一个可学习的 `lambda_l`，并把 `lambda_l` 初始化为 0 或很小的值，让模型可以退回当前 3D-Aware DPT。这样最容易判断提升是否来自物候分支，而不是额外参数强行改变了 decoder。

## 4. 为什么不直接做 Decision Fusion

概念上的 class-month Decision Fusion 需要每个时间步有类别 logits：

```text
Z_t -> [B, K, H, W]
```

但当前 `3D-Aware DPT` 的时间顺序是先把多层特征融合，再经过 `TemporalQueryPool`，最后只输出一次类别 head。它没有现成的 per-month logits，因此不能直接把 `P_ext[c, month_t]` 加到当前 head 的输出上。

若后续确实要做 Decision Fusion，需要额外设计：

```text
F_l,t -> shared spatial head -> Z_t [B,K,H,W]
      -> class-month temporal fusion using P_ext
      -> final logits
```

这已经是新的 decoder 变量，应该另立实验编号，不能和简单的先验旁路注入混在一起。

## 5. 公平实验矩阵

| 实验 | 特征输入 | 先验位置 | 研究问题 |
| --- | --- | --- | --- |
| K0 | 单层或多层、但配置固定 | 无先验 | 当前 decoder 基线 |
| K2-feature | `[B,L,T,D,H,W]` | 每层重组后旁路调制 | 物候先验能否改善时空特征 |
| K2-shared | `[B,L,T,D,H,W]` | 所有层共享 `q_t` | 是否需要层特定的先验适配 |
| K2-decision | `[B,L,T,D,H,W]` | per-month logits 后融合 | 先验在决策层是否更有效 |
| K3-data | 同 K2 | 使用 train-only `P_data` | 数据统计和人工知识谁更可靠 |

普通早期融合多层 DPT/UPerNet 若仍使用 `[B,L,D,H,W]`，不应与 `K2-feature` 声称是在同一输入信息量下比较物候效果。要比较“decoder 架构”，所有模型应读取同一份时间保留缓存；要比较“物候注入”，则只改变先验分支，保持 encoder、缓存、loss 和训练折不变。

## 6. 当前项目的结论

目前不需要给第 3、6、9、12 层分别查物候表。正确路线是：

1. `P_ext[c,m]` 只查一张表；
2. `months[:,t]` 把序列位置映射到自然月份；
3. 只有保留 `T` 的 decoder 才做逐月注入；
4. 中间层通过共享先验表示和可学习的层适配处理；
5. 已经平均掉时间的缓存不能逆向恢复逐月先验，必须重新生成 temporal cache。

当前代码中，最接近这条路线的是 `configs/galileo_3d_aware_dpt.yaml`。普通 `multi_layer_dpt` 仍应先作为早期融合的 decoder 对照，不能把它和时间保留模型的物候实验结果混为一谈。

## 7. 项目路线更新：统一使用时间保留缓存

根据后续协作决定，主实验不再让不同 decoder 读取不同的信息量。所有新的特征缓存统一保留时间维：

```text
temporal_v2:
features[l, t] -> [B, T, D, H_l, W_l]
months        -> [B, T]
```

原先的 `[B, D, H, W]` spatial cache 只作为历史 baseline 或缓存兼容性测试，不作为物候研究的主输入。

在统一缓存下，三条 decoder 路线可以这样定义：

| 路线 | 时间融合位置 | 空间/层级融合位置 |
| --- | --- | --- |
| temporal single-layer DPT | 最终层先做 temporal readout | 单层 DPT |
| temporal multi-layer DPT | 各层保留 `T`，再做共享或层特定 temporal readout | DPT 跨层融合 |
| temporal UPerNet | 各层保留 `T`，再做 temporal readout | UPerNet 的 PPM/FPN 融合 |
| 3D-Aware DPT | 多层时空 block 中联合融合 `L,T,H,W` | 时空跨尺度融合后输出 |

物候先验的公共注入点应位于各路线的 temporal fusion 之前。这是一个 decoder 侧的**旁路残差注入**，不是修改 Galileo encoder、删除月份或用先验筛选输入：

```text
temporal cache F[l,t] + months[:,t] + P_ext[:,month_t]
                         |
                   shared prior side branch
                         v
                 temporal fusion of each decoder
                         v
                 decoder-specific layer fusion
```

这样比较 decoder 时，所有模型看到相同的 Galileo 表征和相同的物候输入；改变的只是“时间如何融合”和“空间层级如何融合”。对于不同层，可以增加很轻量的 `layer_adapter_l`，但它只适配特征通道，不改变 `P_ext` 的月份语义。

严格的先验消融使用配置开关：

```yaml
phenology:
  enabled: false  # no-prior baseline
```

与：

```yaml
phenology:
  enabled: true   # prior side branch
  path: data/priors/pastis_ext_prior_draft.csv
  strength: 0.1
```

`enabled=false` 时不会构造 `PhenologyPriorAdapter`，因此 baseline 不增加先验分支参数。`strength=0` 可以用于检查旁路的数值回退，但因为模块仍然存在，它不是最严格的参数量对照。

因此当前最推荐的顺序是：

1. 先为所有需要比较的 decoder 生成同一套 `temporal_v2` 缓存；
2. 用 `alpha=0` 跑无先验的 temporal DPT/UPerNet 对照；
3. 在公共 temporal fusion 前接入共享 `MetaEncoder(month, P_ext[:, month])`；
4. 最后才比较 layer-specific adapter、FiLM/AdaGN 或 per-month decision fusion。
