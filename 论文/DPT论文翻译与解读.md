# Vision Transformers for Dense Prediction：DPT 论文翻译与解读

> 论文：Vision Transformers for Dense Prediction  
> 作者：René Ranftl、Alexey Bochkovskiy、Vladlen Koltun  
> arXiv：<https://arxiv.org/abs/2103.13413>  
> 官方代码：<https://github.com/isl-org/DPT>

这篇论文提出了 DPT（Dense Prediction Transformer），研究如何把 Vision Transformer 用于深度估计和语义分割等密集预测任务。

对我们当前项目最重要的不是 DPT 在自然图像上的具体指标，而是它提供了一种完整的思想：

> Transformer 的 token 具有全局上下文，但最终仍然需要被重新组织为空间特征图，并通过多层级融合恢复像素级预测。

这正对应 Galileo + PASTIS 项目中的核心问题：Galileo 输出的是 Transformer token，decoder 必须把这些 token 有效转换成作物分割图。

---

## 1. 什么是 dense prediction

分类任务通常输出一张图像对应的一个标签：

```text
整张图 -> 一个类别
```

密集预测则要求对空间中的每个位置进行预测：

```text
图像 -> H × W 个像素预测
```

典型任务包括：

- 语义分割：每个像素预测类别；
- 深度估计：每个像素预测深度值；
- 表面法线估计：每个像素预测方向；
- 光流和其他像素级回归任务。

密集预测同时需要两种能力：

1. 局部精度：保留边界、纹理和细小结构；
2. 全局一致性：理解整幅图像中的长距离关系和整体语义。

只依赖局部卷积容易缺少全局上下文，只依赖高层语义又可能丢失像素级细节。因此 encoder 和 decoder 的连接方式非常重要。

---

## 2. 论文为什么认为 Transformer 适合密集预测

### 2.1 卷积网络的限制

传统卷积网络通常通过连续下采样扩大感受野：

```text
高分辨率局部特征
    -> 下采样
低分辨率大范围特征
    -> 下采样
更低分辨率语义特征
```

这样做有效，但会带来两个问题：

- 深层特征的空间分辨率降低；
- 细粒度空间信息在深层逐渐丢失，decoder 只能尝试恢复，而不能真正找回已经消失的信息。

卷积的另一个特点是单次操作的感受野有限。想让一个像素看到远处区域，通常需要堆叠很多层卷积或使用下采样、空洞卷积等结构。

### 2.2 Transformer 的优势

ViT 把图像切成 patch，每个 patch 变成一个 token。Transformer 的自注意力允许每个 token 与其他 token 交互：

```text
一个空间 patch 的表示
    <-> 其他所有 patch 的表示
```

因此，在 Transformer 的每个阶段，token 都能获得全局上下文。

同时，标准 ViT 在初始 patch embedding 后通常不再逐阶段下采样，token 数量保持不变。这意味着：

- 不同 Transformer 层的 token 数量相同；
- 每个 token 仍然对应初始图像中的一个 patch；
- 早期层和深层都可以重新排列为空间网格；
- 深层表示具有全局感受野，同时保留相对细的空间粒度。

这为密集预测提供了一个有利条件：深层特征不必像普通 CNN 一样被压缩到很小的空间分辨率。

---

## 3. DPT 的整体结构

DPT 保留 encoder-decoder 框架，但把 CNN backbone 换成 Transformer：

```text
输入图像
    ↓ patch embedding
Transformer encoder
    ↓ 从多个深度读取 token
Reassemble：token -> 空间特征图
    ↓ 投影、重采样、逐级融合
卷积式 fusion decoder
    ↓ task-specific head
密集预测结果
```

DPT 的关键流程可以概括成：

```text
Transformer tokens
  -> Read
  -> Concatenate
  -> Resample
  -> Fusion
  -> Prediction head
```

其中 `Read`、`Concatenate` 和 `Resample` 共同构成 Reassemble 操作。

---

## 4. Transformer token 如何变成空间特征图

### 4.1 ViT 的 token 结构

对于大小为 `H×W` 的图像，如果 patch size 为 `p`，空间 token 数量为：

```text
N_p = (H / p) × (W / p)
```

论文中的 ViT 还包含一个不对应具体空间 patch 的 readout token，类似分类任务中的全局 token：

```text
t = [t_0, t_1, ..., t_Np]
```

其中：

- `t_0`：readout token；
- `t_1 ... t_Np`：空间 patch tokens。

### 4.2 Read：如何处理 readout token

readout token 不是一个明确的空间位置，但它可能包含全局信息。论文比较了三种处理方式：

#### Read-ignore

直接丢弃 readout token：

```text
{t_1, ..., t_Np}
```

#### Read-add

将全局 readout token 加到每一个空间 token 上：

```text
t_i' = t_i + t_0
```

#### Read-proj

将每个空间 token 与 readout token 拼接，再通过 MLP 投影回原来的维度：

```text
t_i' = MLP(concat(t_i, t_0))
```

论文默认采用 `Read-proj`，因为它既保留空间 token，也让每个位置能够获得全局信息。

### 4.3 Concatenate：恢复空间排列

移除或处理 readout token 后，空间 token 根据原始 patch 位置重新排列：

```text
[t_1, t_2, ..., t_Np]
    -> [H/p, W/p, D]
```

这一步不是学习操作，而是根据 patch 的空间顺序进行 reshape/reassemble。

它成立的前提是：

1. token 数量确实对应空间 patch 数量；
2. token 排列顺序与空间 patch 顺序一致；
3. 没有把时间、通道组或特殊 token 错当成空间 token。

这正是我们此前检查 Galileo token 的原因。普通 ViT 的 token 可以直接按二维空间网格重排，但 Galileo 的 space-time token 还包含时间和通道组维度，必须先进行结构化聚合。

### 4.4 Resample：构造不同分辨率

重新排列后，DPT通过 `1×1` 卷积调整通道数，再通过卷积或转置卷积进行重采样，得到不同空间分辨率的特征图：

```text
token grid -> 投影 -> 上采样或下采样 -> image-like feature map
```

DPT 将不同 Transformer 层的表示分配到不同输出分辨率：

- 较浅层特征通常重采样到较高空间分辨率；
- 较深层特征通常重采样到较低空间分辨率；
- 后续 fusion 模块再逐级融合这些特征。

需要注意：DPT 的多分辨率主要是 decoder 人为构造出来的，Transformer 各层原始 token 网格本身通常保持相同大小。

---

## 5. DPT 的多层特征融合

### 5.1 为什么要使用多个 Transformer 层

不同深度的 Transformer 表示通常承担不同功能：

- 浅层：更接近输入 patch，保留局部纹理、边缘和细粒度空间差异；
- 中间层：融合局部与上下文信息；
- 深层：具有更强的全局语义和类别判别能力。

如果只使用最后一层，模型可能具有很强的语义理解，但边界和细节恢复能力不足。如果只使用浅层，又可能缺少全局上下文。

因此，DPT 的基本思想是：

```text
浅层局部信息 + 中层上下文 + 深层语义
                  ↓
             融合后进行密集预测
```

### 5.2 论文使用的层级

对于 ViT-Base，论文从第 3、6、9、12 层读取特征；对于 ViT-Large，则使用第 5、12、18、24 层。

这和我们组员提出的 Galileo 多层方案高度一致：

```text
Galileo layers: [3, 6, 9, 12]
```

但需要保持严谨：Galileo 的 Transformer 结构、token 语义和预训练方式与 ViT 不完全相同，因此这是一个受 DPT 启发的实验设计，而不是原论文方案的直接复现。

### 5.3 Fusion 模块

DPT 使用类似 RefineNet 的多阶段特征融合模块。每个阶段通常包含：

1. 对当前特征进行投影；
2. 与来自另一层级的特征融合；
3. 使用残差卷积单元增强特征；
4. 上采样到更高空间分辨率。

整体过程可以表示为：

```text
最深层低分辨率特征
       ↓ Fusion + 上采样
加入下一层特征
       ↓ Fusion + 上采样
加入更浅层特征
       ↓ Fusion + 上采样
高分辨率预测特征
```

它和简单的 `concat -> convolution -> upsample` 相比，更强调分阶段、逐级的空间恢复。

---

## 6. 论文实验与结论

DPT 在两类密集预测任务上进行实验：单目深度估计和语义分割。

### 6.1 深度估计

论文报告 DPT 相比当时的强卷积基线取得明显提升，最大相对性能提升超过 28%。在 NYUv2、KITTI 等较小数据集上，DPT 也能够通过微调取得较好结果。

论文认为，DPT 的优势主要体现在：

- 预测更加细粒度；
- 全局结构更连贯；
- 对输入分辨率变化更不敏感；
- Transformer 的全局感受野能够改善大范围结构判断。

### 6.2 语义分割

DPT 在 ADE20K 上取得 49.02% mIoU，并在 Pascal Context 等数据集上也取得当时的领先结果。

这说明 DPT 的价值不是只适用于深度回归，而是适用于一般的像素级预测。

### 6.3 预训练的重要性

论文也指出，Transformer 要发挥能力，通常需要足够大的预训练数据。更好的预训练方法和更多预训练数据，都会影响 DPT 的最终效果。

这个结论与我们的研究非常相关：我们不是从零训练一个普通 Transformer，而是使用已经在遥感数据上预训练的 Galileo，再训练下游 decoder。合理的研究问题是：

> 当 encoder 已经包含大量遥感知识时，DPT-style decoder 是否能够比普通 decoder 更充分地读取这些知识？

---

## 7. 这篇论文和我们三条路线的关系

这里需要严格区分论文原始 DPT 和当前项目中的三个实验名称。

### 7.1 外部 baseline：原 Galileo 普通 decoder

按照我们已经确定的研究叙事，论文或原实现中的普通 decoder 结果是外部 baseline：

```text
Galileo encoder + 原始普通 decoder
```

它回答的是：原始 Galileo 下游解码方案能达到什么水平？

### 7.2 路线 1：单层 DPT-style decoder

当前项目的单层 DPT 只读取 Galileo 最终层特征：

```text
Galileo final layer
    -> token 结构化聚合
    -> [D, 16, 16] spatial feature map
    -> projection + convolution + upsampling
    -> segmentation logits
```

它体现了 DPT 的“token 重新组织为空间图 + 卷积解码”思想，但不完全等于原论文的完整 DPT，因为原论文默认从多个层级读取特征。

因此更准确的称呼是：

```text
single-layer DPT-style decoder
```

而不是“完整 DPT 复现”。

### 7.3 路线 2：多层 DPT

多层 DPT 读取 Galileo 的第 3、6、9、12 层：

```text
Galileo layer 3  -> reassemble/project
Galileo layer 6  -> reassemble/project
Galileo layer 9  -> reassemble/project
Galileo layer 12 -> reassemble/project
                         ↓
                  progressive fusion
                         ↓
                  segmentation map
```

这条路线最接近 DPT 论文的核心思想。它对应的研究假设是：

> 作物分割需要同时使用局部光谱/边界信息和深层作物语义，单一最终层可能不足以保留所有密集预测所需的信息。

### 7.4 路线 3：UPerNet-style decoder

UPerNet-style decoder 也可以读取多个 Galileo 层，但采用金字塔池化和 FPN 风格融合。

它和 DPT 的主要区别在于：

- DPT 更强调 Transformer 层级特征的 Reassemble 和逐级 RefineNet 融合；
- UPerNet 更强调多尺度上下文、金字塔池化和 FPN 式自顶向下融合。

因为 Galileo 各层的原始空间 token 网格可能相同，所以 UPerNet 在当前项目中应被理解为“跨层语义金字塔 decoder”，而不是完全照搬 CNN 中不同空间分辨率的 FPN。

---

## 8. DPT 论文对 Galileo token 问题的启发

### 8.1 DPT 的 reshape 不是随意操作

DPT 论文中 token 到空间特征图的重组依赖一个前提：每个 token 与一个图像 patch 一一对应。

普通 ViT：

```text
[B, 1 + H_grid × W_grid, D]
    -> 去掉 readout token
    -> [B, H_grid × W_grid, D]
    -> [B, D, H_grid, W_grid]
```

Galileo：

```text
[B, H_grid × W_grid × T × channel_group, D]
    -> 按结构解析 space-time token
    -> 对 T 和 channel_group 聚合
    -> [B, H_grid × W_grid, D]
    -> [B, D, H_grid, W_grid]
```

所以我们之前发现的 token 顺序问题非常关键：如果在 Galileo 的完整序列上直接按照 `N / grid_tokens` reshape 并平均，可能会混合错误的空间位置。

当前项目采用 `spacetime_mean` 的结构化聚合，解决的是“空间网格是否被正确恢复”的工程问题；但它仍然是固定平均，不能等同于 DPT 或 U-TAE 的学习式多尺度时间建模。

### 8.2 Galileo 没有必要照搬 readout token 处理

DPT 原论文专门讨论了 ViT 的 readout token，因为 ViT 通常有一个全局分类 token。

Galileo 的 token 结构不同，并不应直接把“最后一个 token”或“第一个 token”当成 DPT 的 readout token。必须根据 Galileo 的官方输出结构确认：哪些 token 属于 space-time、space、time、static，以及哪些 token 是有效空间特征。

这说明 DPT 提供的是 decoder 设计原则，而不是可以直接套在任意 Transformer 输出上的 reshape 公式。

---

## 9. 这篇论文支持哪些科研假设

### 假设一：更强 decoder 可以释放预训练表示

如果 Galileo 的 encoder 已经包含足够的遥感信息，那么原始普通 decoder 可能没有充分利用这些信息。DPT-style decoder 通过投影、空间重组和逐级融合，有可能提升 dense prediction 性能。

### 假设二：多层特征比单一最终层更适合分割

深层特征语义强，但局部边界可能变得不够细；浅层特征局部信息强，但类别判别能力可能不足。多层融合可以测试这两类信息能否互补。

### 假设三：输入和 token 重组必须先正确

DPT decoder 不能恢复输入端已经错误混合或丢失的空间信息。因此，任何 decoder 对比都必须固定并验证：

- token 的空间顺序；
- 时间和通道组的聚合方式；
- 特征图空间尺寸；
- 输入归一化和 patch size；
- void 标签和类别数。

否则，decoder 结果无法被公平解释。

---

## 10. 阅读这篇论文后的项目结论

目前可以把我们的研究路线写成：

```text
原 Galileo decoder
        ↓ 外部 baseline
single-layer DPT-style decoder
        ↓ 只改 decoder，检验强解码器是否有帮助
multi-layer DPT
        ↓ 检验跨 Transformer 层融合
UPerNet-style decoder
        ↓ 比较另一种多层融合机制
```

其中真正接近 DPT 原论文核心思想的是多层 DPT，而单层 DPT 是一个必要的控制实验：它能够告诉我们，性能提升究竟来自“decoder 结构变强”，还是必须依赖“多层特征融合”。

如果单层 DPT 就明显超过原始 decoder，说明原 decoder 的读取能力可能是主要瓶颈；如果只有多层 DPT 或 UPerNet 超过 baseline，则说明层级特征融合对作物分割很重要；如果三者都没有提升，则需要回头检查 Galileo 输入、token 聚合和预训练表示与 PASTIS 任务之间的匹配关系。

---

## 11. 一句话总结

DPT 的核心不是“使用 Transformer”，而是：**从 Transformer 的多个深度读取 token，将它们严格重组为空间特征图，再通过逐级融合恢复细粒度、全局一致的密集预测。**

对我们的项目来说，它既解释了为什么 DPT 可能优于 Galileo 原始普通 decoder，也明确提醒我们：多层 DPT 才是 DPT 思想的完整体现，而当前单层 DPT 应被视为一个重要的控制路线。

---

## 术语表

| 英文 | 中文 | 含义 |
|---|---|---|
| dense prediction | 密集预测 | 对图像中每个空间位置进行预测 |
| token | 标记/令牌 | Transformer 处理的向量单元 |
| patch embedding | 图像块嵌入 | 将图像 patch 映射为 token |
| readout token | 读出 token | 不对应具体空间 patch 的全局 token |
| reassemble | 重新组装 | 将 token 恢复成空间特征图 |
| resample | 重采样 | 调整特征图的空间分辨率和通道数 |
| fusion | 融合 | 合并不同层级或不同分辨率的特征 |
| receptive field | 感受野 | 一个特征能够看到的输入空间范围 |
| fine-grained | 细粒度 | 对边界、小目标和局部细节的精确表达 |
| globally coherent | 全局一致 | 预测结果在整体结构上合理连贯 |
| semantic segmentation | 语义分割 | 对每个像素预测类别 |

