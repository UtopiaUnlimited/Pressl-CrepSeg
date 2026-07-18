# 后半阶段核心规划：面向视觉任务的异构先验知识注入

日期：2026-07-17

任务背景：基于 frozen Galileo 多时相特征的 PASTIS 作物语义分割

规划依据：助教对后半阶段研究目标的进一步解耦与指导

## 1. 方向调整

现有基线、Temporal Readout、3D-Aware DPT、分类别指标和可视化已经构成完整的研究基础。后半阶段不再以继续堆叠 decoder 或单纯刷新 mIoU 为核心目标，而是集中回答：

> 如何把文本、结构化表格或 metadata 形式的异构先验，稳定、可解释并可复用地注入视觉神经网络？

作物物候是这个通用问题在当前任务上的具体实例。项目需要同时完成两个层面：

1. **共性：** 建立统一的异构先验编码和视觉特征融合范式，使模块不依赖某一个 decoder，也不局限于遥感任务。
2. **个性：** 将作物类别、月份、物候阶段、置信度和文本描述转化为适合作物分割的可学习表示，并处理物种差异、混合类别、物候漂移和弱判别特征。

之前提出的困难类别诊断、类别均衡和多原型分析仍然保留，但降为任务诊断与扩展支线，不再取代“异构先验注入”这条核心主线。

## 2. 核心研究问题与预期贡献

### 2.1 核心研究问题

后半阶段围绕四个问题展开：

1. 不同数据形式的先验怎样映射到统一表示空间？
2. 全局先验怎样根据图像内容，对不同时间、空间位置和类别产生不同作用？
3. 怎样证明性能变化来自正确知识，而不是额外参数、普通月份编码或随机条件分支？
4. 怎样让同一个先验模块以较小改动接入不同视觉网络和不同任务？

### 2.2 预期贡献

项目最终应形成三层贡献：

- **通用接口贡献：** 将 numeric metadata、categorical metadata 和 text description 转为统一的 prior tokens。
- **通用融合贡献：** 设计带内容相关性、置信度和残差门控的先验注入模块。
- **任务化贡献：** 将作物物候表示为类别—时间—阶段—置信度知识，并验证它在多时相作物分割中的作用边界。

最终叙事不要求“任何先验一定提升精度”，而要求能够回答：什么知识、通过什么机制、在什么条件下会被视觉模型有效利用。

## 3. 总体框架

建议将框架暂命名为 `Heterogeneous Prior Injection (HPI)`，计算流程为：

```text
Raw heterogeneous prior
  ├─ structured numeric metadata
  ├─ categorical metadata
  └─ text description
            ↓
        Prior Encoder
            ↓
Prior Tokens [B, Np, Dp] + mask + confidence + type
            ↓
   Projection to vision dimension
            ↓
Content-aware Gated Fusion Block
            ↓
Vision features / temporal features / logits
            ↓
          Task Head
```

通用框架只规定统一的 token 接口和融合方式。PASTIS 物候任务负责产生具体的 prior tokens，但不能把作物类别、月份表或 Galileo hidden size 写死在通用模块内部。

### 3.1 统一先验接口

建议每批先验统一表示为：

```text
prior_tokens:     [B, Np, Dp]
prior_mask:       [B, Np]
prior_confidence: [B, Np]
prior_type:       [B, Np]
prior_entity_id:  [B, Np]  # 可选，例如 crop class id
prior_time_id:    [B, Np]  # 可选，例如 calendar month
```

接口允许不同任务只替换 `Prior Encoder`，而复用后续融合模块。

### 3.2 三类基础编码器

第一版优先支持三类输入：

1. **连续数值与时间：** 使用归一化数值、正弦/Fourier 时间编码和 MLP。
2. **离散 metadata：** 使用 categorical embedding，例如类别、地区、传感器或阶段类型。
3. **文本描述：** 使用冻结文本编码器或离线生成的文本 embedding，再投影到统一维度。

助教提到的“正弦编码 + MLP”可作为结构化 metadata 的工程起点；其提及的 DiffusionSat 类工作可作为后续文献检索线索，但正式引用前需要核对准确论文名称和实现。

## 4. 通用融合模块设计

当前实现把某个月的完整类别向量经过 MLP 后广播到所有空间位置。它可以作为最简单的 `Global Add` 基线，但存在两个局限：

- 同一个月份残差对所有像素相同，不能根据图像内容选择相关知识；
- 自由 MLP 可能把正确、置乱或随机先验重新编码成普通条件向量，使“知识内容是否正确”不容易被严格识别。

因此建议按由简到难的顺序比较四种融合方式。

### 4.1 F0：Global Add

```text
F' = F + alpha * Project(GlobalPrior)
```

作用：保留现有实现，作为最低复杂度融合基线。

### 4.2 F1：FiLM / AdaLN

```text
gamma, beta = MLP(PriorTokens)
F' = gamma * Norm(F) + beta
```

作用：检验先验作为全局条件调制视觉特征是否优于直接相加。实现简单，适合作为第一个新模块。

### 4.3 F2：Gated Cross-Attention

```text
Q = Wq * VisionTokens
K, V = Wk/Wv * PriorTokens
A = softmax(QK^T / sqrt(d) + confidence_bias + mask)
DeltaF = Wo * (A V)
F' = F + sigmoid(g(F, DeltaF)) * DeltaF
```

作用：让每个图像位置根据自身内容选择相关先验 token，是建议的核心方案。

设计要求：

- gate 初始强度较小或采用零初始化残差，保证接入时不会破坏已有视觉分支；
- attention 支持 prior mask 和 confidence bias；
- 模块同时支持二维特征 `[B,D,H,W]` 和时间特征 `[B,T,D,H,W]`；
- 不读取像素真实类别，避免标签泄漏；
- 输出维度与输入视觉特征一致，使其可插拔到不同 decoder 前。

### 4.4 F3：Decision Fusion

```text
logits = image_logits + lambda * prior_conditioned_logits
```

作用：把先验影响限制在最终决策层，作为 feature fusion 的对照。任务相关部分与通用 HPI block 分开实现。

## 5. 作物物候的任务化表示

### 5.1 不再只使用一张类别—月份分数表

物候知识建议拆成以下字段：

```text
crop class
calendar month
phenological stage
relative discriminative value
source confidence
regional applicability
text description
valid / unknown mask
```

每个 token 可以写为：

```text
z(c,m,s) = E_class(c)
         + E_time(m)
         + E_stage(s)
         + MLP(value, confidence)
         + Project(text_embedding)
```

未知字段通过 mask 表达，不能用强行填充的确定值伪装成可靠知识。

### 5.2 处理类别内部异质性

对于 `Fruits, vegetables, flowers`、`Leguminous fodder`、`Mixed cereal` 等聚合类别，不使用单一确定曲线。可选方案为：

- 多个物候 token 或多个子原型；
- 较低 confidence；
- 宽时间窗口；
- unknown mask；
- 文本中显式保留“多物种、管理方式不确定”等描述。

对于 `Meadow`、`Grapevine`、`Orchard` 等长期覆盖或多年生类别，应允许模型降低时间先验权重，避免用不适合的季节曲线强行调制视觉特征。

### 5.3 类别身份必须与输出语义对齐

推理阶段不知道像素真实类别，因此不能按标签选择先验。建议让图像 token 同时关注完整的类别先验库，由内容相关 attention 选择相关 token。

同时需要避免一个完全自由的 MLP 把 class-shuffled prior 的行置换重新吸收。可采用以下约束之一：

- `prior_entity_id` 与分割分类器的类别 prototype 共享或显式对齐；
- 在 decision branch 中固定“第 c 类证据只与第 c 类先验融合”的对角对应；
- 冻结部分先验编码器，并只学习通用投影；
- 将正确与置乱关系直接作用于 attention mask 或 logit bias，而不是只作为 MLP 输入。

该约束是 P1/P2 消融具有解释性的前提。

## 6. 最小实验矩阵

### 6.1 固定开发骨干

后续模块开发只固定一个稳定且训练成本可控的时间模型，建议优先使用：

```text
Temporal Readout + Multi-layer DPT
```

原因是其时间模块与空间 decoder 边界清晰，适合插入统一先验模块。完成模块筛选后，只把最佳方案迁移到一个结构不同的模型，例如 `3D-Aware DPT` 或 `Temporal Readout + UPerNet`，用于验证可插拔性。

### 6.2 融合机制比较

| 编号 | 视觉骨干 | 先验 | 融合方式 | 研究目的 |
| --- | --- | --- | --- | --- |
| H0 | 固定 | 无 | 无 | 严格基线 |
| H1 | 固定 | 正确 numeric prior | Global Add | 复现当前方案 |
| H2 | 固定 | 正确 numeric prior | FiLM/AdaLN | 检验条件调制 |
| H3 | 固定 | 正确 numeric prior | Gated Cross-Attention | 核心候选 |
| H4 | 固定 | 正确 numeric prior | Decision Fusion | 决策层对照 |

先用 seed 42 在 fold4 validation 初筛。只选择一个最佳融合方案进入后续先验表示和多 seed 实验。

### 6.3 先验表示比较

固定最佳融合模块后，再比较：

| 编号 | 先验表示 | 目的 |
| --- | --- | --- |
| R1 | numeric class-month curve | 结构化数值基线 |
| R2 | class-month-stage-confidence tokens | 加入物候结构和不确定性 |
| R3 | text description embedding | 检验文本知识条件 |
| R4 | structured + text | 检验混合异构知识 |

第一阶段不同时训练大型语言模型。文本编码器优先冻结或离线预计算，避免把增益混同为额外模型容量。

## 7. 必须保留的反事实对照

只比较“有先验/无先验”不足以支持知识注入有效。每个候选融合方案至少需要：

| 对照 | 设置 | 排除的解释 |
| --- | --- | --- |
| C0 | 无先验、无模块 | 视觉基线 |
| C1 | 正确先验 | 目标实验 |
| C2 | 相同参数量 + random tokens | 额外容量 |
| C3 | class-shuffled prior | 正确类别对应关系 |
| C4 | month-shifted prior | 正确时间对应关系 |
| C5 | uniform / unknown prior | 普通条件偏置 |
| C6 | strength=0 或 gate=0 | 数值回退一致性 |

成功标准不仅是 `C1 > C0`，而是正确先验需要同时优于随机、类别置乱和月份偏移先验。否则只能说明“增加条件分支可能有用”，不能说明知识内容被有效利用。

## 8. 评价体系

### 8.1 任务性能

- validation/test mIoU、macro F1、loss；
- 19 类 per-class IoU/F1；
- 季节性类别、长期覆盖类别和聚合类别的分组指标；
- 混淆矩阵和固定样本错误图。

### 8.2 先验是否真正被使用

- 正确、置乱、随机、月份偏移先验之间的差异；
- gate 均值、方差和不同层的强度；
- image-to-prior attention 的类别、月份和空间分布；
- prior residual 相对 vision feature 的范数比例；
- 把 prior 替换为 uniform 后输出 logits 的变化。

### 8.3 通用性

- 同一 HPI block 是否能接入至少两个不同 decoder；
- numeric metadata 和 text embedding 是否共用相同 token 接口；
- 参数量、显存和训练时间增量；
- 缺失字段、低置信度或部分月份缺失时是否仍能运行。

### 8.4 鲁棒性场景

在完整数据实验后，可加入两个更需要先验的受控场景：

1. 随机缺失部分月份；
2. 只使用部分训练样本。

如果先验只在信息不足时有效，也属于明确且合理的结论。

## 9. 两条分工线

### A. 共性：通用异构先验注入

负责内容：

- 图像 + metadata/text 融合文献矩阵；
- `PriorBatch` 和 `PriorEncoder` 统一接口；
- FiLM、Gated Cross-Attention、Decision Fusion；
- mask、confidence、gate 和通用诊断；
- 跨 decoder 接入和资源开销评估。

主要交付：一个与 PASTIS 类别数和 Galileo hidden size 解耦的可复用模块。

### B. 个性：作物物候知识建模

负责内容：

- 作物、月份、阶段、来源和置信度的定义；
- 聚合类别、多年生类别和 unknown 状态处理；
- numeric、stage token 和 text description 三种表示；
- 正确、类别置乱、月份偏移和 uniform 先验；
- 物候相关分类别分析和可视化。

主要交付：一套有来源、有置信度、有不确定性处理的物候 prior dataset/encoder。

### 共同负责

- 固定实验协议；
- P0/P1/P2 与新反事实消融；
- 多 seed 验证；
- 论文/汇报中的结论边界。

## 10. 实施阶段与停止标准

### 阶段 0：已有结果收口

- 完成四种 Temporal Readout test；
- 保留已有 decoder 和 3D-Aware DPT 作为骨干结果；
- 整理当前 Global Add 物候实验；
- 不再新增普通 decoder。

### 阶段 1：统一接口与数值回退

- 实现 `PriorBatch`；
- 把现有 numeric prior 接入统一接口；
- 验证 gate=0 时与 H0 输出一致；
- 输出 prior residual/gate 诊断。

若无法做到严格回退或不同 decoder 需要大量专用代码，应先修正接口，不进入大规模训练。

### 阶段 2：融合模块初筛

- 固定一个开发骨干；
- 比较 H0–H4；
- seed 42、fold4 validation 初筛；
- 不提前根据 fold5 test 选择模块。

建议内部准入标准：总体 val mIoU 提升至少 `0.5` 个百分点，或预先指定类别组平均 IoU 提升至少 `2.0` 个百分点且总体下降不超过 `0.3` 个百分点。

### 阶段 3：知识正确性验证

- 对最佳融合方案运行 C0–C6；
- 只有正确先验稳定优于 random、class-shuffled 和 month-shifted，才声称知识内容有效；
- 否则分析为容量效应、普通条件效应或现有知识表示不足。

### 阶段 4：物候表示和通用性

- 比较 R1–R4；
- 最佳方案补 seed `43/44`；
- 接入第二个 decoder；
- 可选运行 missing-month 或 low-data 场景。

若正确先验仍无稳定增益，停止追求精度提升，转而总结注入机制、失败原因和适用条件，不再无界扩展先验表或模块组合。

## 11. 文献检索框架

文献不局限于遥感，按问题而不是领域检索：

- remote sensing image + metadata conditioning；
- medical image + clinical/tabular metadata fusion；
- vision-language conditioning；
- diffusion model metadata conditioning；
- FiLM、AdaLN、cross-attention、prompt tokens；
- multimodal missing data and uncertainty-aware fusion；
- knowledge-guided neural networks；
- temporal prototype learning and class-conditioned attention。

每篇文献记录：先验形式、编码器、融合位置、融合算子、是否有反事实对照、是否处理缺失/不确定性、是否可跨 backbone。

## 12. 近期执行清单

### 研究定义

- [ ] 将项目后半阶段标题统一为“异构先验知识注入”，物候作为任务实例。
- [ ] 固定一个开发骨干和一个跨结构验证骨干。
- [ ] 明确 numeric、stage/confidence、text 三种先验输入格式。
- [ ] 核对助教提到的 DiffusionSat 类论文及其 metadata encoder 实现。

### 工程实现

- [ ] 建立 `PriorBatch` 数据结构与 mask/confidence 字段。
- [ ] 将现有 Global Add 改接统一接口，保持旧 checkpoint/配置可复现。
- [ ] 实现 FiLM/AdaLN 基线。
- [ ] 实现支持 2D/temporal feature 的 Gated Cross-Attention。
- [ ] 增加 gate、attention、residual norm 的日志。
- [ ] 增加 random、class-shuffled、month-shifted、uniform 对照配置。

### 实验与证据

- [ ] 完成 H0–H4 的 fold4 初筛。
- [ ] 对最佳方案完成 C0–C6。
- [ ] 输出总体、分类别、分组和混淆指标。
- [ ] 通过后补 R1–R4、seed 43/44 和第二 decoder。

## 13. 最终研究叙事

建议将最终工作概括为：

> 本项目不把提升作物分割精度作为唯一目标，而是研究异构先验如何被编码为统一 token，并通过内容相关、置信度感知和门控残差方式注入视觉网络。作物物候用于验证结构化数值、类别时间关系和文本描述能否在多时相遥感分割中提供独立信息；正确、随机、类别置乱和时间偏移对照用于区分知识内容、模型容量与普通条件效应。

该叙事允许出现三种都有研究价值的结果：

1. 正确先验稳定提升，证明知识内容和注入机制有效；
2. 只在缺失月份或少样本条件下提升，说明先验主要改善信息不足场景；
3. 正确先验不优于反事实先验，说明当前知识表达或任务条件存在冗余，并明确后续方法的适用边界。

核心成果应是“可复用的先验注入方法 + 严格的知识有效性验证”，而不是再增加一个只对当前配置有效的 decoder。
