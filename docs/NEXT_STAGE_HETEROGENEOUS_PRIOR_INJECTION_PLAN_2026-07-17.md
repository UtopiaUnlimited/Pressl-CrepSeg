# 当前唯一执行规划：内容感知的异构先验知识注入

状态：**CURRENT / 后半阶段唯一总规划**

首次整理：2026-07-17

最后更新：2026-07-18

任务：基于 frozen Galileo 多时相特征的 PASTIS 作物语义分割

> 本文件决定“接下来做什么”。若其他规划、讲稿或旧运行手册与本文冲突，以本文为准。文档状态见 [README.md](README.md)。

## 0. 当前优先级：先得到先验体系结果

剩余研究时间有限，执行优先级固定为：

1. **第一优先级：** 用已经跑通的 CA-HPI 和冻结先验 v1，取得一组可与 3D-Aware 无先验结果比较的正式结果；
2. **第二优先级：** 结果出现后，只补最能解释结果的 FiLM 或 class-shuffled 对照；
3. **第三优先级：** 时间仍充足时才扩展 stage、文本、更多 seed 或第二 decoder。

在第一组 M1 结果产生前，不继续扩充通用框架，不重做大规模文献型先验，不启动完整消融矩阵。当前 `pastis_ext_prior_v1.csv` 是首轮冻结输入，fold4/fold5 结果不得反向修改该表。

## 1. 已经冻结的方向

### 1.1 核心目标

后半阶段不再以继续设计 decoder、刷最高 mIoU 或单独修补几个困难类别为核心目标。需要解决的研究问题是：

> 如何把数值 metadata、离散属性和文本知识等异构先验，转化为统一表示，并根据视觉内容稳定、可解释、可复用地注入神经网络？

作物物候是验证这一通用方法的任务实例，而不是方法本身。项目必须同时体现：

- **共性：** 一套不写死 PASTIS、19 类、Galileo 或某个 decoder 的先验接口与融合模块；
- **个性：** 把作物的类别、月份、物候阶段、知识强度、置信度和未知状态组织成可学习知识。

### 1.2 固定视觉基线

主开发骨干固定为：

```text
frozen Galileo temporal_v2 features
  -> 3D-Aware DPT (native deep skip)
  -> segmentation logits
```

选择它是因为它是当前项目结果最好的结构。之后除非发现实现错误，不再把“换 decoder”当成研究进展。`Temporal Readout + Multi-layer DPT` 只在方法成熟后承担跨结构验证。

### 1.3 旧实验的准确定位

- 已完成的 P0/P1/P2 基于 **Single-layer Temporal Readout**，不是 3D-Aware DPT。
- P1/P2 使用的是旧 `Global Add` 旁路：把类别—月份向量编码后在 Galileo 特征边界全局广播相加。
- 这些结果只说明旧注入方式存在很弱的可研究信号，完整数据见 [DECODER_EXPERIMENTS.md](DECODER_EXPERIMENTS.md)。
- **下一步不是把旧 P1/P2 原样搬到 3D-Aware 上重跑。** 旧方案只作为历史基线和失败诊断，资源优先用于新的注入方法。

## 2. 方法定义：CA-HPI

新方法暂命名为：

> **Content-Aware Heterogeneous Prior Injection（CA-HPI，内容感知异构先验注入）**

它由“统一先验编码”和“内容感知门控融合”两部分组成：

```text
numeric / categorical / text prior
                ↓
       modality-specific adapters
                ↓
 PriorBatch: tokens + mask + confidence + type
                ↓
visual tokens query the complete prior token set
                ↓
 confidence-aware cross-attention + zero-init gate
                ↓
         residual visual features
```

创新重点不是“多加一个 metadata MLP”，而是让不同位置和月份的视觉特征主动选择相关知识，并显式处理知识类型、缺失项和可信度。

### 2.1 通用输入接口

通用模块只接收以下批数据，不直接读取 CSV、作物名称或数据集配置：

```text
prior_tokens:      [B, Np, Dp]
prior_mask:        [B, Np]       # True 表示有效 token
prior_confidence:  [B, Np]       # [0, 1]
prior_type_id:     [B, Np]       # numeric / categorical / text / ...
prior_entity_id:   [B, Np]       # 可选，如类别、患者、地区
prior_time:        [B, Np, Kt]   # 可选，如月份、日期、热时间
```

约束：

- `Np`、`Dp`、类别数和时间长度可变；
- 缺失知识使用 mask，不以伪造的确定数值填充；
- confidence 是模型可见的输入，也是 attention bias 的组成部分；
- 任务相关的数据读取与通用融合块分离。

### 2.2 异构先验编码

第一版支持三种适配器，并统一输出 prior token：

| 输入 | 编码方式 | 例子 |
| --- | --- | --- |
| 连续值 / 周期时间 | 归一化 + Fourier/正弦编码 + MLP | 月份、经纬度、物候强度、置信度 |
| 离散 metadata | embedding + MLP | 类别、阶段、地区、传感器类型 |
| 文本知识 | 冻结或离线 text embedding + projection | 作物阶段描述、不确定性说明 |

第一轮只实现结构化输入；文本编码器不参与端到端训练。这样可以先检验融合机制，再判断文本是否提供额外信息。

### 2.3 内容感知门控融合

视觉特征 `F ∈ R^(B×D×T×H×W)` 展平为 `V ∈ R^(B×Nv×D)`，其中 `Nv=T×H×W`。视觉 token 作为 query，完整先验库作为 key/value：

```text
Q = Wq LN(V)
K = Wk PriorTokens
U = Wv PriorTokens

A = softmax(QK^T / sqrt(d)
            + log(confidence + eps)
            + mask_bias)

Delta = Wo(AU)
gate  = sigmoid(MLP([LN(V), Delta]))
Vout  = V + tanh(alpha) * gate * Delta
```

其中 `alpha` 是零初始化的可学习标量。它保证：

1. 模块刚接入时严格退化为原 decoder 输入和无先验基线；
2. 模型可按位置和月份决定是否使用知识；
3. 低置信度与缺失 token 不会被当作强知识；
4. 可以记录 attention、gate 和 residual norm，检查先验是否真的被使用。

推理阶段始终向每个视觉 token 提供完整先验库，禁止用像素真实类别选择先验，因此不存在标签泄漏。

### 2.4 唯一注入位置：decoder 前公共特征边界

CA-HPI 只放在 Galileo temporal feature pyramid 与 decoder 之间：

```text
Galileo / temporal_v2
  -> F[l] [B,T,D,H,W], l = 1...L
  -> shared CA-HPI + layer embedding + zero-init alpha[l]
  -> F'[l] [B,T,D,H,W]
  -> any time-preserving decoder
```

四层共享同一套 prior encoder、Q/K/V 和 gate，只保留 layer embedding 与逐层残差强度。这样先验模块不读取、也不修改 3D-Aware、Temporal Readout、DPT 或 UPerNet 的内部结构。

这里保留的是 decoder 前位置，废弃的是旧 `Global Add` 机制：旧方法把同一个 768 维残差广播到全部空间位置；CA-HPI 则让每个时空视觉 token 查询完整先验库，再通过 confidence-aware gate 决定残差。

当前不实现 decoder 内部 8×8 注入，不做注入位置消融，也不在输入端、decoder 内部和 logits 端叠加多套模块。

### 2.5 方法的通用性边界

CA-HPI 不包含以下任务常量：

- PASTIS 的 19 类映射；
- 固定 12 个月；
- Galileo 的 768 维；
- 某个 decoder 的内部通道数和模块布局；
- 某张物候 CSV 的字段名。

新任务只需要实现自己的 prior adapter，并提供 `[B,Np,Dp]`、mask 和 confidence。视觉侧只需把 decoder 接收的时间特征金字塔交给公共前置模块。

### 2.6 拟验证的组合创新点

CA-HPI 不是把 cross-attention 换一个名字。相对现有简单方案，项目需要验证以下组合是否真正成立：

- 相对 Global Add：从“同月知识广播给全部像素”变为“每个视觉时空 token 选择知识”；
- 相对普通 FiLM：从单个全局条件向量扩展为可变长、多类型、带 mask/confidence 的知识集合；
- 相对普通 cross-attention：增加知识置信度偏置、unknown 处理和严格零初始化回退；
- 相对只面向 PASTIS 的模块：把通用接口与作物物候 task adapter 分开；
- 相对只报告有/无先验：用 random、class-shuffled 和 month-shifted 证明正确知识内容的作用。

这些是待实验支持的方法贡献，不在结果出来前宣称优于已有工作；正式新颖性还需结合 [相关论文调研](RELATED_WORK_HETEROGENEOUS_PRIOR_INJECTION_2026-07-17.md) 继续核对。

## 3. 物候任务实例如何定义

### 3.1 物候 token

每条知识不再只是 `P[class, month]` 中的一个无来源分数，而定义为：

```text
class / entity
calendar time
phenological stage
knowledge value
source confidence
scope / region
valid-or-unknown mask
optional text description
```

当前已实现的 R1 结构化 token 为：

```text
z(c,m) = E_entity(c)
       + Fourier(month_m)
       + MLP(value)
       + MLP(confidence)
       + E_type
```

`E_stage(s)`、scope 和文本 projection 属于 R2–R4，尚未混入 R1，避免第一轮同时改变融合机制和知识表示。

`E_entity` 在所有正确、随机、置乱和月份偏移实验中保持同一套参数；对照只改变知识内容或其对应关系，避免把额外类别 embedding 误判为物候收益。

### 3.2 不确定类别

对聚合、多年生或资料不可靠类别，不强行制作尖锐的单峰曲线：

- 聚合类别允许多个 token / 多个候选阶段；
- 多年生类别使用宽时间窗或低 confidence；
- 未知阶段设置 mask；
- 文本可保留“多物种、管理方式或地区不确定”等语义；
- gate 应允许视觉模型忽略无帮助的时间知识。

### 3.3 类别选择与标签泄漏

模型看见的是所有类别的物候 token，而不是真实像素类别。视觉 query 根据图像内容选择 token。per-class 结果只用于训练后的分析，不能反向决定测试像素使用哪条曲线。

## 4. 实验设计

### 4.1 第一结果只比较两件事

所有新实验固定 3D-Aware DPT、`temporal_v2`、数据划分、loss、优化器、seed 和 checkpoint 规则：

| 编号 | 方法 | 目的 |
| --- | --- | --- |
| B0 | 3D-Aware DPT，无先验 | 匹配协议的主基线 |
| M1 | 3D-Aware + CA-HPI structured prior | proposed method |

现有 B0 fold5 `mIoU=0.59945` 暂作阶段性参考；它缺少完整审计信息，因此优先从服务器找回 config/commit/checkpoint/best fold4 元数据，而不是立刻重跑。M1 使用 fold4 选择 `best_val_miou.pt` 后只评估一次 fold5，先形成第一组结果。B1 FiLM 与旧 Single-layer Global Add 均不阻塞该结果。

### 4.2 第一结果之后的科研补强

只有 M1 第一结果完成后，才根据结果决定补强项。优先级是：B1 FiLM，其次一个 class-shuffled 对照；其余控制仅在时间允许时运行。

| 对照 | 改变内容 | 排除的解释 |
| --- | --- | --- |
| C0 | 无模块 | 原视觉基线 |
| C1 | module on，训练与评估均固定 `alpha=0` | 数值回退与接线错误 |
| C2 | 正确 structured prior | 目标实验 |
| C3 | 相同形状 random tokens | 额外参数/条件分支 |
| C4 | class-shuffled prior | 正确类别—知识对应 |
| C5 | month-shifted prior | 正确时间关系 |
| C6 | uniform / unknown prior | 普通全局偏置 |
| C7 | 正确 prior，但 confidence 全部置 1 | confidence 是否有贡献 |

第一轮 seed 42 只用于筛除明显无效设计。不能根据前几个 epoch 判断成败，也不能看 fold5 test 选模块。候选固定后补 seed 43/44，最后一次性评估 test。

### 4.3 融合机制通过后再比较先验形式

| 编号 | 先验形式 | 回答的问题 |
| --- | --- | --- |
| R1 | class-month numeric curve | 最小结构化知识是否可用 |
| R2 | class-month-stage-value-confidence | 显式阶段与不确定性是否有用 |
| R3 | frozen text description embedding | 文本知识是否提供独立信息 |
| R4 | structured + text | 异构知识是否互补 |

不要在首轮同时更换融合机制和先验形式，否则无法判断增益来源。

### 4.4 可插拔性作为接口验收

代码接口从第一版就必须让同一个 CA-HPI overlay 接到 3D-Aware 和各 Temporal Readout 路线，且：

- `PriorBatch`、prior encoder、Q/K/V 和 gate 完全复用；
- decoder 内部不增加任何 CA-HPI 专用代码；
- Single 只处理其实际消费的 final layer，Multi/UPerNet/3D-Aware 处理各自消费的层集合；
- 正式方法筛选仍固定 3D-Aware，其他结构只做接口验证或最终泛化实验。

## 5. 如何判定研究是否成功

本项目不以“mIoU 一定大幅提升”为唯一成功标准，证据分三级：

### 强证据

- 正确先验稳定优于无先验、random、class-shuffled 和 month-shifted；
- 多 seed 方向一致；
- attention/gate 显示不同视觉内容选择不同知识；
- 同一接口支持至少两种先验形式，并能接入第二个视觉结构。

### 有限但成立的证据

- 总体 mIoU 接近，但预先定义的季节性类别、缺失月份或少样本场景稳定受益；
- 正确先验仍明显优于错误先验；
- 模型能对低 confidence、unknown 或多年生类别自动降低 gate。

### 负结果

- 正确先验与 random/shuffled/shifted 无稳定差异；或
- 增益完全来自参数量和条件分支；或
- gate 长期接近零，说明完整视觉时序已经覆盖这部分知识。

负结果也应保留，但到此停止无界增加物候表和模块组合，转而总结知识冗余、表示不足或任务适用边界。

## 6. 分阶段工作规划

| 阶段 | 工作 | 交付物 | 进入下一阶段的条件 |
| --- | --- | --- | --- |
| S0 方向冻结 | 统一术语、方法、插入点和文档状态 | 本规划 + 文档索引 | 团队不再混淆旧 P1/P2 与新方法 |
| S1 接口与单元验证 | `PriorBatch`、structured adapter、CA-HPI、shape/mask/no-op 测试 | 可插拔模块与测试 | `alpha=0` 与 B0 输出数值一致 |
| S2 第一结果 | B0、M1，固定 seed 42 / fold4 | M1 val/test 与诊断；B0 阶段性对照 | 得到可汇报的先验体系结果 |
| S3 最小补强 | 根据 M1 结果选择 B1 或 class-shuffled | 一个最关键对照 | 能解释增益或负结果来源 |
| S4 物候建模 | R1–R4；处理 stage/confidence/unknown | 物候 token 数据与消融 | 找到知识内容的有效边界 |
| S5 通用性验证 | 第二 decoder；可选 missing-month / low-data | 插拔成本、资源和鲁棒性报告 | 形成最终论文证据链 |

实现状态（2026-07-18）：S0、S1 已完成。decoder 前 CA-HPI、structured phenology adapter、confidence 映射、配置 overlay、训练诊断和单元测试均已实现；本地 `llm` 环境使用真实 `temporal_v2` train/val 缓存完成了 2-batch 训练—验证冒烟，四层 strength 在首次优化后均离开 0。首轮输入已冻结为 `pastis_ext_prior_v1.csv`；当前唯一阻塞结果的是代码同步和服务器正式 M1 训练，不是 FiLM、文本或更多模块。

当前代码入口：

- [models/prior_injection.py](../models/prior_injection.py)：`PriorBatch`、通用 adapter 边界、structured encoder、CA-HPI 和多层前置注入；
- [models/phenology.py](../models/phenology.py)：物候表与 confidence 的 task adapter；
- [ca_hpi_structured.yaml](../configs/prior_injection/ca_hpi_structured.yaml)：当前 structured overlay；
- [test_prior_injection.py](../tests/test_prior_injection.py)：回退、mask/confidence、跨 decoder 和互斥性测试。

训练诊断按层、按 train/val 写入 TensorBoard `prior/...`，并保存为 `prior_diagnostics_history.json/csv`。核心字段包括 effective strength、gate mean/std、归一化 attention entropy、attention top-1、attended confidence、候选 residual ratio 和实际 applied residual ratio。完整 attention/gate 张量不会跨 batch 保留或写入 checkpoint。

## 7. 下一次工程工作的明确顺序

本文档完成后，按以下顺序推进，不再从旧方案清单中随机挑实验：

1. [x] 写 `PriorBatch` 与 task adapter 的接口规格和 shape 测试；
2. [x] 实现 numeric/categorical structured adapter，先不接文本模型；
3. [x] 实现 CA-HPI，并验证 mask、confidence 和 `alpha=0` 回退；
4. [x] 在 decoder 前公共 temporal feature pyramid 边界接入，不修改 decoder 内部；
5. [x] 在本地真实缓存跑 2-batch 冒烟，确认训练、验证、反向传播和诊断落盘；
6. [x] 实现并记录 strength/gate/attention/residual/confidence 的训练日志；
7. [x] 冻结 `pastis_ext_prior_v1.csv`，首轮结果前不再改曲线；
8. [ ] 提交同步后在服务器补 2-batch 冒烟，确认服务器缓存和资源链路；
9. [ ] 直接运行 M1 seed42，训练最多 50 epoch，由 fold4 选择 `best_val_miou.pt`；
10. [ ] 固定 checkpoint 后评估一次 fold5，并写入实验台账；
11. [ ] 根据第一结果决定只补 B1 FiLM 或 class-shuffled，不预先铺开全部实验。

## 8. 两条协作工作线

### 共性线：通用方法

- `PriorBatch` 与 adapter registry；
- 已实现 Fourier/MLP 与 categorical embedding；text projection 待实现；
- CA-HPI、FiLM 基线、mask/confidence/no-op；
- attention/gate/residual 诊断；
- 第二 decoder 的轻量适配。

### 个性线：物候知识

- 类别—月份—阶段—置信度记录；
- 聚合类、多年生类和 unknown 规则；
- 正确、random、class-shuffled、month-shifted 数据；
- 来源审计与 text description；
- 季节性/多年生/聚合类别分组评估。

两条线通过统一 `PriorBatch` 对接，不在模型代码里直接读取物候表。

## 9. 实验纪律

- train=`fold1/2/3`，val=`fold4`，test=`fold5`；
- 所有先验内容和统计只使用外部资料或 train folds；
- checkpoint 固定按 `best_val_miou.pt` 选择；
- test 不参与模块、epoch、先验曲线或阈值选择；
- 每次记录 config、prior version、cache、commit、seed、最佳 epoch 和资源开销；
- 总体指标、19 类指标、分组指标与反事实差异同时报告；
- 不用“跑了几个 epoch 没提升”作为停止依据。

## 10. 最终研究叙事

项目最终要讲的是：

> 我们提出 CA-HPI，将数值、类别和文本等异构先验编码为带类型、置信度和缺失掩码的统一 token；视觉时空特征通过内容感知 cross-attention 主动选择相关知识，并以零初始化门控残差安全注入。我们在多时相作物分割中把物候知识具体化为类别—时间—阶段—置信度 token，并用随机、类别置乱和月份偏移对照区分知识内容、额外容量与普通条件效应。

核心交付物是“一个可复用的先验注入方法 + 一条严格的知识有效性证据链”，而不是新的 decoder，也不是单独为某几个类别刷分。
