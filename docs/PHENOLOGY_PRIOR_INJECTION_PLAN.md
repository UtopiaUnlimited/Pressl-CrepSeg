# PASTIS 物候先验构建与旁路注入计划

本文档根据 2026-07-14 助教与组内讨论整理，用于统一“人工物候先验、月份编码、Meta Data Encoder、Decision Fusion、月份筛选”等概念，并给出可以直接分工的后续任务。

## 一、研究目标

当前模型能够端到端学习多时相特征，但“输入了时间序列”不等于“已经显式利用了作物物候”。本阶段要回答的核心问题是：

> 在 frozen Galileo 提供的逐月上下文化特征上，人工整理的法国作物物候知识能否作为可解释的软先验，帮助模型区分光谱相似但生长周期不同的作物？

研究对象不是单纯增加一个 decoder，而是建立以下证据链：

```text
PASTIS 类别与法国作物周期
  -> 可追溯的类别-月份先验
  -> 不使用测试真值的旁路注入
  -> 与无先验模型做受控比较
  -> 用分类别结果证明哪些作物受益
```

## 二、先统一几个概念

### 2.1 月份编码

月份编码只告诉模型某个特征对应哪个自然月，例如 1 月、2 月或 10 月。它可以是月份编号、周期性正余弦编码或可学习 embedding。

月份编码本身不包含“冬小麦在什么月份更有辨识度”这类作物知识。当前 3D-Aware DPT 已包含 month embedding，因此后续不能把“增加月份编码”单独宣称为人工物候先验创新。

### 2.2 人工物候先验

人工物候先验来自可引用的农学资料，描述不同作物在法国相应地区的播种、生长、成熟和收获时间。建议表示为软矩阵：

```text
P_ext[class, calendar_month] in [0, 1]
```

矩阵每一行是一种作物的年度物候曲线。允许一行出现多个峰值，以表示多次种植、多个生长期或较宽的生长期窗口。不能简单用一个“开始月到结束月”覆盖所有类别。

### 2.3 数据驱动物候先验

数据驱动先验只由 train folds `1/2/3` 统计，例如各类别逐月 NDVI、光谱均值、标准差和有效观测数量：

```text
P_data[class, calendar_month]
```

它与人工先验是两种不同证据：

- `P_ext`：外部农学知识，体现人类经验。
- `P_data`：PASTIS 训练数据中的实际传感器表现。

二者必须先分别实验，再讨论是否融合。val/test 标签不得用于生成或修正先验。

### 2.4 旁路注入

旁路注入表示 Galileo 主分支保持不变，月份和物候知识由独立轻量分支进入时间融合或决策层：

```text
S2 monthly-12
  -> frozen Galileo
  -> F [B, L, T, D, H', W'] ---------> temporal/spatial decoder
                                               ^
month ids + P_ext/P_data -> prior encoder -----|
                                               |
                                      segmentation logits
```

旁路可以生成时间门控、FiLM/AdaGN 参数或类别相关的决策偏置。它不等于必须删除 Galileo 特征，也不改变 Galileo 原始波段输入。

### 2.5 月份筛选

月份筛选是另一条更强的路线：根据物候窗口屏蔽或删除部分月份。它不是旁路注入的同义词。

测试时真实作物类别未知，因此禁止使用真实标签选择月份。否则会产生标签泄漏。变长输入优先通过固定 `T=12` 加 temporal mask 处理，而不是为每个样本创建不同形状的张量。

## 三、第一阶段：构建可追溯的物候先验

### 3.1 类别范围

PASTIS 有效标签为 `0..18`。具体映射见 [`docs/next.md`](next.md#pastis-的19个有效类别)。处理原则如下：

- ID 0 Background：使用中性先验，不查询作物周期。
- 明确作物类别：查询法国国家级和尽可能细的地区级资料。
- Meadow、Fruits/Vegetables/Flowers、Leguminous Fodder、Orchard、Mixed Cereal 等集合类别：不能伪造单一精确周期，应使用宽窗口、混合曲线或低可信度标记。
- 无可靠来源的月份保持中性，不能凭印象补数值。

### 3.2 人工资料表

先建立可审计的原始资料表，而不是直接在代码中写 12 个权重：

| 字段 | 含义 |
| --- | --- |
| `class_id` / `class_name` | PASTIS 类别 |
| `crop_group` | 冬季谷物、春季谷物、夏季作物、多年生作物等候选粗类 |
| `region` | 资料适用的法国区域或国家级范围 |
| `phase` | 播种、出苗/返青、快速生长、峰值、衰老、收获 |
| `start_period` / `end_period` | 起止月份或旬 |
| `cycle_id` | 多生长周期时区分不同周期 |
| `source` | 论文、政府或农业机构链接 |
| `confidence` | high / medium / low |
| `notes` | 类别映射、地区差异和不确定性说明 |

计划产物：

```text
data/priors/pastis_france_phenology_sources.csv
data/priors/pastis_france_phenology_prior.csv
data/priors/pastis_ext_prior_draft.csv
docs/PHENOLOGY_PRIOR_SOURCES.md
docs/PHENOLOGY_PRIOR_DRAFT.md
docs/PHENOLOGY_LAYER_HANDLING.md
```

类别到检索对象的第一版映射和来源候选已经整理在 [`PHENOLOGY_PRIOR_SOURCES.md`](PHENOLOGY_PRIOR_SOURCES.md)。当前结论是：19 个有效标签中，13 类可以对应主要作物但部分仍有用途/亚型歧义，5 类只能对应作物组，Background 应保持中性。

优先使用法国或欧洲官方农业资料、JRC/FAO 作物日历和同行评议论文。一般网页只能作为线索，不能作为唯一依据。

### 3.3 从资料表生成软矩阵

将生长阶段映射到 12 个自然月，输出 `P_ext[K,12]`。建议遵循：

1. 生长旺盛或最有判别力月份赋较高值。
2. 播种、返青、成熟等过渡期保留中等值。
3. 非生长期保留较低但非零值。
4. 多周期作物允许多个峰值。
5. 低可信度类别的曲线应更平缓，避免强行控制模型。

当前样本序列的数组位置不应被假定为自然年 1 月开始。模型和分析脚本必须使用数据中的 `month_indices` 或 `representative_dates` 查询先验，不能用 `t=0` 直接索引“1 月”。

### 3.4 数据集内部验证

仅使用 train folds `1/2/3`，按类别和真实月份统计：

- Sentinel-2 各波段的均值、中位数和离散程度。
- NDVI 等植被指数的月度曲线。
- 每月原始观测数、插值比例和有效像素数。
- 类别间差异最大的月份。
- 人工先验与训练集曲线的一致和冲突位置。

外部先验与训练数据不一致时不能静默修改。应保留原始 `P_ext`，另生成 `P_data`，把两者差异作为实验和讨论内容。

## 四、两条候选技术路线

### 4.1 路线 A：完整时序 + Meta Data Encoder 软旁路

这是首选主路线。保留全部 12 个月和现有 `temporal_v2` 特征，不重新设计 Galileo 输入。

```text
Galileo temporal features F_t
  -> shared per-month prediction/readout Z_t

calendar month + P_ext[:, month]
  -> Meta Data Encoder
  -> temporal gate / class-month bias

Z_t + prior signal
  -> soft Decision Fusion over T
  -> final segmentation
```

第一版建议使用类别相关的软决策融合，因为它最直接表达“不同作物关注不同月份”：

```text
prior_score[c,t] = P_ext[c, month_t]
weight[c,t] = softmax_t(learned_score[c,t] + alpha * log(prior_score[c,t] + eps))
logits[c] = sum_t weight[c,t] * logits_t[c]
```

其中 `alpha` 表示模型对人工先验的信任程度，应为非负并从接近 0 的位置开始，保证模型可以退回无先验行为。上述公式是设计草案，正式实现前需结合当前 decoder 的输出位置确认张量接口。

后续可比较两种特征级注入：

```text
FiLM:  F'_t = gamma(prior_t) * F_t + beta(prior_t)
AdaGN: F'_t = gamma(prior_t) * GN(F_t) + beta(prior_t)
```

FiLM/AdaGN 与决策融合不能在第一轮同时加入，否则无法归因提升来源。

### 4.2 路线 B：物候窗口筛选或掩码

根据先验只保留部分月份，或为其余月份设置 attention mask。该路线用于检验“删除低价值月份是否有益”，优先级低于路线 A。

主要风险：

1. 推理时未知真实类别，不能按真值类别选择窗口。
2. 不同类别的月份并集可能接近全年，失去筛选意义。
3. 两阶段“先预测粗类再选月份”会引入额外错误传播。
4. 硬删除可能损失收获后地表、冬季覆盖等间接证据。
5. 地区、年份和播期差异会造成物候漂移。

可接受的实现方式包括：

- 保留 `T=12`，用 soft/hard temporal mask 屏蔽月份。
- 先预测粗粒度作物组，再使用组级窗口，但需单独评估第一阶段误差。
- 对每个候选类别在决策层应用不同时间 mask；这本质上仍属于类别条件 Decision Fusion。

## 五、类别层级先验的定位

Hierarchical Semantic Segmentation 可以利用“粗作物组 -> 具体作物”的标签结构，但它解决的是类别层级，不直接解决物候注入。

候选层级必须有农学依据，并对 Background、集合类别和多年生作物给出明确处理。该方向应在时间先验主实验完成后单独开展，不能与 Meta Data Encoder、FiLM 和硬筛月份一次性叠加。

## 六、受控实验矩阵

所有实验固定 train=`fold1/2/3`、val=`fold4`、test=`fold5`、Galileo 权重、输入协议、loss、optimizer、decoder 容量和模型选择规则。除非实验目的就是改变 decoder，否则只改变先验来源或注入方式。

### 6.1 当前冻结的第一阶段：三组最小先验消融

当前主问题不是“先验能否适配所有 decoder”，而是：**在固定的强时序模型上，正确的作物-月份知识是否提供了超出已有 month embedding 的信息。**

因此第一阶段固定使用保留完整 `T=12` 的 `3D-Aware DPT`。不使用已平均时间的 2D decoder，也不把先验同时接到单层、多层、Adapted DPT 与 UPerNet。三组配置除 `phenology` 和日志目录外完全相同：相同 `temporal_v2` 缓存、frozen Galileo、模型、loss、Prodigy、AMP、有效 batch、early stopping 和 `best_val_miou` 选择规则。

| 编号 | 配置 | 先验输入 | 回答的问题 |
| --- | --- | --- | --- |
| P0 | `galileo_3d_aware_dpt_phenology_none.yaml` | 无；`PhenologyPriorAdapter` 不构造 | 当前 3D-Aware DPT 的严格基线 |
| P1 | `galileo_3d_aware_dpt_phenology_ext.yaml` | 正确的 `P_ext` | 人工物候先验是否提升模型 |
| P2 | `galileo_3d_aware_dpt_phenology_ext_shuffled.yaml` | 固定类别置乱的 `P_ext` | 提升是否来自正确类别-月份对应，而非额外分支参数 |

P2 保留每一条原始月份曲线和值域，仅将 class ID `1..18` 映射到另一条曲线；Background (`0`) 保持中性。映射是一个无固定点的固定置乱，种子为 `20260719`，完整的 `class_id -> source_class_id` 写入 `pastis_ext_prior_class_shuffled.csv`，因此任何人都能复现这一错误先验。

执行顺序：先以 seed 42 跑 P0/P1/P2；若 P1 同时优于 P0 和 P2，再以相同配置补 P0/P1 的 seed 43、44。验证集仅用于 early stopping、强度选择和模型选择；实验设计固定后才在 fold5 上评估。`strength=0` 仅用于数值回退检查，不代替 P0，因为模块参数仍会存在。

结论判据：P1 优于 P0 说明加入先验有潜在收益；P1 也优于 P2，才支持“收益来自正确的物候语义”。P1 与 P2 接近时，不能将提升归因于外部农学知识。

### 6.2 后续扩展矩阵（不属于当前第一阶段）

| 编号 | 实验 | 先验 | 注入方式 | 回答的问题 |
| --- | --- | --- | --- | --- |
| K0 | 保留 T 的无先验模型 | 无 | 当前时间融合 | 基准性能 |
| K1 | 月份编码对照 | 仅 month id | month embedding | 显式月份身份是否有作用；当前方案五已具备 |
| K2 | 人工物候先验 | `P_ext` | 类别条件软 Decision Fusion | 外部农学知识是否有效 |
| K3 | 数据驱动先验 | `P_data` | 与 K2 相同 | PASTIS 内部统计是否有效 |
| K4 | 双来源先验 | `P_ext + P_data` | 固定或可学习组合 | 两种知识是否互补 |
| K5 | 人工先验特征调制 | `P_ext` | FiLM 或 AdaGN，二选一 | 注入位置是否重要 |
| K6 | 月份筛选 | `P_ext` | temporal mask | 硬/软筛选是否优于完整时序 |

必须包含以下反事实对照：

- Uniform prior：所有月份相同。
- Class-shuffled prior：打乱作物类别与曲线的对应关系。
- Month-shifted prior：整体平移月份曲线。
- `alpha=0`：确认实现可以恢复无先验结果。

如果真实先验优于这些对照，才能把收益归因于物候知识，而不是额外参数或正则化效果。

## 七、评估和证据要求

每个主要实验至少报告：

- best fold4 val mIoU 和对应 epoch。
- fold5 test mIoU、19 类 per-class IoU、Precision、Recall 和 F1。
- 至少 3 个 seed 的均值和标准差。
- 与 K0 相比的分类别增减。
- 先验置信度与类别提升之间的关系。
- 固定样本的 GT、K0、先验模型和错误图。
- 学到的类别-月份权重与人工先验的对比图。

重点观察光谱相近但物候不同的类别，以及样本稀少、人工先验可信度较低和集合类别。不能只报告总体 mIoU。

## 八、后续工作安排

### P0：先验准备与审计

- [ ] 核对 `0..18` 类别与法国农业资料中的名称映射。
- [ ] 为每个明确作物收集至少一个可靠来源，记录地区、阶段和可信度。
- [ ] 对集合类别制定“宽窗口/混合曲线/中性先验”策略。
- [ ] 生成并人工复核 `P_ext[K,12]`，保留来源到数值的转换说明。
- [ ] 确认所有脚本使用真实月份，而不是时间数组下标。

### P1：训练集物候分析

- [ ] 编写 train-only 分析脚本，不使用 val/test 标签。
- [ ] 输出类别-月份样本数、有效观测数和插值比例。
- [ ] 输出逐类 NDVI 与主要波段月度曲线及不确定性。
- [ ] 生成 `P_data[K,12]`，并与 `P_ext` 做一致性热力图。
- [ ] 根据分析结果确定第一批重点类别和易混淆类别对。

### P2：最小可行旁路实验

- [x] 固定 P0/P1/P2 的 3D-Aware DPT 配置、`best_val_miou` 选择规则和评价入口。
- [x] 在 Galileo `temporal_v2` 特征边界实现共享的先验旁路残差注入：`[B,T,768,H,W] + [B,T,768,1,1]`，随后才进入任意时间保留 decoder。
- [x] 实现 prior loader、外部 `P_ext`、无先验和 class-shuffled 对照表。
- [ ] 等待并验收 train/val/test 三份 `temporal_v2` 缓存。
- [ ] 先以 seed 42 完成 P0/P1/P2，再根据判据决定是否补种子。

### P3：扩展实验

- [ ] 运行 K3 和 K4，比较人工先验与数据先验。
- [ ] 在 K2 有明确收益后，再测试 FiLM 或 AdaGN。
- [ ] 将月份筛选作为 K6 消融，使用 mask 处理固定长度输入。
- [ ] 物候主线稳定后，再讨论粗细类别层级和 HSS。

## 九、协作分工建议

| 工作包 | 主要产物 | 依赖 |
| --- | --- | --- |
| WP1 类别与外部资料 | 来源表、类别映射、`P_ext` | 无 |
| WP2 数据分析 | 曲线、热力图、`P_data` | PASTIS train 数据 |
| WP3 模型接口 | prior loader、旁路、配置与测试 | WP1 的矩阵格式 |
| WP4 训练评估 | K0-K6 结果表、per-class 指标、可视化 | WP2/WP3 |
| WP5 论文表述 | 方法图、先验来源、消融结论和局限性 | WP1-WP4 |

每个物候资料条目和模型结果都需要记录来源或 Git commit。任何涉及先验内容、类别映射或月份顺序的修改，都应同步更新本文档和对应数据文件。

## 十、当前决策

1. 首先完成外部人工先验与训练集统计，不立即叠加多个模型模块。
2. 主路线采用完整 12 月时序与软旁路注入。
3. 第一种注入优先使用类别条件 Decision Fusion，便于解释和消融。
4. 硬月份筛选作为后续实验，不使用真实标签选择月份。
5. FiLM、AdaGN 和层级语义分割均为扩展，不与第一版同时加入。
