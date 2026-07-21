# 当前执行规划：V4 类别-时序原型记忆先验

状态：**CURRENT / 唯一后续实验规划**

首次整理：2026-07-21

任务：在不改变 Galileo 输入、预训练权重、冻结策略和 3D-Aware DPT 主干的前提下，检验更细粒度的类别-时序先验是否能提高 PASTIS 作物语义分割。

> 本文决定接下来做什么。与 V1--V3 的计划、讲稿或旧运行手册冲突时，以本文为准；结果仍只写入 [DECODER_EXPERIMENTS.md](DECODER_EXPERIMENTS.md)。

> **时间受限执行决定（2026-07-21）：** 本阶段只训练 `V4-K1` 与 `V4-K4` 两个模型。二者除每个类别-月份组的原型数外完全相同；不再并行运行边界头、Lovasz、contrastive loss、M2/M3/M4 或文本先验。

## 1. 先把现状说清楚

### 1.1 已固定且不再改动的视觉基线

```text
PASTIS 月度 Sentinel-2 输入
  -> Galileo 论文输入处理
  -> frozen Galileo encoder
  -> temporal_v2 四层、12 个月特征缓存
  -> 3D-Aware DPT (native deep skip)
  -> 19 类分割 logits
```

- 训练：fold1 + fold2 + fold3；验证：fold4；测试：fold5。
- 选择 checkpoint 只看 fold4 `val_mIoU`；每个候选固定后才允许评估一次 fold5。
- V4 不重新设计 encoder，不改变图像预处理，也不使用测试标签构造任何信息。

### 1.2 V1--V3 的准确定位

| 版本 | 方法 | 现在的定位 |
| --- | --- | --- |
| V1 | 物候 `Global Add` | 历史低复杂度基线；已废弃，不再扩展 |
| V2 | CA-HPI | 通用异构 token 注入对照；保留代码和诊断 |
| V3 | SA-SFiLM，M1+M2+M3+M4 | 当前已实现的多源先验方案；结果未显示稳定增益，停止继续堆叠外部源 |
| V4 | 类别-时序原型记忆 | **下一步唯一主实验** |

V3 的问题不主要是 attention 或 FiLM 不够复杂，而是 M2 气候、M3 土壤和 M4 经纬度都是 patch 级、甚至区域级属性：它们无法在同一张 tile 内区分不同像素的作物类别。M4 保留接口但不再作为 V4 输入；M2/M3 也不再追加新的分辨率版本。

## 2. V4 要回答的研究问题

> 能否用**仅由训练折标签和 frozen Galileo 特征构成**的类别-时序原型库，让每一个视觉 token 查询与自身外观、月份相符的作物原型，从而提供比全局环境属性更细的先验？

这里的“先验”不是手工向每个像素广播一条日历曲线，而是训练集中的可检索类别表征：相同作物在相近月份通常具有相近的时空特征；不同作物或不同月份则应相互可分。

这一思路与“以训练像素的特征均值或多个中心作为类别原型进行稠密预测”的原型分割范式一致，但本项目把原型显式组织为“类别 x 月份”，并保持 Galileo 和 3D-DPT 不变。[Prototype View (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Rethinking_Semantic_Segmentation_A_Prototype_View_CVPR_2022_paper.html) 与 [Class-Wise Memory Bank (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Alonso_Semi-Supervised_Semantic_Segmentation_With_Pixel-Level_Contrastive_Learning_From_a_Class-Wise_ICCV_2021_paper.html) 是方法依据。

## 3. V4 方法设计

### 3.1 原型从哪里来

只使用 fold1--3 的 `temporal_v2` 缓存和对应训练标签，离线构造原型库：

```text
训练样本的 frozen Galileo 最终层特征: [T, 768, 16, 16]
训练 mask 以多数投票下采样到 16 x 16
按 (类别 c, 月份 m) 收集有效空间 token
每组 token 求中心，得到 P[c, m, k]
```

- 第一版 `k=1`：每个“类别-月份”只有一个均值原型，共 `19 x 12 = 228` 个 token。
- 只有第一版出现正信号才尝试 `k=4`：用训练特征聚成多个子原型，表示同类作物的不同长势、地块条件或观测差异。
- 每个原型保存样本数和质量分数；样本过少的类别-月份 token 设为无效，而不是伪造高置信度值。
- 原型文件按 fold、缓存版本、特征层、随机种子和 Git commit 命名，纳入实验台账；不能用 fold4/5 反向更新。

### 3.2 训练过程如何避免标签泄漏

验证和测试时，原型库始终只来自 fold1--3。

训练时采用严格的**交叉建库**：将 fold1--3 的 patch 按 `patch_id` 固定分为 A/B 两半；A 内样本只查询 B 建成的库，B 内样本只查询 A 建成的库。这样当前样本的标签不会以原型形式回流给自身。

这比“直接把本 patch 也放进原型均值”更干净，也让 V4 能面对答辩时的标签泄漏质疑。

### 3.3 原型怎样进入 3D-Aware DPT

V4 第一版只注入 Galileo 的最终层时序特征，不碰四层特征中的浅层空间细节：

```text
F: final Galileo feature               [B, T, 768, 16, 16]
  -> 每个 (月份, 空间位置) token 作为 query
  -> cosine / cross-attention 查询 P[c, m, k]
  -> 得到 prototype context
  -> 小强度、逐 token 门控残差写回 F
  -> 原始四层 temporal pyramid 进入不变的 3D-Aware DPT
```

具体约束：

1. query 看见所有类别和月份的原型，**不知道该像素真实类别**；
2. 原型检索使用类别和月份信息，但由视觉 token 自己决定匹配对象；
3. 残差强度零初始化或极小初始化，保证训练开始时等价于 B0；
4. gate 可将无帮助的原型作用压到零；
5. 记录每类、每月的检索质量、门控均值和实际残差比，不能只报最终 mIoU。

V4 第一版不同时修改 decoder 内部结构、不加 M2/M3/M4、不加文本 token，也不把原型直接作为真值类别 logits。

### 3.4 可选辅助损失，只在主实验成功后启用

若 V4 注入本身在 fold4 有正向信号，才加轻量 prototype contrastive loss：让训练 token 靠近正确的类别-月份原型、远离其它类别原型。它的系数从很小值开始，并与“无辅助损失”的 V4 比较。

这一步不是主方法的前提。若直接注入没有收益，就不靠增加损失硬凑结果。

## 4. 必须按顺序执行的实验

| 编号 | 配置 | 唯一改变 | 回答的问题 | 进入下一步的条件 |
| --- | --- | --- | --- | --- |
| B0 | 无先验 3D-Aware DPT | 无 | 当前视觉基线是否可复现 | 有完整 config、commit、best epoch 与 fold4 记录 |
| P0 | prototype-only 诊断 | 只以原型相似度形成类别分数，不训练注入 | frozen 特征本身是否保留类别-月份结构 | 不作为主结果，只判断信号是否存在 |
| P1 | V4, `k=1` | 正确交叉原型库 + 门控注入 | 原型先验是否优于 B0 | fold4 最佳 mIoU 至少不低于 B0，且诊断非退化 |
| C1 | V4, `alpha=0` | 模块连线存在但残差关闭 | 是否存在接线/参数量假增益 | 数值应与 B0 对齐 |
| C2 | V4, class-shuffled bank | 原型的类别身份随机置换 | 增益是否来自正确类别语义 | 正确 P1 应优于 C2 |
| C3 | V4, month-shifted bank | 原型月份循环偏移 | 增益是否来自正确时间关系 | 正确 P1 应优于 C3 |
| P2 | V4, `k=4` | 单原型改为多原型 | 类内多样性能否带来额外价值 | 仅在 P1 有正信号后运行 |
| P3 | V4 + contrastive loss | 加辅助损失 | 表征约束是否必要 | 仅在 P1 或 P2 有正信号后运行 |

本轮只执行 P1（`K=1`）和 P2（`K=4`）的 seed42：二者训练到统一最大 epoch，并由 fold4 选 `best_val_miou.pt`。先比较两条曲线和 checkpoint，再决定是否评估一次 fold5；C1--C3、额外 seed 与辅助损失留给有时间的后续验证。

## 4.1 本轮可直接执行的命令

以下命令假设当前目录是项目根目录，且已经存在完整的 `temporal_v2` train/val 缓存。原型构建只读取训练缓存、在 CPU 上运行一次；生成的 `.npz` 是本地运行数据，不提交 Git。

```bash
conda run --no-capture-output -n presl python -u -B scripts/build_class_temporal_prototypes.py --config configs/galileo_3d_aware_dpt.yaml --cache-format temporal_v2 --temporal-dtype float16 --prototypes-per-group 1 4
```

生成下列两个文件后，分别训练：

```text
data/priors/class_temporal_prototypes/pastis_fold123_final_layer_class_temporal_prototypes_k1_online_v1.npz
data/priors/class_temporal_prototypes/pastis_fold123_final_layer_class_temporal_prototypes_k4_online_v1.npz
```

```bash
conda run --no-capture-output -n presl python -u -B scripts/train_cached.py --config configs/galileo_3d_aware_dpt.yaml --prior-config configs/prior_injection/class_temporal_prototype_k1.yaml --cache-format temporal_v2 --temporal-dtype float16
```

```bash
conda run --no-capture-output -n presl python -u -B scripts/train_cached.py --config configs/galileo_3d_aware_dpt.yaml --prior-config configs/prior_injection/class_temporal_prototype_k4.yaml --cache-format temporal_v2 --temporal-dtype float16
```

评估时必须使用相同的 `--prior-config`，使 checkpoint 重新加载同一份原型 archive；不能移动或删除这两个 `.npz` 文件。

## 5. 什么叫“有效”，什么叫“及时止损”

### 有效

- P1 在 fold4 相对 B0 的增益稳定为正，且优于 class-shuffled 与 month-shifted 对照；
- 门控和检索并非恒为零或永远选同一个原型；
- 提升不是只来自单一偶然类别，至少有可解释的 per-class IoU 或混淆矩阵变化；
- 固定候选在 seed43/44 的方向不反转。

### 无效，立即停止继续扩展

- P1 与 B0、C2、C3 没有稳定差异；
- gate 长期近零，或残差虽大但 fold4 变差；
- P2 只靠更多 token/参数超过 P1，却不能超过同规模随机 token 对照；
- 只在 fold5 看起来好，但 fold4 没有预先选择依据。

“无效”是可报告的研究结论：对于完整 frozen Galileo 时序表征，训练集类别-时序记忆没有提供超出视觉特征的可迁移信息。此时不再继续堆叠原型模块。

## 6. V4 后的备选路线，不与 V4 并行开工

| 优先级 | 方向 | 为什么值得做 | 触发条件 |
| --- | --- | --- | --- |
| R1 | 边界辅助头 + 边界损失 | 田块边界是 PASTIS 的真实难点；无需外部数据，推理仍只输出语义 mask | V4 结束后，错误图显示边界混淆明显 |
| R2 | 时间子序列一致性 | 对随机缺少月份的输入维持接近完整序列的预测，迫使 decoder 稳定使用时序 | 误差集中在云/缺时相或不同时间长度样本 |
| R3 | `CE + Dice + Lovasz` 与受限类别采样 | 直接对齐 mIoU，并检查少数类是否拖累总体分数 | per-class IoU 显示少数类、混淆而非边界是主要瓶颈 |
| R4 | 独立地块边界数据 | 若能取得与标签独立、推理可得的法国地块矢量，可作为空间结构输入 | 必须先确认时间、区域覆盖与非标签泄漏 |

不能将 PASTIS `INSTANCES` 标注直接作为正式模型输入。它可作为 oracle 上界或训练期边界监督的标签来源，但不是部署时自然可得的外部先验。

## 7. 本阶段明确不做的事情

- 不继续追逐高分辨率气象下载；即使分辨率改善，它仍是区域背景而不是像素级作物证据。
- 不用 M4 经纬度作为主输入；保留接口即可。
- 不同时叠加 V4、边界头、Lovasz、采样、文本先验和新 decoder。
- 不以 test mIoU 选超参数、挑 checkpoint 或决定是否保留某个模块。
- 不把 cache、checkpoint、日志或原始大数据提交到 Git。

## 8. 当前交付状态

1. [x] `scripts/build_class_temporal_prototypes.py`：按训练折和 A/B 交叉建库规则生成 `K=1` 与 `K=4` archive。
2. [x] `models/prototype_memory.py`：最终层按月份检索、门控残差写回和 TensorBoard 诊断。
3. [x] `configs/prior_injection/class_temporal_prototype_k1.yaml` 与 `class_temporal_prototype_k4.yaml`：本轮两条训练配置。
4. [x] 单元测试：archive shape、A/B 交叉选择、`alpha=0` 回退和 3D-DPT 前向连接。
5. [ ] 实验台账：待记录两个模型的 config、commit、最佳 epoch、fold4 指标和最终选中模型的一次 fold5 结果。

class-shuffled、month-shifted、contrastive loss 和多 seed 将在本轮结果具有正信号后再实现。

## 9. 一句话口径

> V4 不再向每个像素广播低分辨率环境属性，而是从训练折中构建类别-月份原型记忆，让每个视觉时空 token 基于自身内容检索可解释的作物表征，并以受控残差注入固定的 3D-Aware DPT；通过类别置乱、月份偏移和交叉建库验证它学到的是正确知识，而非标签泄漏或额外参数带来的偶然增益。
