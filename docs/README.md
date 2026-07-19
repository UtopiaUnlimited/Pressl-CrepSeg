# 项目文档导航与状态

最后更新：2026-07-18

> 先看本页，再决定读取哪份文档。`CURRENT` 决定现在做什么；`REFERENCE` 只提供事实与资料；`ARCHIVED/LEGACY` 只解释已经过去的工作，不能拿来生成新的任务清单。

## 建议阅读顺序

1. [当前唯一执行规划](NEXT_STAGE_HETEROGENEOUS_PRIOR_INJECTION_PLAN_2026-07-17.md)
2. [正式实验数据台账](DECODER_EXPERIMENTS.md)
3. [异构先验相关论文调研](RELATED_WORK_HETEROGENEOUS_PRIOR_INJECTION_2026-07-17.md)
4. [物候资料来源审计](PHENOLOGY_PRIOR_SOURCES.md)
5. 需要服务器操作时看 [服务器使用指南](../服务器使用指南.md)

## CURRENT：当前有效

| 文档 | 作用 | 使用规则 |
| --- | --- | --- |
| [NEXT_STAGE_HETEROGENEOUS_PRIOR_INJECTION_PLAN_2026-07-17.md](NEXT_STAGE_HETEROGENEOUS_PRIOR_INJECTION_PLAN_2026-07-17.md) | **唯一总规划**；定义 3D-Aware 基线、CA-HPI 方法、插入点与实验顺序 | 发生冲突时以它为准 |
| [DECODER_EXPERIMENTS.md](DECODER_EXPERIMENTS.md) | 正式实验台账；保存总体、逐类结果和审计状态 | 新结果只追加到这里，不从讲稿反推数据 |
| [RELATED_WORK_HETEROGENEOUS_PRIOR_INJECTION_2026-07-17.md](RELATED_WORK_HETEROGENEOUS_PRIOR_INJECTION_2026-07-17.md) | 当前方法的论文依据与相关工作矩阵 | 用于方案论证和正式引用核对 |
| [M2_M3_CLIMATE_SOIL_PRIOR_CATALOG.md](M2_M3_CLIMATE_SOIL_PRIOR_CATALOG.md) | M2 气象与 M3 土壤先验的资料、字段、接入和泄漏边界 | 冻结训练表、CA-HPI 接口与可直接运行的实验命令 |
| [M4_GEOGRAPHIC_PRIOR_AND_SOURCE_BALANCING.md](M4_GEOGRAPHIC_PRIOR_AND_SOURCE_BALANCING.md) | M4 地理上下文、通用静态数值 adapter 与多源注意力平衡 | M4 数据边界、受控配置和诊断解释 |
| [服务器使用指南.md](../服务器使用指南.md) | 当前服务器操作说明 | 只用于人工指导；Codex 不自动连接服务器 |

## REFERENCE：仍有用，但不决定路线

| 文档 | 保留内容 | 注意事项 |
| --- | --- | --- |
| [PHENOLOGY_PRIOR_SOURCES.md](PHENOLOGY_PRIOR_SOURCES.md) | PASTIS 类别、法国物候资料和映射可信度 | 当前物候 task adapter 的资料依据 |
| [PHENOLOGY_PRIOR_DRAFT.md](PHENOLOGY_PRIOR_DRAFT.md) | `P_ext` 草案、来源到数值的审计、已有资产 | 其中旧训练命令属于 Global Add 历史流程 |
| [PHENOLOGY_LAYER_HANDLING.md](PHENOLOGY_LAYER_HANDLING.md) | Galileo 多层/时间张量和旧旁路边界分析 | 新 CA-HPI 只保留 decoder 前公共边界 |
| [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) | Galileo、PASTIS、时序遥感和 decoder 背景文献 | 基线背景，不是当前方法清单 |
| [next.md](next.md) | 19 类映射与早期问答 | 优先用于查类别名；“后续实验”部分为历史讨论 |

## ARCHIVED / LEGACY：已经过去的工作

以下文档不得再被解释为“接下来要做”：

| 文档 | 历史定位 |
| --- | --- |
| [NEXT_STAGE_CLASS_TARGETED_PLAN_2026-07-16.md](NEXT_STAGE_CLASS_TARGETED_PLAN_2026-07-16.md) | 已压缩；困难类别定向优化的旧候选主线 |
| [PHENOLOGY_PRIOR_INJECTION_PLAN.md](PHENOLOGY_PRIOR_INJECTION_PLAN.md) | 已压缩；固定类别—月份表与 Global Add 旧计划 |
| [PHENOLOGY_RESEARCH_LOGIC.md](PHENOLOGY_RESEARCH_LOGIC.md) | 已压缩；旧 768 维全局旁路数据流 |
| [PHENOLOGY_RUNBOOK.md](PHENOLOGY_RUNBOOK.md) | 旧 Global Add overlay 的复现命令，不是 CA-HPI 手册 |
| [DEFENSE_ISSUES_AND_ACTIONS_2026-07-13.md](DEFENSE_ISSUES_AND_ACTIONS_2026-07-13.md) | 答辩后的阶段问题清单 |
| [PROGRESS_REPORT_2026-07-11.md](PROGRESS_REPORT_2026-07-11.md) | 早期 baseline 阶段报告 |
| [PROGRESS_REPORT_2026-07-13_LINEAR_COMPARISON.md](PROGRESS_REPORT_2026-07-13_LINEAR_COMPARISON.md) | 线性 head 与 decoder 阶段对比 |
| [MIDTERM_PRESENTATION_SCRIPT.md](MIDTERM_PRESENTATION_SCRIPT.md) | 中期汇报讲稿 |
| [PROGRESS_PRESENTATION_SCRIPT_2026-07-16.md](PROGRESS_PRESENTATION_SCRIPT_2026-07-16.md) | 2026-07-16 阶段汇报讲稿 |
| `../服务器使用指南(1).md` | 已压缩；旧服务器指南副本 |

这些文件仍保留，是为了复盘决策、审计历史结论和复现旧实验。需要完整旧版时查 Git 历史，不要把大段旧计划复制回当前文档。

## 三条防混淆规则

1. **基线：** 后续方法开发固定 3D-Aware DPT；旧 P1/P2 实际基于 Single-layer Temporal Readout。
2. **方法：** 当前创新是 CA-HPI 异构先验注入；旧 `Global Add` 只是历史基线。
3. **记录：** 规划写入当前唯一总规划，结果写入实验台账，资料证据写入来源/论文文档，三者不要混写。
