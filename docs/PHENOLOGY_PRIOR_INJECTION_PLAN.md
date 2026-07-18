# [ARCHIVED] 固定物候表与 Global Add 旁路计划摘要

状态：**ARCHIVED / 旧物候注入方案**

原始阶段：2026-07-14 至 2026-07-17

归档日期：2026-07-18

> 本文件只记录旧方案的来龙去脉，不再是当前实施计划。新方法见 [NEXT_STAGE_HETEROGENEOUS_PRIOR_INJECTION_PLAN_2026-07-17.md](NEXT_STAGE_HETEROGENEOUS_PRIOR_INJECTION_PLAN_2026-07-17.md)，完整旧版保留在 Git 历史中。

## 旧方案回答的问题

旧研究问题是：在 frozen Galileo 的逐月特征上，人工整理的法国作物物候表能否作为软先验改善分割？当时把知识表示为：

```text
P_ext[class, calendar_month] in [0, 1]
```

并区分了以下概念：

- month embedding 只表示自然月份，不等于作物物候；
- `P_ext` 来自外部农学资料；
- `P_data` 只能由 train folds 统计；
- test 真值不得参与先验生成或按类选择月份；
- 聚合、多年生和低可信类别应使用宽窗口、低置信度或 unknown，而不是伪造精确曲线。

这些数据治理原则仍然有效，来源审计见 [PHENOLOGY_PRIOR_SOURCES.md](PHENOLOGY_PRIOR_SOURCES.md) 和 [PHENOLOGY_PRIOR_DRAFT.md](PHENOLOGY_PRIOR_DRAFT.md)。

## 旧实现是什么

旧 `Global Add` 在任何 decoder 之前执行：

```text
P_ext[:, month_t]
  -> small adapter
  -> q[t] [B,T,768,1,1]

F'[l,t] = F[l,t] + q[t]
```

同一月份残差广播到全部空间位置和选中的 Galileo 层。它没有读取像素真值，但也不能根据图像内容选择相关类别知识，因此只保留为最低复杂度历史基线。

## 已完成实验的准确映射

- 已完成 P0/P1/P2 使用 **Single-layer Temporal Readout**；
- P0：无先验；P1：正确外部表；P2：固定 class-shuffled 表；
- 它们不是 3D-Aware DPT 实验；
- 完整总体指标与 19 类结果见 [DECODER_EXPERIMENTS.md](DECODER_EXPERIMENTS.md)。

旧结果只产生弱信号，不足以说明这种注入方式适合困难类别，也不构成把相同旁路扩到所有 decoder 的理由。

## 对当前工作的价值

保留三项历史价值：

1. 可追溯物候来源、自然月份索引和不确定性规则；
2. correct / class-shuffled / month-shifted / uniform 等反事实思想；
3. `Global Add` 作为新 CA-HPI 的低复杂度参考。

不再执行的旧路线：

- 不把一张 `class × month` 软矩阵视为完整物候建模；
- 不继续把同一个全局残差平铺到更多 decoder；
- 不以旧 P1/P2 的前几个 epoch 或单次 test 决定新方向；
- 不把月份筛选、FiLM、Decision Fusion 和层级监督一次性叠加。

## 当前接续关系

旧方案中的原始资料转为新方法的 task adapter 输入；旧 Global Add 被 CA-HPI 的“视觉内容查询完整先验库 + confidence-aware attention + zero-init gate”取代。物候仍是应用实例，但核心创新已经转向通用异构先验注入。
