# [ARCHIVED] 旧 Global Add 研究逻辑摘要

状态：**ARCHIVED / 仅用于理解与复现旧接口**

归档日期：2026-07-18

> 标题中的旧“一条主线”已经失效。当前唯一研究主线见 [NEXT_STAGE_HETEROGENEOUS_PRIOR_INJECTION_PLAN_2026-07-17.md](NEXT_STAGE_HETEROGENEOUS_PRIOR_INJECTION_PLAN_2026-07-17.md)。完整旧版保留在 Git 历史中。

## 旧接口的数据流

所有旧物候实验读取时间保留缓存：

```text
F: [B,L=4,T=12,768,16,16]
months: [B,T]
```

外部表 `P_ext[19,12]` 按真实自然月份取值，经适配器生成全局残差：

```text
q: [B,T,768,1,1]
F' = F + q
```

再把 `F'` 交给 Temporal Readout 或 3D-Aware decoder。该设计的优点是先验 overlay 与 decoder config 分离；局限是同一残差广播给所有空间位置，且注入点绑定 Galileo 的 768 维特征边界。

## 旧配置组合

```text
base decoder config + optional --phenology-config overlay
```

- 无 overlay：P0；
- `configs/phenology/external.yaml`：P1；
- `configs/phenology/external_class_shuffled.yaml`：P2。

这个组合接口仍可用于复现旧方法，但不再代表新 CA-HPI。旧命令见 [PHENOLOGY_RUNBOOK.md](PHENOLOGY_RUNBOOK.md)。

## 必须记住的历史事实

1. 已有 P1/P2 结果来自 **Single-layer Temporal Readout**，不是 3D-Aware；
2. 旧代码的先验维度是 Galileo `768`，3D-Aware 内部工作维度是 `256`；
3. Temporal Readout 与空间 decoder 是两个模块；
4. 旧 overlay 不读取像素真实类别，因此没有直接标签泄漏；
5. 但它缺少内容感知选择与置信度门控，因此研究能力有限。

## 与新方法的关系

新 CA-HPI 保留“统一先验接口、真实月份、错误先验对照和 decoder 解耦”四个原则，也保留 decoder 前公共边界；废弃的是“在该边界全局广播相加”的机制。新模块由每层视觉时空 token 查询完整先验 token 集合，再把结果交给任意 time-preserving decoder。

此文件只能用于解释旧结果或复现旧旁路，不能用于决定下一项实验。
