# `P_ext` 先验草案说明

> 状态：研究草案，不是最终物候标注。数值用于验证“外部物候知识是否有增益”，不能直接解释为法国所有区域的精确月度概率。

## 1. 这张表解决什么问题

当前文件：

```text
data/priors/pastis_ext_prior_draft.csv
```

它把 `PASTIS` 的 19 个有效训练类别整理成一个按自然月份排列的软先验矩阵：

```text
P_ext[class_id, calendar_month] in [0, 1]
```

每一行的数值表达“这个月份对该类别的潜在判别价值/活动程度”，不是严格概率。模型使用前应通过 `month_indices` 把自然月份重新对齐到当前样本的 12 个时间步；不能假设 `t=0` 就是 1 月。

## 2. 为什么不是 19 种植物的一一对应

PASTIS 官方补充材料明确说明：法国 FLPIS 原始作物分类有 73 个类别，数据集选择其中满足样本量和区域覆盖条件的类别，形成 18 类 nomenclature；未进入这 18 类的地块使用 void label。官方表中另有一个非农业背景类。因此：

- `0 Background` 不是作物；
- `1..18` 是 PASTIS 的任务类别，不等于 18 个独立植物学物种；
- `19 Void` 不参加训练和指标，不能生成物候先验；
- `Meadow`、`Fruits, vegetables, flowers`、`Leguminous fodder`、`Orchard`、`Mixed cereal` 本身就是土地利用或混合类别；
- `Corn`、`Spring barley`、`Beet`、`Potatoes`、`Sorghum` 虽然有明确作物名，但用途、亚型或管理方式仍可能改变周期。

原始依据：

- [PASTIS 官方补充材料](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Garnot_Panoptic_Segmentation_of_ICCV_2021_supplemental.pdf)
- [PASTIS 官方仓库](https://github.com/VSainteuf/pastis-benchmark)
- [逐类来源候选表](../data/priors/pastis_france_phenology_sources.csv)

## 3. 当前映射分层

| 层级 | 类别 ID | 处理方式 |
| --- | --- | --- |
| A：主要作物明确 | 2, 4, 5, 7, 8, 10, 11, 15 | 可以先使用来源支持的宽窗口，仍保留区域不确定性 |
| B：作物明确但用途/亚型不明 | 3, 6, 9, 13, 18 | 使用更宽的草案窗口，不得当成精确物候日历 |
| C：聚合或混合类别 | 1, 12, 14, 16, 17 | 当前矩阵使用均匀占位，优先由 `P_data` 或成分分析替代 |
| D：非农业背景 | 0 | 均匀占位，不注入作物知识 |
| 排除 | 19 | void，不进入训练、验证、测试指标或先验矩阵 |

## 4. 数值如何解释

草案采用非零软权重，例如：

- `1.0`：候选的高判别价值/高活动窗口；
- `0.7-0.9`：较强但受区域、品种或管理影响的窗口；
- `0.3-0.6`：过渡期、冬季覆盖或收获后的残留信息；
- `0.1-0.2`：低权重月份，但不强行判定为“不可能”。

这样设计是为了让模型能够反驳不准确的人工先验。第一版不使用 0/1 硬屏蔽，也不把“没有资料”解释为“该月份没有作物”。

## 5. 建议的使用顺序

1. **K0：无外部先验。** 固定当前 encoder、输入时间处理、decoder、loss 和 checkpoint 选择规则，作为唯一基准。
2. **K2-A：只注入 A 类。** B/C/D 类使用均匀先验；先观察明确作物是否受益。
3. **K2-B：注入 A+B。** 检验宽窗口和歧义处理是否引入错误偏置。
4. **K3：训练集统计先验。** 只使用训练 folds `1/2/3`，构造 `P_data`，不使用验证或测试标签。
5. **K4：比较或融合 `P_ext` 与 `P_data`。** 外部知识和数据统计必须保留独立版本，不能事后根据测试结果改写外部表。

所有实验都应保留：

- `alpha=0` 的回退结果；
- uniform prior 对照；
- class-shuffled prior；
- month-shifted prior；
- 总体 mIoU、每类 IoU，以及与先验置信度对应的类别变化。

## 6. 这张表还不能支持的结论

目前不能据此声称“某个 PASTIS 类别在法国的某几个月一定生长”。PASTIS 覆盖四个 Sentinel-2 区域，气候、品种、播期、收获方式不同；同时 PASTIS 类别来自土地登记体系，不是为物候研究设计的物种级标签。最终论文中应把这张表称为：

> source-backed soft phenology prior hypothesis

而不是 ground-truth phenology calendar。

下一步必须用训练集 `fold1/2/3` 的逐类、逐月 NDVI/光谱曲线检查：外部先验的峰值是否和数据中的变化一致；若冲突，应保留冲突作为实验结果，并生成独立的 `P_data`，不能直接修改 `P_ext`。

## 7. 已生成的训练集数据证据

已经运行 [`scripts/analyze_train_phenology.py`](../scripts/analyze_train_phenology.py)，仅使用训练 folds `1/2/3` 的 1455 个 patch，沿用项目的 `aggregate_monthly_s2(start_offset=1, num_timesteps=12)`，并用 PASTIS 的 B4/B8 计算逐类 NDVI 描述统计。

输出文件：

- [`pastis_train_ndvi_stats.csv`](../data/priors/pastis_train_ndvi_stats.csv)：19 个有效类别 × 12 个自然月份的均值、标准差、像素数和 patch 数；
- [`pastis_data_prior_draft.csv`](../data/priors/pastis_data_prior_draft.csv)：将每类月均 NDVI 做类内 min-max 缩放得到的软先验候选；
- [`pastis_train_phenology_manifest.json`](../data/priors/pastis_train_phenology_manifest.json)：折划分、聚合方式、波段和 void 处理记录。

这套 `P_data` 只回答“训练数据中呈现出怎样的月度光谱变化”，不等于真实物候日历，也不替代外部来源。下一步应把 `P_ext`、`P_data`、uniform 三者画成热力图，并在统一的 `temporal_v2` 缓存上分别做对照。

## 8. 当前交接状态

### 现在可以直接运行

- 生成或读取统一的 `temporal_v2` 特征缓存；
- 运行现有的单层 DPT、多层 DPT、UPerNet-style 和 3D-Aware DPT；
- 运行 [`analyze_train_phenology.py`](../scripts/analyze_train_phenology.py) 重新生成训练集 `P_data`；
- 审阅和修改 `P_ext` 草案、来源表以及层级处理方案。

### 仍需继续实现

物候旁路目前已经接入 `3D-Aware DPT`，但还没有覆盖所有未来 decoder 变体：

- temporal single-layer DPT 的统一接口；
- temporal multi-layer DPT 和 temporal UPerNet 的对应实现；
- 多随机种子和 test-only 最终评估流程。

当前第一阶段不需要覆盖所有 decoder。三组最小对照已经准备好：无先验、正确外部先验、固定 class-shuffled 外部先验。它们只依赖统一的 `temporal_v2` 缓存；详细科研判据见 [`PHENOLOGY_PRIOR_INJECTION_PLAN.md`](PHENOLOGY_PRIOR_INJECTION_PLAN.md#61-当前冻结的第一阶段三组最小先验消融)。在完成该对照前，不应报告“物候先验已经普遍提升了所有 decoder”。

### `temporal_v2` 缓存验收条件

P0/P1/P2 只能读取下列目录，分别对应 train、val、test：

```text
data/cache/galileo-base-patch8/monthly12_tile64_patch4_hl3-6-9-12_temporal-v2_tfp16_train/
data/cache/galileo-base-patch8/monthly12_tile64_patch4_hl3-6-9-12_temporal-v2_tfp16_val/
data/cache/galileo-base-patch8/monthly12_tile64_patch4_hl3-6-9-12_temporal-v2_tfp16_test/
```

每个 `.npz` 必须含有：

```text
temporal_features_by_layer: [4, 12, 768, 16, 16]
months: [12]
target: [64, 64]
```

旧的空间缓存即使含有 `months`，只要它保存的是 `features: [768,16,16]` 而没有 `temporal_features_by_layer`，就已经平均掉 T，不能用于物候旁路。

### 远端缓存训练命令

以下命令应在**缓存所在机器的项目根目录**运行。将 `$envName` 替换为该机器实际的 Conda 环境名称；将 `$trainCache`、`$valCache` 替换为组员生成的 `temporal_v2` 目录。训练阶段只使用 train/val cache，fold5 test cache 留到 P0/P1/P2 的实验设计和验证集模型选择固定后再使用。

```powershell
$envName = "llm"
$trainCache = "D:\\path\\to\\monthly12_tile64_patch4_hl3-6-9-12_temporal-v2_tfp16_train"
$valCache = "D:\\path\\to\\monthly12_tile64_patch4_hl3-6-9-12_temporal-v2_tfp16_val"
```

先用 P0 做极小 smoke test，确认缓存键、张量形状、CUDA 和 DataLoader 都正常：

```powershell
conda run -n $envName python -B scripts/train_cached.py `
  --config configs/galileo_3d_aware_dpt_phenology_none.yaml `
  --train-cache-dir $trainCache `
  --val-cache-dir $valCache `
  --cache-format temporal_v2 `
  --temporal-dtype float16 `
  --max-train-batches 2 `
  --max-val-batches 2 `
  --epochs 1 `
  --device cuda
```

smoke test 成功后，以完全相同的 cache 和设备依次训练 P0、P1、P2。三者配置默认均为 seed 42；不要在三次之间修改 batch size、loss、optimizer 或 early stopping。

```powershell
# P0：严格无先验基线
conda run -n $envName python -B scripts/train_cached.py `
  --config configs/galileo_3d_aware_dpt_phenology_none.yaml `
  --train-cache-dir $trainCache `
  --val-cache-dir $valCache `
  --cache-format temporal_v2 `
  --temporal-dtype float16 `
  --device cuda

# P1：正确外部物候先验
conda run -n $envName python -B scripts/train_cached.py `
  --config configs/galileo_3d_aware_dpt_phenology_ext.yaml `
  --train-cache-dir $trainCache `
  --val-cache-dir $valCache `
  --cache-format temporal_v2 `
  --temporal-dtype float16 `
  --device cuda

# P2：类别对应置乱的错误先验
conda run -n $envName python -B scripts/train_cached.py `
  --config configs/galileo_3d_aware_dpt_phenology_ext_shuffled.yaml `
  --train-cache-dir $trainCache `
  --val-cache-dir $valCache `
  --cache-format temporal_v2 `
  --temporal-dtype float16 `
  --device cuda
```

当前旁路实现已经接入 `3D-Aware DPT`：先验由 `P[class, month]` 经过 `PhenologyPriorAdapter` 投影到 decoder 通道，在每层 `Reassemble3D` 之后、任何 temporal attention 和跨层融合之前残差注入。Galileo encoder 保持冻结，时间轴保持完整。

消融时使用：

- 无先验 P0：`configs/galileo_3d_aware_dpt_phenology_none.yaml`；
- 外部先验：`configs/galileo_3d_aware_dpt_phenology_ext.yaml`，`phenology.enabled: true`；
- 错误先验 P2：`configs/galileo_3d_aware_dpt_phenology_ext_shuffled.yaml`；
- 数据先验：作为 P1 有明确收益后的下一阶段，只将 `phenology.path` 改为 `data/priors/pastis_data_prior_draft.csv`。
