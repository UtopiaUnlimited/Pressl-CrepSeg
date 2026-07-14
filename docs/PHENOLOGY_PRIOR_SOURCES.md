# PASTIS 类别与法国物候资料映射审计

本文档是 [`PHENOLOGY_PRIOR_INJECTION_PLAN.md`](PHENOLOGY_PRIOR_INJECTION_PLAN.md) 的资料准备附件。它首先回答“PASTIS 的 19 个有效类别能否与网上查到的 19 种植物一一对应”，再规定后续怎样把来源转成物候先验。

原始映射表维护在：

```text
data/priors/pastis_france_phenology_sources.csv
```

## 一、核心结论：不能全部一一对应

PASTIS 的 19 个有效语义 ID 是 `0..18`，并不等于 19 种独立植物。官方资料说明其中包含一个非农业背景类和 18 个农业地块类别；原始 ID 19 另为 void label，不参与训练。

根据官方类别名，可以分为：

| 类型 | 数量 | 类别 |
| --- | ---: | --- |
| 可直接对应一种主要作物 | 13 | Soft winter wheat、Corn、Winter barley、Winter rapeseed、Spring barley、Sunflower、Grapevine、Beet、Winter triticale、Winter durum wheat、Potatoes、Soybeans、Sorghum |
| 只能对应作物组或土地利用组 | 5 | Meadow、Fruits/Vegetables/Flowers、Leguminous fodder、Orchard、Mixed cereal |
| 非作物 | 1 | Background |

上述 13 类中的 Corn、Spring barley、Beet、Potatoes 和 Sorghum 仍有子类型或用途歧义，因此“可直接对应”不代表已经可以写入单一精确日历。

### 1.1 可以建立作物级日历的类别

以下类别可以查询对应作物的法国播种、生长和收获信息：

- 软质冬小麦、冬大麦、冬油菜、向日葵、冬小黑麦、冬硬粒小麦、大豆。
- 玉米、高粱需要区分籽粒与青贮/饲用用途。
- 春大麦在法国既可能于冬末春初播种，也存在秋播管理方式。
- 葡萄和马铃薯可以对应具体作物，但品种、用途和地区导致周期差异较大。
- Beet 很可能对应法国农业中的甜菜，但官方英文标签只写了 `Beet`；在找到原始法国地块代码到 PASTIS 类别的聚合规则前，只能标为待确认。

### 1.2 不能建立单一作物日历的类别

- `Meadow`：可能由多种禾本科和豆科植物构成，还受到割草、放牧、夏季停长和秋季再生影响。
- `Fruits, vegetables, flowers`：把大量生命周期不同的园艺作物合并成一个标签。
- `Leguminous fodder`：可能包含苜蓿、三叶草、饲用豌豆、野豌豆等一年生或多年生作物。
- `Orchard`：只表示果园土地利用，不提供果树种类和品种。
- `Mixed cereal`：可能是同一谷物的品种混播，也可能是多个谷物或谷物与豆科混播。

这五类不能为了凑齐 19 行而人工指定一种代表植物。第一版应使用中性/宽窗口先验，或优先依赖 train-only 数据曲线。

## 二、PASTIS 的区域并不单一

PASTIS 官方仓库说明数据来自法国本土，共 2,433 个 patch。对本地 `metadata.geojson` 的 2,433 条记录进行审计后，数据分布在四个 Sentinel-2 瓦片：

| Tile | Patch 数 | 几何中心（WGS84，约） | 大致区域 |
| --- | ---: | --- | --- |
| `T30UXV` | 531 | `(-0.8583, 49.0508)` | 法国西北部，Normandy 一带 |
| `T31TFJ` | 623 | `(4.9652, 43.8666)` | 法国东南部，Rhône 下游一带 |
| `T31TFM` | 723 | `(5.0088, 46.4518)` | 法国中东部，Bourgogne-Franche-Comté 周边 |
| `T32ULU` | 556 | `(7.0958, 48.3203)` | Grand Est / Alsace 一带 |

这些区域的纬度、温度和种植制度不同。人工先验至少需要保留地区不确定性；若后续能稳定地将 tile 作为元数据输入，可进一步研究 `P(class, month, region)`，但第一版先使用全国宽窗口并控制先验强度。

## 三、资料源审计

### 3.1 类别名称

类别名称以 [PASTIS 官方 `label_names.json`](https://github.com/VSainteuf/pastis-benchmark/blob/main/documentation/label_names.json) 为准，不能从中文名称反推物种。

PASTIS 官方仓库还说明标签来自法国 Land Parcel Identification System，因此这些名称更接近农业地块申报类别，不保证都是植物分类学意义上的单一物种。

### 3.2 JRC 作物日历的适用边界

欧盟 JRC 的作物日历方法提供播种、生长和收获期字段，并支持国家级/次国家级表达，适合作为先验表的格式参考：

<https://data.jrc.ec.europa.eu/dataset/jrc-10112-10003>

但对当前公开的 `crop_calendar_gaul1.csv` 进行检查后，其中没有法国记录。因此它不能直接提供本项目的法国月份数值，只能支持“按地区、按生长阶段构建作物日历”的方法设计。

### 3.3 法国作物技术来源

第一版优先使用以下机构：

| 作物组 | 优先机构 | 当前覆盖 |
| --- | --- | --- |
| 小麦、大麦、黑小麦、玉米、高粱、马铃薯 | ARVALIS | 已找到类别级或地区级候选页面 |
| 油菜、向日葵、大豆 | Terres Inovia | 已找到播种/收获或地区建议页面 |
| 甜菜 | Institut Technique de la Betterave (ITB) | 已找到播种和季节观测页面；仍需确认 PASTIS Beet 的定义 |
| 葡萄 | INRAE / IFV / 地区 BSV | 已找到物候阶段与地区观测资料 |
| 草地 | Institut de l'Elevage / INRAE | 可描述生长季，但不能还原为单一植物 |
| 果园和综合园艺类 | CTIFL / INRAE / 地区资料 | 必须先确认内部作物组成，否则保持组级先验 |

CSV 中的 `primary_source_url` 目前是每类的首个可靠候选来源，不代表已经完成最终月份标注。后续每个阶段值都应能追溯到具体来源、地区和适用年份。

## 四、映射等级与先验策略

| `mapping_type` | 含义 | 第一版处理 |
| --- | --- | --- |
| `direct` | 类别可以对应一种主要作物 | 构建作物级软曲线，保留地区宽度 |
| `ambiguous` | 作物大体明确，但用途、亚型或管理方式不明确 | 使用混合/宽曲线，先调查原始标签组成 |
| `group` | 标签本身包含多种植物或管理类型 | 中性或数据驱动先验，不指定单一植物 |
| `neutral` | 非作物背景 | 使用中性先验 |

人工曲线不得直接使用绝对 `0/1`。外部来源没有覆盖某个月或某个集合类时，应记录“不确定”，而不是将其解释为“不生长”。

## 五、下一步资料工作

### P0：先解决映射歧义

- [ ] 查找 PASTIS 从法国原始地块代码聚合到 18 类的规则，重点确认 Beet。
- [ ] 判断 Corn 和 Sorghum 是否混合了籽粒与青贮/饲用用途。
- [ ] 判断 Potatoes 是否包含早熟、鲜食、种薯和加工类型。
- [ ] 调查 Spring barley 中秋播与春播样本的可能比例。
- [ ] 查找五个集合类别的原始子类构成；找不到时保持组级策略。

### P1：逐地区填写物候阶段

- [ ] 按四个 tile 对应区域，记录播种、出苗/返青、快速生长、开花/峰值、成熟和收获期。
- [ ] 每条记录保留来源、地区、年份、品种/用途和可信度。
- [ ] 对同一作物的地区差异生成范围，不用单一日期覆盖全法国。
- [ ] 多周期或多类型类别允许多行 `cycle_id`，最后再转换成 12 月软曲线。

### P2：与 PASTIS 训练数据核对

- [ ] 仅使用 folds `1/2/3` 绘制类别-月份 NDVI 和波段曲线。
- [ ] 按 tile 分组，检查外部日历与实际光谱峰值是否一致。
- [ ] 外部资料与数据冲突时保留两套记录，分别生成 `P_ext` 和 `P_data`。
- [ ] 先评审映射和来源，再生成可输入模型的数值矩阵。

## 六、当前可执行结论

1. 不能将 19 个有效标签写成“19 种植物”。
2. 第一批人工物候先验应优先覆盖 13 个作物可识别类别。
3. 其中 5 个歧义类别先使用宽曲线，等待原始标签组成核查。
4. 5 个集合类别和 Background 不应强行套用单一作物日历。
5. 模型必须保留无先验退路，并用 uniform、class-shuffled 和 month-shifted 先验做反事实对照。

