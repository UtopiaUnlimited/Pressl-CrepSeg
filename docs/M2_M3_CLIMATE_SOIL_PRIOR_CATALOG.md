# M2/M3 气象与土壤先验资料规范

最后更新：2026-07-18
状态：M2/M3 原始资料已下载、训练表已冻结、CA-HPI adapter 已接入并通过单元测试；尚未产生正式训练结果。

## 1. 目的与边界

当前 CA-HPI 首轮 M1 使用的是 19 类 x 12 月的法国作物物候表。M2 和 M3 不替换 M1，而是为每个 PASTIS 样本补充两类与标签独立的环境知识：

```text
M1: 作物类别 x 月份的外部物候知识
M2: 样本位置 x 月份的气象环境
M3: 样本位置的静态表层土壤环境
```

所有环境先验只在 Galileo temporal feature pyramid 和 decoder 之间，由 CA-HPI 注入；不修改 Galileo encoder、输入影像、目标 mask 或 decoder 内部结构。

```text
temporal_v2 Galileo features [B,L,T,768,16,16]
  + M1 class-month phenology tokens
  + M2 patch-month climate tokens
  + M3 patch-static soil tokens
  -> shared CA-HPI
  -> 3D-Aware DPT
```

`M2`、`M3` 是资料类别和实验编号。当前 `prior_injection.sources` 支持 `phenology_table` / `class_month_table`、`climate_table` 和 `soil_table`；三类 token 可单独或组合后由同一 CA-HPI 接入时序特征金字塔。

## 2. 统一的科学规则

1. 先验在 train、val、test 三个 split 都必须可获得；不得读取目标 mask、预测结果或类别真值。
2. 数据源、时间范围、空间提取方法、字段单位、缺失处理和版本哈希要在训练前冻结。
3. 特征标准化、缺失值填补参数和土壤不确定性标度只能用 folds 1/2/3 拟合；fold4 只用于模型选择，fold5 只做一次最终评估。
4. 现有 PASTIS folds 在同一批法国区域内随机划分。因此环境先验在该协议下不构成标签泄漏，但可能利用地区共现规律；后续需要补一个按 Sentinel-2 tile 或区域留出的空间泛化评估。
5. 任何来自 AI 的建议只能作为检索和整理辅助。最终数值必须来自可引用资料，不能让 AI 自行编造月度权重、土壤值或作物分布。

PASTIS 的 `metadata.geojson` 已保存 patch geometry 和 `ID_PATCH`，缓存中保存 `patch_id`、`sample_id`、`tile_id`、`months` 与 `dates`。这足以在后续由 `patch_id` 关联外部环境资料；当前缓存不保存经纬度，因此需要额外建立 metadata lookup，而不是重做 Galileo 特征缓存。

## 3. M2：ERA5-Land 月度气象先验

### 3.1 选定资料源

- 数据集：ECMWF / Copernicus Climate Data Store 的 ERA5-Land monthly averaged data。
- 官方页面：<https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=overview>
- 技术说明：<https://confluence.ecmwf.int/pages/viewpage.action?pageId=505384848>
- 覆盖：1950 年至今，月尺度；CDS 常规网格为 0.1 x 0.1 度，原生陆面再分析约 9 km。
- 许可：CDS 页面标记为 CC-BY；下载前仍需记录实际访问日期、数据集 DOI `10.24381/cds.68d2bb30` 和请求参数。

它适合提供区域天气背景，不能被表述为对 64 m 像素的实测天气。一个 ERA5-Land 网格会覆盖大量 PASTIS 子块，这是它的限制，也是它作为低频环境先验而非第二幅高分辨率影像的原因。

### 3.2 第一版字段

对每个 patch 的中心点，按项目现有作物年 `2018-10` 至 `2019-09` 提取以下 12 个自然月。ERA5-Land 海面格可能为无数据，因此自动脚本固定选择距离中心点最近、且四个变量在 12 个月内均有效的陆地格；若最近有效格超过 `0.25` 度则报错，而不是填零或静默借用远处区域。

| 字段 | ERA5-Land 变量 | 保存单位 | 用途 |
| --- | --- | --- | --- |
| `t2m_c` | `2m_temperature` | 摄氏度 | 热量条件、物候提前/滞后 |
| `tp_mm` | `total_precipitation` | mm/月 | 水分供给 |
| `ssrd_mj_m2` | `surface_solar_radiation_downwards` | MJ/m2/月 | 光照与能量输入 |
| `swvl1` | `volumetric_soil_water_layer_1` | 体积含水量 | 表层土壤湿润背景 |

转换规则：`t2m_c = K - 273.15`。对 `monthly_averaged_reanalysis`，降水和辐射累积量是**日累计的月平均值**，所以 `tp_mm = m * 1000 * 当月天数`，`ssrd_mj_m2 = J/m2 / 1e6 * 当月天数`；`swvl1` 不转换。保留原始值和转换后的值的生成脚本；模型只读取经过 train-fold 标准化后的转换值。

不把 `potential_evaporation`、风速、雪深等一次全部加入第一版，避免先改变变量数量又改变融合机制。M2 首轮只验证这四项是否带来稳定信息。

### 3.3 冻结数据表

建议生成并版本化小型表，而不是在训练时联网请求：

```text
data/priors/era5_land_pastis_cropyear_v1.csv
```

每行是一条 `patch_id x month` 记录：

```csv
patch_id,year,month,lon,lat,era5_lat,era5_lon,t2m_c,tp_mm,ssrd_mj_m2,swvl1,valid,source_version
10009,2018,10,...
```

`valid=false` 时产生 masked token，不用零值冒充无降水或低温。第一版的 token confidence 仅表达资料可用性：有效 ERA5 记录为 `1.0`，缺失记录为 `0.0` 并 mask；不能把它解释成逐格观测误差。

### 3.4 CA-HPI token 设计

新增 `PatchClimatePriorEncoder` 后，每个样本输出 12 个 climate token：

```text
numeric_values: [B,12,4] = [temperature, precipitation, radiation, soil_water]
time_values:    [B,12,1] = month / 12
type_id:        climate
entity_id:      optional ERA5 grid-cell id，不作为必需输入
mask/confidence:[B,12]
```

M2 token 不携带任何真实作物类别。视觉 token 只根据自身影像特征和月份，决定是否查询某个月份的气象上下文。

### 3.5 M2 禁止项

- 不使用按 PASTIS ground truth、fold4 或 fold5 调整过的天气权重。
- 不使用同一 patch 的标签统计去构造“该地区更可能是什么作物”的气象表。
- 不把 ERA5 网格值插值为虚假的 10 m 精度。
- 不在每次训练时重新请求 CDS；必须下载一次、固定请求 JSON 与文件哈希。

## 4. M3：SoilGrids 静态土壤先验

### 4.1 选定资料源

- 数据集：ISRIC SoilGrids 2.0。
- 官方资料页：<https://docs.isric.org/globaldata/soilgrids/index.html>
- 字段与单位：<https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html>
- 批量子集获取：<https://docs.isric.org/globaldata/soilgrids/wcs.html>
- 覆盖：全球 250 m、六个标准深度层；公开数据带预测分位数，Q0.05 与 Q0.95 构成 90% 预测区间。

SoilGrids 是全球模型预测的土壤属性图，不是 PASTIS 地块实测。它适合作为带不确定性的静态环境 context，不能称为真实地块采样结果。

### 4.2 第一版字段和深度

第一版只取农作物根系最相关的 `0-5 cm`、`5-15 cm`、`15-30 cm` 三层；每层先取下列六项中位/均值产品和 Q0.05/Q0.95：

| 字段 | SoilGrids 名称 | 常规单位 | 保留理由 |
| --- | --- | --- | --- |
| `ph` | `phh2o` | pH | 土壤酸碱度 |
| `soc_gkg` | `soc` | g/kg | 有机质背景 |
| `clay_pct` | `clay` | % | 持水、质地 |
| `sand_pct` | `sand` | % | 与 clay 共同表示质地 |
| `cec_cmolkg` | `cec` | cmol(c)/kg | 养分保持能力 |
| `nitrogen_gkg` | `nitrogen` | g/kg | 氮素背景 |

不同时加入 `silt`，因为 sand、silt、clay 组成和为 100%，直接全加会引入冗余。`bulk_density`、coarse fragments、water content 可作为 M3 后续消融，不进入首轮。

### 4.3 空间提取与不确定性

1. 先按 `patch_id` 从 `metadata.geojson` 取 geometry 和中心点。
2. 当前自动化的 **M3-center-v1** 对每个 128 x 128 patch 的中心点取最近一个 SoilGrids 250 m 像元，四个 64 x 64 tile 共用该 patch context；它是可复现、低成本的基线，但不能表述为地块面积平均值。
3. 若 M3-center-v1 有正信号，M3-v2 再对 patch polygon 内像元做中位数/面积汇聚，并与中心点版本单独对比，检验更细空间对齐是否值得。
4. SoilGrids 原始整数值必须按官方 conversion factor 转为常规单位。
5. 每个属性同时读取 Q0.05、Q0.95；不确定度为 `IQR90 = Q0.95 - Q0.05`。

建议冻结的表：

```text
data/priors/soilgrids_pastis_surface_v1.csv
```

核心字段：

```csv
patch_id,depth_cm,ph,soc_gkg,clay_pct,sand_pct,cec_cmolkg,nitrogen_gkg,ph_iqr90,...,valid,source_version
10009,0-5,...
```

先用 folds 1/2/3 的土壤记录拟合每项标准化参数；以标准化后的六项 IQR90 平均值生成 depth-token confidence。该标度必须写入 manifest，fold4/fold5 只能套用，不能重算全局范围。

### 4.4 CA-HPI token 设计

新增 `PatchSoilPriorEncoder` 后，每个样本输出三个静态 soil token：

```text
numeric_values: [B,3,6] = 三个深度 x 六项土壤属性
time_values:    none
type_id:        soil
entity_id:      depth id (0-5 / 5-15 / 15-30 cm)
mask/confidence:[B,3]
```

土壤 token 没有时间维，但所有月份的视觉 token 都可查询它。CA-HPI 应自行学习“哪一层 Galileo 特征、哪个时间、哪个空间位置需要土壤背景”。

### 4.5 M3 禁止项

- 不将 SoilGrids 当作地块实测，不忽略其 250 m 分辨率和预测不确定性。
- 不从 PASTIS mask 或同期 RPG 作物登记反推土壤或填补土壤缺失。
- 不用包含 PASTIS 目标作物类别的地块级数据库作为土壤先验。
- 不在训练中在线调用不稳定 REST API；应以 WCS 下载法国四个区域子集后本地固定。

## 5. 推荐实验顺序

| 实验 | 视觉基线 | 先验 token | 用途 |
| --- | --- | --- | --- |
| B0 | 3D-Aware DPT | 无 | 固定基线 |
| M1 | B0 + CA-HPI | 物候 | 正在运行的首轮 |
| M2-only | B0 + CA-HPI | 气象 | 检查气象是否独立有效 |
| M3-only | B0 + CA-HPI | 土壤 | 检查土壤是否独立有效 |
| M1+M2 | B0 + CA-HPI | 物候 + 气象 | 验证区域化物候解释 |
| M1+M3 | B0 + CA-HPI | 物候 + 土壤 | 验证生态位补充 |
| M1+M2+M3 | B0 + CA-HPI | 三类 token | 最终异构组合 |

每一步只依据 fold4 选择 checkpoint 和是否继续；fold5 最多运行一次。对于 M2/M3，除无先验对照外，至少增加一次“样本间置乱但边际分布不变”的 token 对照，判断增益是否来自资料内容，而非额外参数量。

## 6. 实施清单

- [x] 写 `patch_id -> geometry/centroid` 的只读 metadata lookup：`scripts/export_pastis_prior_locations.py`。
- [x] 下载并冻结 ERA5-Land 2018-10 至 2019-09 的月度子集：`29196 = 2433 patch x 12 month` 条有效记录；每个 patch 固定选取最近的完整陆地网格。
- [x] 下载并冻结 SoilGrids 六个属性、三个深度、Q0.05/Q0.95 的四区域子集与版本信息：M3-center-v1 已生成 216 个原始 GeoTIFF、7299 条 patch-depth 记录与 SHA-256 manifest。
- [x] 提供 keyed CSV 校验、冻结和 train-fold 标准化工具：`scripts/prepare_environment_prior_tables.py`；不修改原有 temporal cache。
- [x] 实现 `PatchClimatePriorEncoder` 与 `PatchSoilPriorEncoder`，统一输出 `PriorBatch`。
- [x] 将多个 token encoder 合并为一个 mask-aware prior set，保持现有 CA-HPI fusion 不变。
- [x] 补齐 M2/M3 与旧 M1 的单元测试，并用真实冻结资料验证 M2 `[B,12,128]`、M3 `[B,3,128]` 与 M1+M2+M3 `[B,243,128]`。

### 6.1 已实现的文件接口

先从 PASTIS 的 `metadata.geojson` 导出每个 patch 的中心点。原始 geometry 是 Lambert-93（EPSG:2154）；脚本会转换为 ERA5-Land 与 SoilGrids 常用的 WGS84 经度/纬度：

```powershell
conda run -n presl python -B scripts/export_pastis_prior_locations.py `
  --metadata data/PASTIS/metadata.geojson `
  --output data/priors/pastis_patch_locations_v1.csv
```

在离线下载/提取程序把数值转换到本文件第 3、4 节规定单位后，也可以分别准备两张输入表。气象输入需要 `patch_id,month,t2m_c,tp_mm,ssrd_mj_m2,swvl1`；土壤输入需要 `patch_id,depth_cm,ph,soc_gkg,clay_pct,sand_pct,cec_cmolkg,nitrogen_gkg`。`valid` 和 `confidence` 可选；没有气象 confidence 时默认为 1。土壤若不提供 confidence，必须额外提供每个字段的 `*_q05`、`*_q95`，脚本才会基于 **train folds** 的标准差计算 confidence。该手工接口保留给其他来源；本项目默认使用 6.3 的自动化入口。

```powershell
conda run -n presl python -B scripts/prepare_environment_prior_tables.py `
  --metadata data/PASTIS/metadata.geojson `
  --climate-input data/priors/era5_land_pastis_cropyear_raw.csv `
  --climate-output data/priors/era5_land_pastis_cropyear_v1.csv `
  --climate-stats-output data/priors/era5_land_pastis_cropyear_v1_stats.json `
  --soil-input data/priors/soilgrids_pastis_surface_raw.csv `
  --soil-output data/priors/soilgrids_pastis_surface_v1.csv `
  --soil-stats-output data/priors/soilgrids_pastis_surface_v1_stats.json
```

脚本默认要求每个 PASTIS patch 都有完整的 12 个月气象记录和 3 个土层记录，并且只用 folds 1/2/3 计算 `mean/std`。确有刻意缺失时才加 `--allow-incomplete-patches`；训练配置中的 `allow_missing_patch: false` 仍会阻止“整个 patch 没有先验表记录”的静默错误。

### 6.2 训练 overlay

三份新 overlay 位于 `configs/prior_injection/`：

| 文件 | token 集合 | 每个样本的 token 数 |
| --- | --- | --- |
| `ca_hpi_m2_climate.yaml` | M2 ERA5-Land | 12 |
| `ca_hpi_m3_soil.yaml` | M3 SoilGrids | 3 |
| `ca_hpi_m1_m2_m3.yaml` | M1 物候 + M2 + M3 | 243 (= 228 + 12 + 3) |

它们都通过现有 `--prior-config` 接入任意保留 temporal feature pyramid 的 decoder；原有 `ca_hpi_structured.yaml` 仍是单一 M1 配置，命令和结果不受影响。

### 6.3 拉取仓库后的直接训练

仓库已包含训练时实际读取的 M2/M3 冻结表、train-fold 统计 JSON 与 patch 中心点审计表；不包含可再生的原始 CSV、ERA5 NetCDF 和 SoilGrids GeoTIFF。合作人 `git pull` 后，只要本机已有 `temporal_v2` 的 train/val 特征缓存，即可直接运行以下两个独立对照实验：

```powershell
# M2-only：ERA5-Land 月度气象先验
conda run -n presl python -u -B scripts/train_cached.py --config configs/galileo_3d_aware_dpt.yaml --prior-config configs/prior_injection/ca_hpi_m2_climate.yaml --cache-format temporal_v2 --temporal-dtype float16

# M3-only：SoilGrids 三层静态土壤先验
conda run -n presl python -u -B scripts/train_cached.py --config configs/galileo_3d_aware_dpt.yaml --prior-config configs/prior_injection/ca_hpi_m3_soil.yaml --cache-format temporal_v2 --temporal-dtype float16
```

两条命令的 decoder、优化器、训练轮数和缓存均相同；仅 prior overlay 不同。若本机尚无 `data/cache/galileo-base-patch8/monthly12_tile64_patch4_hl3-6-9-12_temporal-v2_tfp16_train/` 与对应 val 目录，先按项目 README 生成或复制共享缓存；Git 不上传特征缓存。

### 6.4 一键下载与冻结

`scripts/fetch_environment_priors.py` 会自动执行位置导出、ERA5-Land 下载与最近网格采样、SoilGrids WCS 分区下载与 patch 中心点采样、单位转换、raw manifest 写入及最终冻结。原始 NetCDF/GeoTIFF 位于被 Git 忽略的 `data/priors/raw/`；最终训练表和统计 JSON 位于 `data/priors/`。

首次运行前仅需人工完成 CDS 的授权边界：注册/登录 CDS、在 ERA5-Land 页面接受条款，并在用户目录创建 `.cdsapirc`（不要写入仓库）：

```text
url: https://cds.climate.copernicus.eu/api
key: <你的个人 access token>
```

安装一次依赖后，完整命令是：

```powershell
conda run -n presl python -m pip install -r requirements.txt
conda run -n presl python -u -B scripts/fetch_environment_priors.py
```

中途断网或 CDS 排队后直接重复第二条命令即可：已存在的 NetCDF/GeoTIFF/raw 表会被复用。`--overwrite` 才会重建 CSV/JSON；`--redownload` 才应在确认要重新请求外部原始数据时使用。若只需单独重建 M3-center-v1，可用 `--skip-era5 --skip-freeze` 下载 raw 表后，再调用本节 6.1 的 soil prepare 命令。

## 7. 资料结论

M2 气象与 M3 土壤都满足“推理时可获得、与 test 标签独立、能保留来源和不确定性”的基本条件。两者都不能直接塞进当前 YAML；当前优先完成资料冻结和 adapter，再开始任何训练。逐地块 RPG/LPIS 作物登记数据与 PASTIS 标签同源，不能作为本 benchmark 的输入先验。
