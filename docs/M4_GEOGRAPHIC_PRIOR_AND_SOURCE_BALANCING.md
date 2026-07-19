# M4 地理上下文先验与多源平衡 CA-HPI

最后更新：2026-07-19

## 1. 新增信息源

M4 使用 PASTIS 官方 `metadata.geojson` 中每个 patch 的几何中心，经
`scripts/export_pastis_prior_locations.py` 转换为 WGS84 后得到：

```text
patch_id -> [longitude, latitude]
```

它是推理时可获得、与分割标签独立的静态 metadata。经纬度不直接代表作物类别，
但能够补充区域气候、种植制度和物候时移背景。M4 每个样本只生成 1 个 token：

```text
numeric_values: [B,1,2] = train-fold standardized [lon, lat]
mask/confidence:[B,1]
type_id:        geography source id
time_values:    none
```

标准化参数只由 folds 1/2/3 的 1455 个 patch 拟合，冻结在
`data/priors/pastis_patch_locations_v1_stats.json`。可用以下命令重建：

```powershell
conda run -n presl python -B scripts/prepare_patch_numeric_prior.py `
  --input data/priors/pastis_patch_locations_v1.csv `
  --output data/priors/pastis_patch_locations_v1_stats.json `
  --features lon lat `
  --train-folds 1 2 3 `
  --source-name "PASTIS metadata.geojson centroid coordinates"
```

`patch_numeric_table` 是通用接口，并不写死经纬度。未来只要提供
`patch_id + numeric features` 表和 train-fold 统计 JSON，也可以接入高程、坡度、
地形湿度指数或其他推理时可得的静态环境摘要。

## 2. 为什么需要多源平衡

原始多源实现把所有 token 直接拼接后做一次 softmax。M1/M2/M3/M4 的 token 数分别
是 228/12/3/1；当 query-key 分数相近时，来源得到的总注意力质量近似与 token 数成
正比，静态小来源会在学习开始前就处于明显劣势。

新增配置项：

```yaml
prior_injection:
  fusion:
    source_balance_bias_scale: 1.0
```

对来源 `s` 的每个有效 token 增加：

```text
bias_s = -source_balance_bias_scale * log(number_of_valid_tokens_in_source_s)
```

当该值为 1、内容分数与 confidence 相同时，每个有效来源在 softmax 前拥有相同的
总基准质量；来源内部仍由内容、置信度和 mask 决定具体 token 权重。默认值为 0，
所以旧配置与旧 checkpoint 行为不变。

诊断输出在多源场景额外包含：

```text
layer_k/<source_name>/attention_mass
layer_k/<source_name>/valid_token_fraction
```

来源名来自 overlay 中每个 `sources` 项的 `name`，用于检查模型是否真正使用新增来源。

## 3. 受控实验入口

| 配置 | token | 用途 |
| --- | ---: | --- |
| `ca_hpi_m4_geography.yaml` | 1 | M4-only，检查地理 metadata 的独立贡献 |
| `ca_hpi_m1_m2_m3_balanced.yaml` | 243 | 不含 M4 的 source-balanced 匹配基线 |
| `ca_hpi_m1_m2_m3_m4.yaml` | 244 | 在相同 source-balanced 机制下加入 M4 |

训练示例：

```powershell
conda run -n presl python -u -B scripts/train_cached.py `
  --config configs/galileo_3d_aware_dpt.yaml `
  --prior-config configs/prior_injection/ca_hpi_m4_geography.yaml `
  --cache-format temporal_v2 `
  --temporal-dtype float16
```

比较 M4 增益时，应优先比较两份 balanced 组合配置，不能直接把
`ca_hpi_m1_m2_m3_m4.yaml` 与旧的非平衡 `ca_hpi_m1_m2_m3.yaml` 相减，否则无法区分
增益来自地理信息还是注意力校正。

## 4. 科研边界

- 经纬度可能利用区域共现规律，因此除现有随机 fold 协议外，后续应补 tile/区域留出评估。
- 不允许根据 fold4/fold5 标签调整坐标、分区或 confidence。
- M4 是静态区域 context，不应被解释成地块级实测农学属性。
- 若 M4-only 提升明显但区域留出性能下降，应报告为空间捷径而非通用先验增益。
