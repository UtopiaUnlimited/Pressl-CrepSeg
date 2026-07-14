# 问题与解答

## PASTIS 的19个有效类别

PASTIS 的有效语义标签是 `0..18`：**1 个非农业背景类 + 18 个农业/作物类别**，并不是19种互相独立的植物。原始标签 `19` 是 void label，表示应忽略的区域，当前代码会把它映射为 `-1`，不参与 loss 和 mIoU。

| ID | 官方类别名 | 中文含义 | 类别说明 |
| ---: | --- | --- | --- |
| 0 | Background | 非农业背景 | 非农业用地，不是一种作物 |
| 1 | Meadow | 草地/牧草地 | 草本覆盖或牧草地，不限定为某一种植物 |
| 2 | Soft winter wheat | 软质冬小麦 | 秋冬播种、次年收获的软质小麦 |
| 3 | Corn | 玉米 | 玉米种植地 |
| 4 | Winter barley | 冬大麦 | 秋冬播种的大麦 |
| 5 | Winter rapeseed | 冬油菜 | 秋冬播种的油菜 |
| 6 | Spring barley | 春大麦 | 春季播种的大麦，与冬大麦分为不同类别 |
| 7 | Sunflower | 向日葵 | 向日葵种植地 |
| 8 | Grapevine | 葡萄园 | 葡萄种植地，属于多年生木本作物 |
| 9 | Beet | 甜菜 | 官方名称为 Beet，PASTIS 农业场景中通常指甜菜种植地 |
| 10 | Winter triticale | 冬小黑麦 | 秋冬播种的小黑麦，即小麦和黑麦的杂交谷物 |
| 11 | Winter durum wheat | 冬硬粒小麦 | 秋冬播种的硬粒小麦 |
| 12 | Fruits, vegetables, flowers | 水果、蔬菜和花卉 | 多种园艺作物合并成的综合类别，不对应单一植物 |
| 13 | Potatoes | 马铃薯 | 马铃薯种植地 |
| 14 | Leguminous fodder | 豆科饲料作物 | 用作饲料的豆科作物集合，不对应单一植物 |
| 15 | Soybeans | 大豆 | 大豆种植地 |
| 16 | Orchard | 果园 | 多年生果树种植区域，不限定具体果树品种 |
| 17 | Mixed cereal | 混合谷物 | 多种谷物混合种植，无法归入单一谷物类别 |
| 18 | Sorghum | 高粱 | 高粱种植地 |

数据集标签定义来源于 [PASTIS 官方 benchmark](https://github.com/VSainteuf/pastis-benchmark)，逐项编号可对照 [TorchGeo 的 PASTIS 类别映射](https://docs.torchgeo.org/en/stable/_modules/torchgeo/datasets/pastis.html)；代码中的统一映射维护在 `data/pastis.py` 的 `PASTIS_CLASS_NAMES`。

19 个有效标签与法国物候资料并不能全部一一对应。类别映射等级、四个数据区域和首批可靠来源候选见 [`PHENOLOGY_PRIOR_SOURCES.md`](PHENOLOGY_PRIOR_SOURCES.md)，原始协作表见 [`data/priors/pastis_france_phenology_sources.csv`](../data/priors/pastis_france_phenology_sources.csv)。

## 后续实验问题

- 数据输入的维度问题
    - Galileo 在 encoder 内已经对多时相 token 做了上下文建模
    - 前四种
        - 实验沿用官方 probing 接口，在 encoder 输出端对上下文化后的逐月特征求均值；这保留了一部分时序上下文，但 decoder 无法再显式区分月份
    - 第五种
        - 正是保留 T 维并在 decoder 中继续建模时间，用来检验这种压缩是否限制了分割性能
    - 后续添加一（可以当成正常实验的消融）
        - 在实验五基础上把图像改为单张输入，即时间维度全部放到 decoder 处理
    - 后继续添加二
        - 在实验五的基础上看下面的先验知识注入，这样效果可能更好，前面的几类都放着了，当记录对比用
    - 关于特征缓存
        - 保留已有 `spatial_v1` 缓存和旧命令，方案一至四可以马上继续训练
        - 新增 `temporal_v2` 缓存，只保存四层 `[L,T,D,H,W]`，不重复保存时间均值后的特征
        - 方案一至四读取新缓存时临时沿 T 求均值，方案五直接使用完整 T
    - 关于 decoder 空间重采样
        - 方案一、二保持 Galileo 原生 `16x16` 网格到最终上采样；方案三 PPM 保留未池化原始分支
        - 方案四、五的 `8x8` 只承担全局上下文计算，同时保留 final `16x16` 原尺度旁路并注入 `16x16` 融合级
        - 该旁路由 `preserve_native_deep_skip: true` 控制，不需要重新生成缓存
- SSL 模型的调查与选择
    - 目前 Galileo 已经算很好的了，但是还是可以做调查，选好了的话权重在 https://huggingface.co/BiliSakura/ 里面寻找，有很多
- 训练曲线的问题
    - 后期有点过拟合，不知道是不是 loss 设计的问题
    - 加上 Acc F1 指标，最后作图自己画，tensorboard 尺度有问题，会显得很上升
- 先验知识注入
    - 已将助教提出的“法国作物周期检索、Meta Data Encoder 旁路、Decision Fusion、月份筛选”整理为独立方案：[`PHENOLOGY_PRIOR_INJECTION_PLAN.md`](PHENOLOGY_PRIOR_INJECTION_PLAN.md)
    - 当前主路线：保留完整 `T=12`，将人工类别-月份物候矩阵作为软先验，通过旁路进行类别条件 Decision Fusion
    - 并行准备两种先验：外部农学资料 `P_ext` 与仅由 train folds `1/2/3` 统计的 `P_data`，先分别实验再组合
    - 月份筛选作为后续消融；测试时禁止按真实类别筛月份，优先用固定长度 temporal mask 处理
    - FiLM、AdaGN 和层级语义分割暂列扩展，不在第一版同时叠加
- 效果展示
    - 输入数据有几类，分别是什么植物类别，在 markdown 里面解释清楚
    - test 后按类别说 mIoU（这个我们有），因为类别不平衡，然后我们才要用先验，部分类别效果不好的可能会好一点
    - 改成可视化 test 全部图像的展示，因为总共也就几百张图，放在 output 里面
