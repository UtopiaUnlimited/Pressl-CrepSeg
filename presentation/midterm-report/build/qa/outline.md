# Deck Outline

- File: D:\ycy\GISproject\PreSSL-CropSeg\presentation\midterm-report\build\galileo.pptx
- Slides: 14

## Slide 1: (No title)

Body:
- RESEARCH MIDTERM
- 冻结 Galileo 表征上的
- 作物语义分割解码器研究
- PASTIS 中期汇报｜固定遥感预训练编码器，比较像素级解码方式

## Slide 2: (No title)

Body:
- 研究问题：表征已有，如何有效读出？
- 从“跑通模型”转向可检验的 decoder 假设
- 现实瓶颈
- 像素级标注昂贵，遥感预训练具有现实价值
- Transformer token 不能直接等同于分割图
- 最终层语义强，但局部边界可能不足
- 核心假设
- Galileo 冻结表征包含可迁移的作物信息
- 空间细化 decoder 可改善线性读取
- 跨层融合可能补充局部细节与深层语义
- RQ1：空间 decoder 是否改善 linear probe？　RQ2：多层特征是否优于最终层？
- 2/14

## Slide 3: (No title)

Body:
- 任务基础：PASTIS 多时相作物语义分割
- 18 种作物 + 背景；void 像素不参与训练与评价
- 2,433
- 原始空间 patch
- 每个 128×128，10 m/px
- 10
- Sentinel-2 波段
- 多光谱时间序列
- 19
- 有效语义类别
- 背景 + 18 种作物
- 5
- 官方 folds
- 当前使用 1/2/3 → 4 → 5
- Sources: [PASTIS21]
- 3/14

## Slide 4: (No title)

Body:
- 理论依据：四篇论文构成研究链条
- 文献给出研究动机；本项目仍需通过受控实验检验具体假设。
- Sources: [PASTIS21] [SeCo21] [Galileo25] [DPT21]
- 4/14

## Slide 5: (No title)

Body:
- 总体技术路线：固定 encoder，只比较 decoder
- 共享输入与缓存，避免把协议变化误认为结构收益
- 01
- PASTIS 输入
- 12 月 × 10 波段 × 64×64
- 02
- 冻结 Galileo
- ViT-Base；patch size 4
- 03
- 共享缓存
- final + layers 3/6/9/12
- 04
- Decoder 对比
- DPT / UPerNet → 19 类
- Sources: [ProjectProtocol]
- 5/14

## Slide 6: (No title)

Body:
- 输入协议与关键工程修正
- 从原始不规则时序到 Galileo 可复现空间特征
- T×10×128²
- 原始 PASTIS
- 保留原生 10 波段
- 12×10×128²
- 月度聚合
- 2018-10 至 2019-09
- 12×10×64²
- 空间切块
- 每个 patch 生成 4 个 tile
- 768×16²
- 结构化 token 聚合
- 按 H/W/T/group 恢复空间网格
- Sources: [Galileo25] [ProjectProtocol]
- 6/14

## Slide 7: (No title)

Body:
- 三条 decoder 路线
- 所有路线读取同一批 Galileo 共享缓存。
- Sources: [DPT21] [ProjectProtocol]
- 7/14

## Slide 8: (No title)

Body:
- 实验设置：公平比较与本地可复现
- 共同训练设置
- 资源与模型规模
- 多层模型参数量约为单层的 1.84 倍；当前结构比较仍混入容量差异。
- Sources: [Run2026]
- 8/14

## Slide 9: (No title)

Body:
- 阶段结果：方案有效，但绝对性能仍不高
- Fold5 test mIoU；论文 39.2% 为 linear probe 参考，不是同容量比较
- 6.58
- pp · Single vs 论文参考
- 完整下游方案差异
- 0.54
- pp · Multi vs Single
- 单 seed 的微弱正向信号
- 46.32%
- 当前最高 test mIoU
- 仍有明显提升空间
- Sources: [Galileo25] [PASTIS21] [Run2026]
- 9/14

## Slide 10: (No title)

Body:
- 训练诊断：过拟合与 checkpoint 规则错位
- 最高 validation mIoU 与 epoch 100 对比
- 25 / 11
- 最低 val loss epoch
- Single / Multi
- 47 / 22
- 最高 val mIoU epoch
- 与保存规则不一致
- −6.80 pp
- Multi 后期回落
- 100 epochs 明显过长
- Sources: [Run2026]
- 10/14

## Slide 11: (No title)

Body:
- 阶段成果与当前证据边界
- 已经建立
- Galileo 官方 PASTIS 输入协议
- 正确 token 空间聚合与共享缓存
- 两种 DPT 的完整 train / val / test 链路
- UPerNet、TensorBoard 与定性可视化工具
- 尚不能下结论
- 46.32% 不是高性能结果
- 仅一个 seed，缺少均值与方差
- Multi 参数更多，未做容量控制
- 当前仅同分辨率投影后等权相加
- 当前结果不能代表完善 DPT 的上限；中期价值在于建立可信链路与可检验问题。
- Sources: [Run2026]
- 11/14

## Slide 12: (No title)

Body:
- 希望与老师讨论：课题下一步走多深？
- 建议先完成公平 decoder 对比，再由老师确定方法深化或多模态扩展。
- Sources: [PASTISR22] [ProjectPlan]
- 12/14

## Slide 13: (No title)

Body:
- 下一阶段：先做强证据，再扩展研究边界
- 近期
- 实验规范
- best mIoU；early stop；≥3 seeds；UPerNet
- 中期
- 受控矩阵
- 代表性 SSL × DPT / UPerNet；参数匹配
- 进阶
- 研究深化
- 学习式 DPT 或 PASTIS-R 的 S1+S2 融合
- Sources: [ProjectPlan]
- 13/14

## Slide 14: (No title)

Body:
- 参考文献与项目证据
- 所有结果数字均来自论文公开表格或项目已保存日志。
- Sources: [PASTIS21] [SeCo21] [DPT21] [Galileo25] [PASTISR22] [Run2026]
- 14/14
