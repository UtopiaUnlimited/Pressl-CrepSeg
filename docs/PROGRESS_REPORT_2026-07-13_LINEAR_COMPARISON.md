# 阶段成果报告：线性 Head 与空间 Decoder 对比

日期：2026-07-13

分支：`feat/paper-aligned-pastis-input`

任务：PASTIS 19 类作物语义分割

## 1. 本阶段目标

本阶段在已经对齐的 PASTIS 输入、同一份 frozen Galileo 特征和同一 fold 划分下，增加线性分割 head，并重新使用最高 fold4 `val_mIoU` checkpoint 测试已有模型。实验用于分开回答两个问题：

1. Galileo 最终层特征仅经过线性映射时能达到什么水平？
2. 保持训练策略相同后，卷积空间细化和多层特征融合能带来多少结构收益？

本文区分两种线性实验：

- **论文协议线性探测**：AdamW、CE、50 epoch、warmup cosine，配置为 `galileo_linear_probe.yaml`。
- **同协议线性 head**：结构仍只有一个 Linear，但 batch、loss、optimizer、最大 epoch 和早停与 decoder 实验一致，配置为 `galileo_linear_decoder_shared.yaml`。

第二种才是本文与最终层卷积、多层同尺度融合进行结构对比的主要基线。

## 2. 核心结果

### 2.1 核心优化配置一致的结构对比

三种模型使用相同共享缓存、CE + Dice、Prodigy、seed 42，并统一评估最高 `val_mIoU` checkpoint。线性 head 的配置已与另外两个 decoder 的当前配置对齐；历史重跑时间早于早停代码加入，实际训练轮数差异见第 5 节。

| 方法 | 可训练参数 | 最佳 val mIoU | 最佳 epoch | Test loss | Test mIoU | 相对线性 head |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 同协议线性 head | 0.234M | 32.29% | 10 | 1.29389 | 32.09% | - |
| 最终层卷积 baseline | 4.925M | 47.33% | 25 | 1.03063 | 46.02% | +13.92 pp |
| 多层同尺度融合 baseline | 9.058M | **49.28%** | 22 | **1.02015** | **48.01%** | **+15.91 pp** |

主要结论：

1. 仅统一训练策略不能让线性 head 接近卷积 decoder。最终层卷积在相同 Galileo final feature 上提高 13.92 个百分点，说明空间细化和非线性建模是主要性能来源。
2. 多层同尺度融合比最终层卷积进一步提高 1.99 个百分点，说明 Galileo 中间层对分割有额外价值。
3. 多层模型参数量约为线性 head 的 38.7 倍、最终层卷积的 1.84 倍，因此 1.99 pp 增益仍包含容量增加的影响。
4. 三个模型的 fold5 mIoU 均略低于最佳 fold4 mIoU，差值分别为 0.20、1.31 和 1.28 pp，方向正常，没有出现依靠 test 选模型的情况。

### 2.2 论文参考与单次论文协议检查

| 项目 | 训练/选择方式 | Test mIoU |
| --- | --- | ---: |
| Galileo 论文 Table 17 ViT-Base | 论文完整 linear probing protocol | 39.20% |
| 本项目论文协议线性探测检查 | `lr=1e-3`、seed 42、固定 50 epoch，使用 `last.pt` | 25.76% |
| 本项目同协议线性 head | Prodigy + CE/Dice + early stopping，使用最高 val mIoU | 32.09% |

这里的 25.76% **不是论文 39.2% 的完整复现结果**。当前只运行了一个学习率和一个 seed；论文还需要在验证集搜索 16 个学习率并运行 5 次。因此不能据此判断 Galileo 论文结果无法复现。

同一线性结构从论文协议单次检查的 25.76% 提高到统一 decoder 训练协议的 32.09%，说明训练策略影响显著；但两者 loss、optimizer、batch size、checkpoint 规则都不同，不能把 6.33 pp 全部归因于某一个超参数。

## 3. 固定数据与特征协议

| 项目 | 设置 |
| --- | --- |
| Train | folds 1/2/3，5820 个 64×64 tile |
| Validation | fold 4，1928 个 tile |
| Test | fold 5，1984 个 tile |
| 时间处理 | 2018-10 至 2019-09，按月聚合为 T=12 |
| Galileo | ViT-Base，patch size 4，完全冻结 |
| 最终层特征 | `[B, 768, 16, 16]` |
| 多层特征 | layers 3/6/9/12，均为 `[B, 768, 16, 16]` |
| 输出 | `[B, 19, 64, 64]` |
| Void label | 原始 19 映射为 -1，loss 和 mIoU 均忽略 |
| 归一化 | `galileo_norm_no_clip`，std multiplier 2.0 |

三种公平对比模型读取同一目录中的缓存，不重新运行 encoder：

```text
data/cache/galileo-base-patch8/monthly12_tile64_patch4_hl3-6-9-12_{train,val,test}/
```

## 4. 模型结构

### 4.1 同协议线性 Head

```text
[B, 768, 16, 16]
  -> per-patch Linear(768, 19 * 4 * 4)
  -> patch pixel rearrange
  -> [B, 19, 64, 64]
```

该模型只有 233,776 个可训练参数，没有卷积、激活函数、残差块或跨层融合。

### 4.2 最终层卷积 Baseline

```text
[B, 768, 16, 16]
  -> 1x1 projection
  -> 3 residual convolution blocks
  -> bilinear upsampling + smoothing
  -> [B, 19, 64, 64]
```

可训练参数为 4,924,691。它与线性 head 读取完全相同的 final `features`，因此二者差值主要反映 decoder 结构与容量。

### 4.3 多层同尺度融合 Baseline

```text
layers 3/6/9/12, each [B, 768, 16, 16]
  -> independent projection
  -> deep-to-shallow additive residual fusion
  -> residual convolution + upsampling
  -> [B, 19, 64, 64]
```

可训练参数为 9,058,067。所有输入层仍为 16×16，因此该模型是同尺度多层融合，而不是完整 DPT 多尺度 reassemble。

## 5. 训练与模型选择

当前三份对比配置的统一设置如下：

| 参数 | 值 |
| --- | --- |
| Seed | 42 |
| Batch size | 16 |
| Loss | CE + 0.5 × Dice |
| Optimizer | Prodigy |
| 配置学习率 | 1.0 |
| Weight decay | 0.1，decoupled |
| Scheduler | none |
| AMP | false |
| 最大 epochs | 100 |
| 模型选择 | fold4 最高 `val_mIoU` |
| 早停 | patience 12，min_delta 0.001，start epoch 10 |

同协议线性 head 在 epoch 10 达到 32.29% val mIoU，并于 epoch 22 触发早停。这说明早停机制已经实际生效，避免了后续无意义训练。

最终层卷积和多层融合本次使用已有重跑日志，各记录到 epoch 70；最高 val mIoU 分别出现在 epoch 25 和 epoch 22。两次重跑发生在早停功能加入之前，因此没有由当前早停器终止；不过它们保存了 `best_val_miou.pt`，本报告也不再沿用旧报告中按最低 val loss 选择的权重。这个历史执行差异应在多 seed 正式实验中消除。

## 6. Fold5 逐类 IoU

| Class ID | 线性 head | 最终层卷积 | 多层融合 | 多层 - 线性 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 68.92% | 72.42% | 72.69% | +3.77 pp |
| 1 | 56.43% | 60.11% | 60.49% | +4.06 pp |
| 2 | 62.40% | 73.82% | 75.32% | +12.91 pp |
| 3 | 62.35% | 75.38% | 75.77% | +13.42 pp |
| 4 | 17.29% | 46.28% | 50.23% | +32.94 pp |
| 5 | 69.12% | 81.39% | 81.48% | +12.36 pp |
| 6 | 13.34% | 32.31% | 35.87% | +22.52 pp |
| 7 | 9.77% | 38.42% | 41.32% | +31.55 pp |
| 8 | 43.54% | 51.27% | 52.90% | +9.37 pp |
| 9 | 51.71% | 71.24% | 74.83% | +23.11 pp |
| 10 | 7.95% | 12.28% | 21.11% | +13.16 pp |
| 11 | 41.89% | 49.09% | 50.64% | +8.75 pp |
| 12 | 18.21% | 24.77% | 24.83% | +6.61 pp |
| 13 | 18.28% | 34.04% | 34.00% | +15.71 pp |
| 14 | 16.58% | 25.78% | 26.32% | +9.74 pp |
| 15 | 30.74% | 53.66% | 52.80% | +22.06 pp |
| 16 | 18.74% | 38.30% | 40.05% | +21.31 pp |
| 17 | 1.18% | 19.15% | 26.39% | +25.21 pp |
| 18 | 1.33% | 14.63% | 15.06% | +13.74 pp |

线性 head 对 class 17 和 18 几乎无法识别，而空间 decoder 对多个困难类别有大幅提升。多层融合相对最终层卷积的明显收益集中在 class 10、17、4、9 和 6；class 15 略有下降，说明中间层特征并非对所有类别都等量有益。

## 7. 对当前研究问题的解释

1. **Galileo frozen feature 具有可迁移信息，但线性可分性有限。** 同协议线性 head 达到 32.09%，说明最终层特征包含作物类别信息，但仅靠逐 patch 线性分类不足以恢复细粒度空间边界和困难类别。
2. **decoder 设计是当前性能提升的核心变量。** 最终层卷积在不改变 encoder 和输入的情况下达到 46.02%，比线性 head 高 13.92 pp。
3. **多层表征继续带来收益。** 多层同尺度融合达到 48.01%，比最终层卷积高 1.99 pp，支持后续研究更完整的跨层和多尺度融合。
4. **不能把当前多层模型称为完整 DPT。** 它没有构造多分辨率特征金字塔，当前结论只支持“同尺度多层 Galileo 特征有效”。
5. **最高 val mIoU checkpoint 修正了旧报告的模型选择偏差。** 多层模型从旧的最低-loss checkpoint test 46.32% 提升到最高-mIoU checkpoint test 48.01%。

## 8. 局限性

- 所有本项目结果目前只有 seed 42，尚不能报告均值、标准差或显著性。
- 论文协议线性探测只完成固定 `lr=1e-3` 的单次运行，完整 16 学习率 × 5 runs sweep 尚未执行。
- 线性、卷积和多层模型参数量差距很大；当前结果证明完整下游方案更强，不等于单位参数效率更高。
- fold5 已用于阶段性最终报告，后续不能根据这些 test 结果反复修改超参数。
- 当前逐类表仅使用 class ID，尚未结合官方类别名称、像素频率和混淆矩阵。

## 9. 下一步建议

1. 完成论文线性探测的 16 学习率 × 5 runs sweep，得到可与论文 39.2% 正式比较的均值与方差。
2. 对线性 head、最终层卷积和多层融合至少补充 3 个相同 seed，统一报告 mean ± std。
3. 加入参数量匹配的轻量卷积 decoder，判断收益来自空间归纳偏置还是单纯容量增加。
4. 训练已经实现的 Galileo-Adapted 2D DPT、UPerNet-style 与 3D-Aware DPT，保持同一 fold、输入和 checkpoint 规则。
5. 增加混淆矩阵、类别频率和典型 tile 可视化，重点分析 class 4、7、10、17、18。

## 10. 复现命令

同协议线性 head：

```powershell
conda run -n presl python -B scripts/eval_cached.py --config configs/galileo_linear_decoder_shared.yaml --checkpoint checkpoints/galileo_linear_decoder_shared_paper_input_bs16_seed42_cached/best_val_miou.pt --split test
```

最终层卷积 baseline：

```powershell
conda run -n presl python -B scripts/eval_cached.py --config configs/galileo_single_layer_dpt_shared.yaml --checkpoint checkpoints/galileo_single_layer_dpt_shared_paper_input_bs16_rerun_seed42_cached/best_val_miou.pt --split test
```

多层同尺度融合 baseline：

```powershell
conda run -n presl python -B scripts/eval_cached.py --config configs/galileo_multi_layer_dpt_shared.yaml --checkpoint checkpoints/galileo_multi_layer_dpt_shared_paper_input_bs16_rerun_seed42_cached/best_val_miou.pt --split test
```

论文协议固定学习率检查：

```powershell
conda run -n presl python -B scripts/eval_cached.py --config configs/galileo_linear_probe.yaml --checkpoint checkpoints/galileo_linear_probe_lr1e-3_seed42_cached/last.pt --split test
```

本报告原始测试输出：

```text
paper-protocol linear (lr=1e-3, last): test_loss=0.92148, test_mIoU=25.758%
same-protocol linear (best val mIoU):  test_loss=1.29389, test_mIoU=32.093%
final-layer convolution:              test_loss=1.03063, test_mIoU=46.017%
multi-layer same-resolution fusion:   test_loss=1.02015, test_mIoU=48.005%
```
