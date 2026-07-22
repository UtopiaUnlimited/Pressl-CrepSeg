# 结项汇报 PPT QA 摘要

检查对象：`PROJECT_CONCLUSION_DECK_2026-07-22.pptx`

- 页面数：20；页面顺序与 `deck_workspace/outline.json` 一致。
- 演讲者备注：20/20 页均已写入中文逐页讲稿；PPT 内备注文本与 `outline.json` 的 `notes` 字段逐字一致。
- 字体：所有可编辑文字统一为 `Microsoft YaHei`，不再依赖 `Helvetica Neue` 的中文回退。
- 编码：未发现 Unicode replacement character、控制字符或常见 mojibake 字符串。
- 兼容字符：不稳定的 Unicode 下标及希腊字母公式已改为 ASCII 兼容写法。
- 图表：没有柱状图；结果页全部使用表格，19 类 IoU 分为三页。
- 结构检查：溢出 0、重叠 0、占位符 0、几何违规 0、留白警告 0、视觉警告 0。
- 人工检查：使用本机 PowerPoint 将 20 页逐页导出为 1600×900 PNG，并复核接触表、架构图、结果表和结论页；未发现乱码、裁切或低对比度文字。

完整的机器检查结果保存在本地工作区 `deck_workspace/build/qa_fontfix/qa.json`，该构建目录不提交到 Git。
