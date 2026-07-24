# 2026-07-20 TRQ 在线分叉因果定位实验落地

- 计划来源：Obsidian `2026-07-20 TRQ 在线分叉因果定位与修复实验计划.md`。
- 分支：`codex/trq-online-causal-gates`。
- 实现范围：E0-E5 的分析、干预、quality/protection gate 和 launcher。
- 明确不做：E6 Triton decoder、decoded-span cache、fused attention。
- GPU 约束：2026-07-21 起 EPIC 只允许物理 GPU 2/3；launcher 默认 GPU 2，
  对其他编号直接报错。
- 本地验证：CPU 单测、Python 语法、shell `bash -n`、GPU guard dry-run。
- 集群下一步：通过已登录 Chrome code-server 拉取分支，先运行 E0/E1；人工
  failure tags 完整后再按 gate 决定 E2/E3 或 E5。
