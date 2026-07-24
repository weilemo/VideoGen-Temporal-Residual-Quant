# Head-Wise Quant 主线确认

日期：`2026-05-07`

## 本次目标

- 基于 `Self-Forcing + QVG` 当前实现，确认后续是否以这套代码作为 `head-wise quant` 的主框架。

## 已完成

- 复核 `Self-Forcing` BF16 baseline 共享盘日志：
  - `/mnt/users/moweile-20251213/workspace/videoquant/Quant-VideoGen/slurm_logs/qvg_sf_bf16_g-24046.out`
- 确认 BF16 baseline 成功生成两条视频：
  - `0-0_ema.mp4`
  - `1-0_ema.mp4`
- 确认共享盘 `/mnt/users/moweile-20251213/workspace/videoquant/Quant-VideoGen/` 是实验日志和结果的重要落点。
- 基于代码结构分析，确认当前 `Self-Forcing + QVG` 实现适合作为后续 `head-wise quant` 的主代码框架。

## 核心判断

- 当前量化入口集中在 `experiments/Self-Forcing/pipeline/causal_inference.py` 的 cache 管理层。
- KV cache 张量组织天然保留 `head` 维度，适合做 `per-head` 或 `head-group` 策略。
- 现有 KMeans / PRQ 路径已经按 `[B, H, S, D]` 语义处理张量，本质上具备逐 head 独立聚类基础。
- 当前真正的缺口不是数据路径，而是：
  - 全局单一 `quant_config`
  - 单一 `quant_type`
  - 缺少 mixed head policy 的压缩元数据与反量化逻辑

## 本次产出

- 新增正式文档：
  - `/data2/moweile-20251213/workspace/videoquant/Quant-VideoGen/docs/headwise_quant_feasibility.md`

## 结论

- 后续主线明确切换为：基于 `Self-Forcing + QVG` 推进 `head-wise quant`。
- 第一版实现优先做 `head-group mixed precision`，不直接做 12 个 head 全独立策略。
