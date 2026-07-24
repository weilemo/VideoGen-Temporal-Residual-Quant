# 2026-05-10 Naive head-wise quant 实验 + 文档同步

## 背景

- 已有三条实验线：BF16 baseline、QVG INT2 baseline (triton PRQ)、R-HWQ-4h (triton PRQ)
- 需要补充一条不使用 QVG PRQ 的对照实验：random headwise 分组 + 纯 blockwise int2/int4

## 执行

- 运行 `run_random_hwq.sh`，`HIGH_PRECISION_QUANT_TYPE=naive-int4`，`LOW_PRECISION_QUANT_TYPE=naive-int2`
- 第一次 OOM：VAE decode 时显存超 80G（naive 量化返回 bf16 张量，不压缩）
- 第二次加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 跑通
- 产出 2 条视频：`results/selfforcing/rhwq_seed_0_hi_4_naive-int4_lo_naive-int2_64/kc_256_vc_256_nstages_1/`
  - `0-0_ema.mp4`（25MB）
  - `1-0_ema.mp4`（36MB）

## 文件变更

- `STATUS.md`：新增 naive HWQ 实验结果，更新实验产物目录为 `results/`
- `HANDOFF.md`：更新四条实验线信息、显存注意事项
- `MEMORY.md`：新增 Self-Forcing 推理环境依赖

## 发现与注意事项

- naive blockwise 量化不产生显存压缩效果（返回 bf16），需 `expandable_segments:True`
- QVG triton PRQ 返回 int8 packed 格式，A100 80G 可直接跑
- `outputs/` 已废弃，实验结果统一放 `results/`
