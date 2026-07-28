# 快速交接

## 当前入口

```bash
cd /Users/moweile/Obsidian/Knowledge/Research/project/longvideo-kvcache-quant/code/videoquant-trq
```

当前仓库分工（2026-07-11）：

| 目录 | 用途 |
|---|---|
| `videoquant-trq/temporalresidualkvquant` | 当前方法主库，拥有 TRQ、PRQ、head-wise policy 和 Self-Forcing adapter |
| `videoquant-trq/integrations` | Causal、HY、LongCat、Rolling 的 TRQ 适配与启动入口 |
| `videoquant-trq/references/quant-videogen` | 原始 QVG 基线参考 |
| `../qvg` | 学弟的 S2++ 研究仓库，只作为迁移来源与对照 |

新同学先读 `temporalresidualkvquant/README.md` 和
`temporalresidualkvquant/docs/getting_started.md`；TRQ v1 的统一边界见
`temporalresidualkvquant/docs/trq_unification.md`。

## 当前接力点

- 三基线实验统一入口：`experiments/world_model_quant/README.md`。
  - MovieGen10 / HY dev 的自动评测已完成；INT2 上三条基线均显示 TRQ 的
    PSNR/SSIM/LPIPS 优于 naive，INT4 暂无一致优势；
  - Expansion B1 生成于 2026-07-28 完成：Causal 新增五档各 22 条；LongCat 新增
    22 个 BF16 prefix 与五档各 22 条 continuation；HY-WAN holdout 新增 100 条；
  - LongCat 六个扩展目录均为 `22/22, bad=0`，双 GPU lane 均 exit 0；已有视频由
    可解码检查跳过，没有重复生成；
  - Causal 的 `failed:validation` 是旧 10 条与新增 22 条分处不同 worktree 导致的索引
    缺口，不是生成失败。下一步先建立跨结果根的 MovieGen32 manifest；
  - 当前只完成 B1 **远端生成**，尚未完成 32 条扩展集的统一指标、人工盲审、跨 seed
    统计或正式 VBench 验收；这些完成前不启动 B2 MovieGen128；
  - 本地推送后，远端只运行一次 `pull_remote_once.sh`，不启动同步轮询。

- 2026-07-20 在线分叉计划已实现于 `codex/trq-online-causal-gates`：
  - 总协议：`temporalresidualkvquant/docs/online_causal_experiments.md`；
  - CPU E0：`scripts/analysis/reanalyze_existing_online_runs.sh`；
  - GPU E2/E3：`scripts/analysis/run_online_causal_diagnosis.sh`，只允许 GPU 2/3；
  - E4：`run_e4_candidates.sh` + `analyze_protection_gate.py`；
  - E5：`run_quality_gated_long_rollout.sh`，183/501/699 分阶段；VBench 数值仅报告，人工 catastrophe gate 未完成或失败时关闭。
- Self-Forcing E0/E1 自动指标已经完成；当前缺口是 K4V4 contact sheet 与人工 failure
  tags。该人工门完整前不能启动 E5，也不能把 VBench 接近 BF16 写成视觉无损。
- E6 不在本分支；只有质量 winner 冻结后才另开系统优化分支。

- `Self-Forcing` 八条实验线均已跑通，七条完成 VBench 评估（2-prompt），一条完成 32-prompt VBench 评估：
  - 2-prompt 结果位于 `temporalresidualkvquant/results/selfforcing/`：
    - `bf16/` — BF16 baseline (Final Score: 0.6486)
    - `triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1/` — QVG INT2 baseline (0.6469, ↓0.26%)
    - `rhwq_seed_0_hi_4_triton-nstages-kmeans-int4_lo_triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1/` — R-HWQ-4h PRQ (0.6416, ↓1.07%)
    - `rhwq_seed_0_hi_4_packed-naive-int8_lo_packed-naive-int4_64/kc_256_vc_256_nstages_1/` — R-HWQ-4h Packed int8+int4 (0.6479, ↓0.10%)
    - `rhwq_seed_0_hi_4_packed-naive-int4_lo_packed-naive-int2_64/kc_256_vc_256_nstages_1/` — R-HWQ-4h Packed int4+int2 (0.6279, ↓3.19%)
    - `topk_top4_dmd_loss_hi_4_packed-naive-int4_lo_packed-naive-int2_64/kc_256_vc_256_nstages_1/` — Top-K HWQ Packed int4+int2 (0.6303, ↓2.82%)
    - `topk_top4_dmd_loss_hi_4_packed-naive-int8_lo_packed-naive-int4_64/kc_256_vc_256_nstages_1/` — **Top-K HWQ Packed int8+int4 (0.7615, ↓0.23%, 32 prompts)** ← 新增
  - 32-prompt MovieGenVideoBench 结果：`results/selfforcing/vbench_eval_mb32/comparison_summary.json`
- VBench 评估完整结果：`temporalresidualkvquant/results/selfforcing/vbench_eval/comparison_summary.json`
- 评估脚本：`temporalresidualkvquant/scripts/eval/evaluate_experiments.sh` 和 `scripts/eval/aggregate_results.py`
- **Importance top-k policy 已就绪**：
  - 完整 per-layer top-4 head policy: `assets/head_importance/top4_dmd_loss.json`（30 layers × 12 heads, 360 scores）
  - 来源: `references/focused-forcing-code/focusedforcing_sf/dm_loss.json` → `scripts/aggregate_head_importance.py`
  - 已验证: Top-K HWQ 推理 + VBench 评估完成
- **head importance analysis 两阶段方案已验证**：
  - Smoke test 在 A100 80GB 上完整跑通：Phase 1 (inference, ~25.5 GB) + Phase 2 (scoring, ~1.2 GB)
  - 两阶段拆分彻底解决 OOM 问题
  - 启动脚本：`bash scripts/self_forcing/run_head_importance_analysis.sh`（PHASE=inference/scoring/all）

## 下一位 agent 先做什么

1. 先读 `docs/project/STATUS.md` 和 `experiments/world_model_quant/README.md`。
2. 不再启动 Expansion B1 生成；先定位旧 MovieGen10 与新增 22 条结果根，生成 Causal
   与 LongCat 的 MovieGen32 统一 manifest，并逐项做 `ffprobe` 完整性检查。
3. 对统一索引运行 PSNR、SSIM、LPIPS、八维 VBench-derived、prompt-level bootstrap
   95% CI 和逐 prompt 表；LongCat 保持跳过 13 个共享 conditioning frames。
4. 完成 Causal/LongCat failure tags、长时分段漂移和 HY holdout action/感知盲审。
5. 只有 B1 没有 TRQ-only catastrophe 且 INT2 效应方向未反转，才向用户申请启动 B2
   MovieGen128；不得根据生成 `done` 自动放行。
6. Self-Forcing 的独立阻塞仍是 K4V4 contact sheet / catastrophe gate；该门通过前不启动
   501/699-frame rollout。
7. 如涉及新 CUDA 任务，先重新核对当前 EPIC 租约与物理/逻辑 GPU 映射；旧 GPU 编号
   和旧 hostname 只能作为历史证据。

当前 B1 自动评测入口为 `experiments/world_model_quant/run_b1_evaluation.sh`。它要求显式
提供 Causal/LongCat 的 legacy 与 B1 结果根，以及 HY holdout 根；默认分配 GPU 6/7，
但启动前仍必须重新执行 `nvidia-smi` 并核对租约。索引器
`prepare_moviegen32_manifest.py` 会在任何 GPU 工作前 fail closed，因此不要用手工复制
视频绕过重复或缺失检查。

## 当前最重要信息

- 当前核心目标是 `forcing-based long video generation` 的 KV cache 量化。
- 重点质量维度：`identity consistency`、`scene consistency`、`motion continuity`。
- 当前优先路线：
  - 以 `temporalresidualkvquant` 为方法代码库推进 `head-wise quant`
  - 以 `Self-Forcing` 为实验集成入口
  - 从 `temporalresidualkvquant/scripts/self_forcing/` 启动实验
- 当前研究重点：
  - `importance metric`：定义 head 重要性分数（当前: DMD loss，Top-K vs Random = +0.38pp）
  - `importance collection`：离线 calibration / 在线估计 / 前几个 chunk 后固定
  - `policy granularity`：per-layer / per-chunk / sink-history-tail / K-V 分开 / prompt 自适应
- 当前量化方案 VBench 对比：

### 2-prompt 对比（180 frames, vbench_prompts.txt）

| 方案 | Final Score | vs BF16 | Peak VRAM | KV Cache |
|------|------------|---------|-----------|----------|
| BF16 Baseline | 0.6486 | — | ~80 GB * | ~78 GB * |
| Packed int8+int4 (random) | 0.6479 | ↓0.10% | ~40 GB ** | ~25 GB ** |
| QVG INT2 (PRQ) | 0.6469 | ↓0.26% | ~20 GB ** | ~12 GB ** |
| R-HWQ-4h PRQ (int4+int2) | 0.6416 | ↓1.07% | ~20 GB ** | ~12 GB ** |
| Top-K HWQ Packed (int4+int2) | 0.6303 | ↓2.82% | 26 GB | 12.4 GB |
| R-HWQ-4h Packed (int4+int2) | 0.6279 | ↓3.19% | ~26 GB ** | ~12 GB ** |
| R-HWQ-4h Naive (int4+int2) | 0.5954 | ↓8.19% | >76 GB * | >76 GB * |

### 32-prompt 对比（180 frames, MovieGenVideoBench 前 32 prompts）

| 方案 | Final Score | vs BF16 | Peak VRAM | KV Cache |
|------|------------|---------|-----------|----------|
| BF16 Baseline (32p) | 0.7633 | — | 61.9 GB | 48.2 GB |
| **int8+int4 组** |
| TK8 int8+int4 | 0.7616 | ↓0.23% | 37.4 GB | 23.4 GB |
| TK4 int8+int4 | 0.7615 | ↓0.25% | 33.6 GB | 19.7 GB |
| TK6 int8+int4 | 0.7609 | ↓0.31% | 35.4 GB | 22.1 GB |
| TK2 int8+int4 | 0.7609 | ↓0.32% | 31.7 GB | 18.2 GB |
| Rand4 int8+int4 | 0.7600 | ↓0.44% | 33.6 GB | 19.7 GB |
| **int4+int2 组** |
| TK8 int4+int2 | 0.7413 | ↓2.89% | 28.6 GB | 14.4 GB |
| TK6 int4+int2 | 0.7348 | ↓3.75% | 27.6 GB | 13.4 GB |
| TK4 int4+int2 | 0.7302 | ↓4.34% | 26.7 GB | 12.4 GB |
| Rand4 int4+int2 | 0.7268 | ↓4.79% | 26.7 GB | 12.4 GB |
| TK2 int4+int2 | 0.7236 | ↓5.20% | 25.8 GB | 11.5 GB |

**核心结论**：
- int8+int4：全部变体 <0.5% 退化，近乎无损；TK8 最优 ↓0.23%
- int4+int2：top-k 优势显著，TK8 ↓2.89% vs TK2 ↓5.20%；k 越大越好
- int8+int4 下 k 收益递减（k=2~8 几乎无差），int4+int2 下 k 持续增益

\* 据 STATUS.md / MEMORY.md 记录
\** 据量化类型推算（结果在共享盘生成）
32-prompt 所有显存数据均为实测

- 实验产物统一放到 `temporalresidualkvquant/results/`（不要放 `outputs/`）。
- 本机权重位于 `temporalresidualkvquant/ckpts/Self-Forcing/`（不进 git）。
- 运行环境：
  - Self-Forcing 推理：`conda activate forcing`
  - VBench 评估：`conda activate vbench`（注意：`source /mnt/workspace/caipeiliang/miniconda3/etc/profile.d/conda.sh`）
  - VBench 评估如遇 "Too many open files" 错误，需清理 split_clip 目录后重试：
    ```
    rm -rf results/selfforcing/<exp>/kc_256_vc_256_nstages_1/split_clip
    rm -rf results/selfforcing/<exp>/kc_256_vc_256_nstages_1/*_cat_firstframes_videos
    ```

## 如果马上开始新任务

- 先把本次任务写进 `STATUS.md`
- 如果开始代码阅读或实验准备，补一条 `logs/` 日志
- 展示结果时务必包含：显存数据（Peak VRAM / KV Cache）、帧数
- 如果实验结果或 Slurm 日志先落到 `/mnt/users/...` 运行副本，结束后默认同步一份回本地
