# 快速交接

## 当前接力点

- `Self-Forcing` 八条实验线均已跑通，七条完成 VBench 评估（2-prompt），一条完成 32-prompt VBench 评估：
  - 2-prompt 结果位于 `HeadWiseKVQuant/results/selfforcing/`：
    - `bf16/` — BF16 baseline (Final Score: 0.6486)
    - `triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1/` — QVG INT2 baseline (0.6469, ↓0.26%)
    - `rhwq_seed_0_hi_4_triton-nstages-kmeans-int4_lo_triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1/` — R-HWQ-4h PRQ (0.6416, ↓1.07%)
    - `rhwq_seed_0_hi_4_packed-naive-int8_lo_packed-naive-int4_64/kc_256_vc_256_nstages_1/` — R-HWQ-4h Packed int8+int4 (0.6479, ↓0.10%)
    - `rhwq_seed_0_hi_4_packed-naive-int4_lo_packed-naive-int2_64/kc_256_vc_256_nstages_1/` — R-HWQ-4h Packed int4+int2 (0.6279, ↓3.19%)
    - `topk_top4_dmd_loss_hi_4_packed-naive-int4_lo_packed-naive-int2_64/kc_256_vc_256_nstages_1/` — Top-K HWQ Packed int4+int2 (0.6303, ↓2.82%)
    - `topk_top4_dmd_loss_hi_4_packed-naive-int8_lo_packed-naive-int4_64/kc_256_vc_256_nstages_1/` — **Top-K HWQ Packed int8+int4 (0.7615, ↓0.23%, 32 prompts)** ← 新增
  - 32-prompt MovieGenVideoBench 结果：`results/selfforcing/vbench_eval_mb32/comparison_summary.json`
- VBench 评估完整结果：`HeadWiseKVQuant/results/selfforcing/vbench_eval/comparison_summary.json`
- 评估脚本：`HeadWiseKVQuant/scripts/eval/evaluate_experiments.sh` 和 `scripts/eval/aggregate_results.py`
- **Importance top-k policy 已就绪**：
  - 完整 per-layer top-4 head policy: `assets/head_importance/top4_dmd_loss.json`（30 layers × 12 heads, 360 scores）
  - 来源: `external/focused-forcing-code/focusedforcing_sf/dm_loss.json` → `scripts/aggregate_head_importance.py`
  - 已验证: Top-K HWQ 推理 + VBench 评估完成
- **head importance analysis 两阶段方案已验证**：
  - Smoke test 在 A100 80GB 上完整跑通：Phase 1 (inference, ~25.5 GB) + Phase 2 (scoring, ~1.2 GB)
  - 两阶段拆分彻底解决 OOM 问题
  - 启动脚本：`bash scripts/self_forcing/run_head_importance_analysis.sh`（PHASE=inference/scoring/all）

## 下一位 agent 先做什么

1. 先看 `STATUS.md`，再看 `MEMORY.md`
2. 查看 VBench 对比结果：
   - 2-prompt: `results/selfforcing/vbench_eval/comparison_summary.json`
   - **32-prompt**: `results/selfforcing/vbench_eval_mb32/comparison_summary.json` ← 新
3. 查看量化方案文档：`docs/quantization_approaches.md`
4. 再看独立库结构：
   - `HeadWiseKVQuant/README.md`
   - `HeadWiseKVQuant/docs/self_forcing_integration.md`
   - `HeadWiseKVQuant/docs/workspace_structure.md`
   - `HeadWiseKVQuant/src/hwq/headwise.py`
5. **优先任务**：全矩阵比较已完成！下一步：
   - Top-K × PRQ 叠加：DMD top-4 + PRQ int4+int2，可能的 SOTA 路线
   - QVG PRQ INT2 32-prompt baseline（形成 BF16 / PRQ / Top-K 三足对照）
   - 探索更优 importance metric（当前 DMD loss 在 int8+int4 下 top-k vs random 仅 +0.19pp）
   - 考虑不同层可能需不同 k 的非均匀 top-k 策略
   ```
   # QVG PRQ INT2 32-prompt baseline
   CUDA_VISIBLE_DEVICES=0 PROMPTS_PATH=assets/moviegenbench_32.txt \
   OUTPUT_FOLDER=results/selfforcing/qvg_int2_mb32 \
   bash scripts/self_forcing/run_int2_all.sh
   ```
6. 探索更好的 importance metric（当前 DMD loss top-4 在 int4+int2 下仅比 random 高 0.38pp）
7. 如继续跑实验：
   - Packed-naive R-HWQ-4h：`bash scripts/self_forcing/run_packed_naive_hwq.sh`
   - 注意：`conda activate forcing`（包含 omegaconf 等依赖）
8. 如涉及服务器资源，补看 [服务器工作习惯.md](/data2/moweile-20251213/服务器工作习惯.md)

## 当前最重要信息

- 当前核心目标是 `forcing-based long video generation` 的 KV cache 量化。
- 重点质量维度：`identity consistency`、`scene consistency`、`motion continuity`。
- 当前优先路线：
  - 以 `HeadWiseKVQuant` 为方法代码库推进 `head-wise quant`
  - 以 `Self-Forcing` 为实验集成入口
  - 从 `HeadWiseKVQuant/scripts/self_forcing/` 启动实验
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

- 实验产物统一放到 `HeadWiseKVQuant/results/`（不要放 `outputs/`）。
- 本机权重位于 `HeadWiseKVQuant/ckpts/Self-Forcing/`（不进 git）。
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
