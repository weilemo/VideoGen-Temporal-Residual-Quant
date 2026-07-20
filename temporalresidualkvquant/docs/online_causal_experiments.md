# TRQ 在线分叉因果定位与质量 Gate

本文对应 2026-07-20 实验计划，固定执行顺序为：

```text
E0 复算 latent 指标
  -> E1 paired VBench + 人工定性
      -> 质量通过：E5
      -> 质量失败：E2/E3 -> E4 -> E5
          -> winner 冻结后，另开分支做 E6
```

E6 的 Triton decoder、decoded-span cache 和 fused attention 不在本分支实现。
质量 winner 未冻结前，不允许把 codec 语义改动和系统优化混在一起。

## 1. 固定 GPU 边界

当前 EPIC 租约只允许物理 GPU 0/1。所有新增生成 launcher 默认使用 GPU 0，
并对其他编号失败关闭：

```bash
GPU_ID=0 DRY_RUN=1 bash scripts/analysis/run_online_causal_diagnosis.sh
GPU_ID=1 STAGE=e2 bash scripts/analysis/run_online_causal_diagnosis.sh
```

GPU 命令应从已经登录的 code-server 终端执行。本地 macOS 只负责开发、CPU
测试和 Git 同步。

## 2. E0：复算 absolute 与 boundary jump

E0 不生成视频，直接复用 7 月 17/18 日 latent：

```bash
bash scripts/analysis/reanalyze_existing_online_runs.sh
```

每个配置输出：

- `online_absolute_metrics.csv`：`early_median`、`late_median`、`growth`、
  `final`、`p95`、`max`、`theil_sen_slope`；
- `online_boundary_jump.csv`：实际首次量化边界前后各 6 个位置的 median 和 jump；
- `online_latent_absolute_curve.png`：TRQ、BF16 repeat、TRQ excess 及事件线；
- `online_prompt_time_heatmap.png`：prompt/seed × latent-frame excess heatmap；
- `online_config_summary.csv`：K2V2/K2V4/K4V2/K4V4 修订版汇总；
- 兼容旧消费者的 `position_rows.csv`、`pair_summary.csv`、`summary.json`。

新 rollout 会把每次实际转换的 global/local span 写进 latent metadata。Delay、
sink 或 Gradual 配置因此按真实首次低比特事件对齐；旧产物才使用 CLI 的
`--first-quant-frame 24` fallback。

## 3. E1：paired VBench 与人工判读

VBench 输入清单为四列 TSV：

```text
config<TAB>seed<TAB>label<TAB>video_dir
```

至少包含相同 prompt/seed 的 `bf16_a`、`bf16_b` 和一个候选配置。执行：

```bash
E1_RUNS_TSV=/path/to/e1_runs.tsv \
VBENCH_ROOT=/path/to/VBench \
GPU_ID=0 \
OUTPUT_ROOT=~/storage/runs/trq_causal_20260720/e1_vbench \
  bash scripts/eval/run_paired_vbench.sh
```

生成并排视频、early/mid/late contact sheet 和人工标签模板：

```bash
python scripts/eval/create_paired_review.py \
  --run bf16_a,0,/path/to/bf16_a_s0 \
  --run bf16_b,0,/path/to/bf16_b_s0 \
  --run k4v4,0,/path/to/trq_k4v4_s0 \
  --output-dir /path/to/e1_review
```

必须人工填写 `failure_tags.csv` 中的 `catastrophe` 和 `bf16_present`。空白行
不算完成，quality gate 会保持 `INCOMPLETE`。

聚合示例：

```bash
python scripts/analysis/analyze_paired_quality.py \
  --vbench-run bf16_a,0,bf16_a_s0,/path/to/vbench_scores \
  --vbench-run bf16_b,0,bf16_b_s0,/path/to/vbench_scores \
  --vbench-run k4v4,0,k4v4_s0,/path/to/vbench_scores \
  --latent k4v4=/path/to/e0/k4v4/online_absolute_metrics.csv \
  --failure-tags /path/to/e1_review/failure_tags.csv \
  --output-dir /path/to/e1_analysis
```

`quality_gate.json` 只有同时满足以下条件才给出 `PASS`：

- paired Final mean delta 不低于 `-0.005`；
- subject/background/motion mean delta 均不低于 `-0.01`；
- TRQ 独有且可复现的 catastrophe 少于 2 个 pair；
- 每条 pair 已有明确人工判读。

## 4. E2/E3：因果干预

先 dry-run 确认矩阵：

```bash
GPU_ID=0 DRY_RUN=1 bash scripts/analysis/run_online_causal_diagnosis.sh
```

E2 使用 Smoke4 原始 prompt 文件的 index 0/2，因此不会因子集重编号而破坏
与 BF16-A/B 的严格配对：

```bash
GPU_ID=0 STAGE=e2 bash scripts/analysis/run_online_causal_diagnosis.sh
```

矩阵是 Delay-48、Delay-72、BF16-sink24、BF16-sink48、Gradual-3，两个
prompts × 两个 seeds，共 20 条新视频。

E3 使用东京街头 prompt / seed 0：

```bash
GPU_ID=1 STAGE=e3 bash scripts/analysis/run_online_causal_diagnosis.sh
```

它运行 K-only、V-only、layers 0-7、8-19、20-29。第一次实际量化后会对确定性
采样的 query/key positions 记录：K/V cache-read Rel-L2、logits cosine/Rel-L2、
softmax KL、top-1/top-k agreement 和 attention-output Rel-L2。采样上限可通过
`TRQ_ATTENTION_TRACE_MAX_Q/MAX_K/TOPK` 覆盖，默认 `64/256/8`。

## 5. E4：最多四个保护候选

E2/E3 结果出来后再填写 TSV，不预先把所有保护机制混合：

```text
# name k_bits v_bits first schedule gradual protected_sink attention_sink roles layers
p1_k8v2<TAB>8<TAB>2<TAB>24<TAB>bulk<TAB>3<TAB>0<TAB>0<TAB>both<TAB>all
```

运行：

```bash
GPU_ID=0 E4_CANDIDATES_TSV=/path/to/e4.tsv \
  bash scripts/analysis/run_e4_candidates.sh
```

脚本拒绝超过 4 个候选。完成 E0/E1 和 runtime 聚合后，执行：

```bash
python scripts/analysis/analyze_protection_gate.py \
  --quality-gate /path/to/quality_gate.json \
  --baseline-latent /path/to/k2v2/online_absolute_metrics.csv \
  --candidate-latent p1_k8v2=/path/to/p1/online_absolute_metrics.csv \
  --runtime-root p1_k8v2=/path/to/e4/p1_k8v2 \
  --output-dir /path/to/e4/gate
```

`PASS` 需要质量通过、boundary jump 相比 K2V2 降低至少 50%、actual saving
至少 70%。saving 在 60%-70% 的方案标为 `QUALITY_CEILING`，不冒充最佳 Pareto。

## 6. E5：分阶段长 rollout

E5 launcher 必须读取 `quality_gate.json`，winner 不是 `PASS` 时拒绝运行。三个
阶段分开调用，不能在一次队列里越过 501 quality stop rule：

```bash
GPU_ID=0 STAGE=183 WINNER_CONFIG=p1_k8v2 \
QUALITY_GATE_JSON=/path/to/quality_gate.json \
TRQ_K_BITS=8 TRQ_V_BITS=2 \
  bash scripts/analysis/run_quality_gated_long_rollout.sh

GPU_ID=0 STAGE=501 WINNER_CONFIG=p1_k8v2 \
QUALITY_GATE_JSON=/path/to/quality_gate.json \
TRQ_K_BITS=8 TRQ_V_BITS=2 \
  bash scripts/analysis/run_quality_gated_long_rollout.sh
```

501 完成后会按 latent window 180、stride 90 准备视频片段和 manifest。对这些
窗口完成 paired VBench 与人工判读后，只有 501 quality gate 为 `PASS` 才能启动：

```bash
GPU_ID=1 STAGE=699 WINNER_CONFIG=p1_k8v2 \
QUALITY_GATE_JSON=/path/to/quality_gate.json \
E5_501_QUALITY_GATE_JSON=/path/to/501_quality_gate.json \
TRQ_K_BITS=8 TRQ_V_BITS=2 \
  bash scripts/analysis/run_quality_gated_long_rollout.sh
```

所有正式输出仍需记录 commit、checkpoint SHA256、prompt SHA256、seed、GPU、
CUDA/PyTorch、完整 TRQ schedule、实际 state bytes 和视频清单。
