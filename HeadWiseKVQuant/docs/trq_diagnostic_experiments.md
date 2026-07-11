# TRQ Residual Diagnostics

这套实验框架用于回答两个独立问题，不预设结论：

1. 实际 codec residual 是否比同位置的 Raw KV 更集中，并在同预算 INT2 下获得更低误差？
2. residual chain 是否随 span 内 rollout position 产生可观测漂移？

主入口：

```bash
bash scripts/analysis/run_trq_diagnostics.sh
```

只运行单项实验：

```bash
bash scripts/analysis/run_raw_vs_residual_distribution.sh
bash scripts/analysis/run_residual_chain_drift.sh
```

## 数据定义

输入必须是 `quant_type=none` 生成的 BF16 KV dump。分析器会拒绝已有
`quantized_spans` 的 dump。Self-Forcing 缓存中的 K 是 pre-RoPE K，因此本实验
不包含 RoPE predictor。

对同一个非 anchor unit $x_t$，同时计算：

$$
r_t^{\mathrm{oracle}} = x_t - f(x_{t-1})
$$

$$
r_t^{\mathrm{chain}} = x_t - f(\hat{x}_{t-1})
$$

其中 $\hat{x}_{t-1}$ 来自当前 `trq.py` 的真实 PyTorch codec，包括 BF16
scale、padding、asymmetric zero-point 和 bit packing。Raw KV 与 residual 使用
完全相同的非 anchor mask。

## 实验 1：Raw KV vs Residual

主要图：

- `raw_vs_residual_distribution.png/pdf`：K/V 分开绘制 signed density 和
  $|x|$ ECDF，包含 Raw、oracle residual、chain residual。

主要统计：

- `std`、RMS、mean absolute value、$|x|$ 的 p50/p90/p95/p99/max。
- `RMS(residual) / RMS(raw)` 和 p99 收缩比。
- 同一 block size、scale precision 和 asymmetric zero-point 下的直接
  `Q2(raw)` 与 `Q2(chain residual)` Rel-L2、NMSE、SQNR。
- TRQ 完整 state bytes 和 effective bits/value，包含 anchor、scale、zero point
  与 predictor 参数，不能只报告 nominal 2-bit。

预注册判据按 K/V 分别执行：

- 至少两个 held-out dumps。
- concentration：prompt-level bootstrap 的 median log RMS ratio 95% CI 上界
  小于 0，且 paired win rate 不低于 80%。
- INT2 suitability：`NMSE_TRQ2 - NMSE_direct2` 的 bootstrap 95% CI 上界小于 0。

如果 V 或部分层不满足判据，应报告为局部否证，不能用 K 的结果替代 V。

## 实验 2：Residual Chain Drift

实际 chain：

$$
\hat{x}_t^{\mathrm{chain}}
= f(\hat{x}_{t-1}^{\mathrm{chain}})
+ Q\left(x_t-f(\hat{x}_{t-1}^{\mathrm{chain}})\right)
$$

teacher-forced 对照每步使用 raw $x_{t-1}$。二者差值隔离上一 unit 的误差
通过下一步 residual range 间接传播的部分。量化误差不会简单代数相加，因此
只画累计 Rel-L2 不足以判断 drift。

主要图：

- `residual_chain_drift.png/pdf`：chain、teacher-forced、chain excess 随 span
  内 position 变化。
- `chain_drift_layer_heatmap.png/pdf`：layer × chain position 的 excess error。
- `reset_interval_ablation.png/pdf`：重锚间隔 `1/2/4/8/24/no-reset` 对 p95
  chain error 的影响。

主要指标：chain/teacher Rel-L2、excess Rel-L2、prediction perturbation、
residual inflation、相邻误差 cosine、drift slope、late/early ratio 和 max error。

新格式 dump 会记录实际 generation chunk 序列和每个已压缩 span 的 frame 边界；
分析器优先按这些边界重放，并排除运行时尚未压缩的 BF16 tail。旧 dump 没有调度
metadata 时，才回退到 `RUNTIME_RESET_INTERVAL=24`，对应当前常规配置的
`8 generation blocks × 3 frames/block`。`no-reset` 是压力测试，不代表当前运行路径。

## 采集与正式运行

默认采集 4 个 MovieGenBench prompts，2 个 train、2 个 held-out。完整 180-frame
dump 约 48 GiB/prompt，总磁盘预算约 200 GiB。`layer_shards` 模式按 layer 保存，
不会降低总磁盘量，但把分析峰值内存从整模型 dump 降到单层规模。

默认 2 条 held-out 只用于初跑和排错。identity predictor 的正式 MB32 统计应设
`MAX_PROMPTS=32 TRAIN_N=0`，让 32 条全部进入 held-out；全层完整 dump 约需
1.5 TiB，因此先做下面的 6 层 smoke test，再决定正式采集层数和存储位置。

机器和 checkpoint 到位后，一条命令采集并运行两项 identity 实验：

```bash
cd /Users/moweile/Code/LAB/videoquant-trq/HeadWiseKVQuant

SELF_FORCING_CKPT_ROOT=/path/to/Self-Forcing \
COLLECT_DUMPS=1 \
MAX_PROMPTS=4 \
TRAIN_N=2 \
PREDICTOR_MODE=identity \
LAYERS=all \
bash scripts/analysis/run_trq_diagnostics.sh
```

先用 6 层做链路 smoke test：

```bash
COLLECT_DUMPS=1 \
DUMP_LAYERS=0,5,11,17,23,29 \
LAYERS=0,5,11,17,23,29 \
bash scripts/analysis/run_trq_diagnostics.sh
```

使用已经存在的 dump 重跑分析：

```bash
DUMPS_GLOB='/path/to/heldout/*_layer*.pt' \
OUTPUT_DIR='results/trq_diagnostics/identity_int2_mb32' \
bash scripts/analysis/run_trq_diagnostics.sh
```

Affine 必须使用仅由 train split 拟合的参数：

```bash
PREDICTOR_MODE=affine_channel \
PREDICTOR_PARAMS_PATH=assets/trq_predictors/affine_channel_self_forcing_dmd.pt \
DUMPS_GLOB='/path/to/heldout/*_layer*.pt' \
bash scripts/analysis/run_trq_diagnostics.sh
```

常用环境变量：`NUM_BITS`、`ANCHOR_BITS`、`BLOCK_SIZE`、
`PREDICTOR_STRIDE`、`RUNTIME_RESET_INTERVAL`、`RESET_INTERVALS`、`LAYERS`、
`KV`、`DEVICE`、`MAX_DUMPS`、`RUN_ID`、`OUTPUT_DIR`。

## 输出

每次运行写入 `results/trq_diagnostics/<run_id>/`：

```text
manifest.json
resolved_config.json
distribution_rows.jsonl
distribution_summary.csv
chain_drift_rows.jsonl
chain_drift_summary.csv
summary.json
report.md
run.log
figures/*.png
figures/*.pdf
```

`manifest.json` 记录 git 状态、Torch/CUDA/GPU、dump hash/size 和 dump metadata。
新 dump 额外记录 prompt、输出帧数、block size、实际 chunk/span 调度、KV layout
和 `key_position=pre_rope`。

## 结论边界

该实验是固定 BF16 trajectory 上的离线 codec 诊断，可以判断 cache
reconstruction 是否漂移，但不能直接证明生成视频 trajectory 不漂移。后续仍需
相同 prompt/seed 的 BF16 与 TRQ 在线 paired rollout，并用 BF16-vs-BF16 repeat
估计非确定性下限。
