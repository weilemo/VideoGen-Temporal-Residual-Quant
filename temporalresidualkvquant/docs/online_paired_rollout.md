# Online Paired Rollout

该实验用相同 prompt 和 seed 生成三条在线 trajectory：

```text
BF16-A
BF16-B
TRQ
```

`BF16-A vs BF16-B` 是实现非确定性下限，`BF16-A vs TRQ` 是量化引入的
trajectory 偏差。分析器比较完整 clean latent，不从有损 mp4 反推 latent drift。

## 一键 Smoke4

默认 prompt 是固定的 MB32 1-based ID `1,4,12,23`，文件为
`assets/mb32_paired_smoke4.txt`。默认运行 seed `0,1`、180 frames、K2V2：

```bash
export SELF_FORCING_CKPT_ROOT=/path/to/ckpts/Self-Forcing

DRY_RUN=1 bash scripts/analysis/run_online_paired_rollout.sh
bash scripts/analysis/run_online_paired_rollout.sh
```

常用覆盖：

```bash
SEEDS=0 \
TRQ_K_BITS=2 \
TRQ_V_BITS=4 \
OUTPUT_ROOT=results/online_paired/smoke4_k2v4 \
  bash scripts/analysis/run_online_paired_rollout.sh
```

完整 MB32：

```bash
PROMPTS_PATH=assets/moviegenbench_32.txt \
SEEDS=0 \
OUTPUT_ROOT=results/online_paired/mb32_k2v2 \
  bash scripts/analysis/run_online_paired_rollout.sh
```

launcher 会为每个 seed 独立启动 BF16-A、BF16-B 和 TRQ 进程。不要把同一
进程内的两个 sample 当作 BF16 repeat。

## 单条 Launcher 的仪表化开关

`run_bf16.sh` 和 `run_hrq_predictor_ablation.sh` 支持：

```bash
PROFILE_RUNTIME=1 \
SAVE_ROLLOUT_LATENTS=1 \
ROLLOUT_METRICS_DIR=/path/to/rollout_metrics \
  bash scripts/self_forcing/run_bf16.sh
```

对应底层 CLI：

```text
--profile
--save_rollout_latents
--rollout_metrics_dir <path>
```

每个 prompt 的输出：

```text
rollout_metrics/
├── latents/<prompt>-<sample>_<model>.pt
└── runtime/<prompt>-<sample>_<model>.json
```

latent payload 包含 prompt index、seed、帧数、量化配置和 `[T,C,H,W]` 的 BF16
clean latent。runtime JSON 包含：

- pipeline 与 wall-clock 端到端耗时；
- init/text、diffusion、VAE 和逐 block 时间；
- quantize/dequantize 调用次数与 CUDA event 时间；
- K/V actual tensor payload bytes、BF16 等价 bytes、effective bits/value；
- compression ratio、saving fraction；
- max allocated/reserved CUDA memory。

`quantize_ms` 会包含量化阶段内部必要的 cache decode，因而可能与
`dequantize_ms` 有重叠；二者用于定位热点，不能相加后当成总 overhead。端到端
结论只使用 pipeline/wall-clock 时间。

profiling 模式不运行旧的逐层 reconstruction error 和 offload/onload 显存探针，
因为二者会明显污染端到端延迟。旧显存探针只能在单独的非 profiling run 中用
`TRQ_LEGACY_MEMORY_PROBE=1` 开启；需要 profiling 中的逐层误差时设置
`TRQ_PROFILE_RECON_ERROR=1`，并把该 run 标为诊断计时而不是正式 latency。

## 单独分析已有 Latent

```bash
PYTHONPATH=src python scripts/analysis/analyze_online_paired_rollout.py \
  --bf16-a 'results/run/bf16_a_s*/rollout_metrics/latents/*.pt' \
  --bf16-b 'results/run/bf16_b_s*/rollout_metrics/latents/*.pt' \
  --trq 'results/run/trq_k2v2_s*/rollout_metrics/latents/*.pt' \
  --output-dir results/run/analysis
```

分析器按 `(prompt_index, seed, sample_index)` 严格匹配，任一缺失或 latent shape
不一致都会失败。输出：

```text
analysis/
├── position_rows.csv
├── pair_summary.csv
├── summary.json
└── report.md
```

主要统计：

$$
d_{i,t}^{z}(A,B)
=
\frac{\lVert z_{i,t}^{A}-z_{i,t}^{B}\rVert_2}
{\lVert z_{i,t}^{A}\rVert_2+\epsilon}
$$

以及 BF16 noise 扣除后的 late-minus-early growth、Theil-Sen slope 和
prompt-level bootstrap 95% CI。latent gate 通过不等于质量等价；正式结论还必须
补 VBench 与 late-frame 定性检查。

## 超过 180 帧

pre-RoPE `ChunkedKVCache` 已实现 local-window eviction：

- 完整量化 span 只平移 chunk index，保留 packed state；
- 被 eviction 边界截断的 residual span 解码一次，保留部分转成 BF16；
- K 在读出后按真实 global frame offset 应用 RoPE；
- 每次量化使用本地窗口最近一个 24-frame interval，不使用越界的全局 token index。

CPU 单测覆盖 span 截断和平移，但在启动 501/699 帧实验前仍必须先做 183-frame
BF16/TRQ GPU parity。CPU 通过不能替代真实 attention 与视频验证。
