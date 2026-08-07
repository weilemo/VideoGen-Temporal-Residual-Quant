# Conditional Innovation E1

This experiment tests whether the value component not explained by the current
key remains temporally predictable across latent-frame units. It implements the
E0/E1 part of the research plan without making a quantized closed-loop or video
quality claim.

The entry point always evaluates the frozen Cross-KV predictor first. If the
prompt-level Cross/temporal ratio misses Gate 0, the run writes
`STOPPED_CROSS_GATE` and does not fit the temporal correction.

## Inputs

- Raw BF16 KV dumps accepted by `trq.analysis.kv_dump`.
- Explicit, prompt-disjoint calibration and validation dump sets.
- A latent-frame unit size, either from `frame_seq_length` metadata or
  `UNIT_SIZE`.

## Remote launch

```bash
cd temporalresidualkvquant
CALIBRATION_DUMPS='/path/to/calibration/*_layer*.pt' \
VALIDATION_DUMPS='/path/to/heldout/*_layer*.pt' \
LAYERS='8-19' \
UNIT_SIZE=1560 \
QUANTILE_MAX_SAMPLES=262144 \
OUTPUT_DIR="$PWD/results/conditional_innovation/e1_$(date +%Y%m%d_%H%M%S)" \
bash scripts/analysis/run_conditional_innovation_e1.sh
```

Set `REQUIRE_PASS=1` only in a gate-driven orchestrator. A scientific negative
result is still a successful analysis run and should normally exit zero after
writing its report.

## Outputs

- `summary.json`: Gate 0/Gate 1 verdicts and prompt-bootstrap confidence intervals.
- `layer_head_rows.csv`: per-prompt/layer/head Cross, Oracle, and shuffled metrics.
- `prompt_rows.csv`: prompt-level ratios used by the bootstrap.
- `conditional_innovation_params.pt`: frozen Cross parameters and, only after
  Gate 0 passes, fitted bounded gamma values.
- `cross_kv_checkpoint.pt`: Cross-KV parameters written before Gate 0
  evaluation, so a later diagnostic failure does not erase the fitted model.
- `hybrid_checkpoint.pt`: Cross-KV plus gamma parameters written before Gate 1
  evaluation when Gate 0 passes.
- `progress.json`: last completed experiment stage.
- `manifest.json`: exact dump paths, arguments, and source Git SHA.
- `report.md`: compact human-readable result with explicit evidence boundary.

The shuffled negative control takes the prior innovation from a different
validation prompt while preserving layer/head/unit geometry. The experiment
uses only complete units so the final partial unit cannot create an alignment
artifact.

MSE, NMSE, and all gate ratios are exact full-data reductions. Distribution
standard deviations are also exact. Only the diagnostic absolute p99 uses a
fixed-seed, proportionally stratified sample capped by
`QUANTILE_MAX_SAMPLES`; the recorded `*_p99_samples` columns make that bound
explicit.

## Interpretation

- `STOPPED_CROSS_GATE`: current K is not a validated V base predictor; do not
  use correction to rescue the story.
- `FAIL_CORRECTION_GATE`: Cross-KV passed, but the K-unexplained V innovation
  did not show sufficiently broad held-out persistence.
- `PASS_E1`: the information hypothesis passed. This only authorizes the next
  reconstructed-state fixed-bit experiment; it is not evidence of video or
  system benefit.

## E2 fixed-bit reconstructed-state experiment

E2 consumes a completed `PASS_E1` directory and reuses its frozen Cross-KV and
gamma parameters. It evaluates packed fixed-bit Direct-V (with the shared K
codec held fixed), Temporal, Cross, Oracle Hybrid, Closed-loop Hybrid,
within-prompt shuffled control, and fixed reset-span variants. K and
Closed-loop V prediction use reconstructed state; the oracle is explicitly
diagnostic.

```bash
E1_DIR=/path/to/pass_e1 \
OUTPUT_DIR="$PWD/results/conditional_innovation/e2_k4v4" \
LAYERS='8-19' \
UNIT_SIZE=1560 \
KEY_BITS=4 VALUE_BITS=4 ANCHOR_BITS=4 BLOCK_SIZE=64 \
RESET_SPANS='2,4,8' \
bash scripts/analysis/run_conditional_innovation_e2.sh
```

Every dump/layer record is written atomically under `shards/`; rerunning the
same output directory skips completed records. The aggregate report includes
packed payload, scale/zero-point, predictor, gamma, and K payload bytes.

The current raw dumps do not contain matched BF16 query tensors. Consequently
E2 can pass its reconstructed-state core but its overall status remains
`BLOCKED_E2_SUBGATES`. It cannot authorize E3 until attention-output error,
event recovery, and saturation are measured. Quantization-aware gamma,
the unconstrained concat-capacity baseline, and event-aware reset also remain
explicit gaps; fixed reset spans are reported.

## Gate-driven serial queue

The queue can take over an already-running E1. It waits for `analysis.complete`,
stops on a negative E1 result, launches resumable E2 only on `PASS_E1`, and
then stops at the attention gate instead of silently starting video jobs:

```bash
E1_DIR=/path/to/active_e1 \
E2_OUTPUT_DIR=/path/to/e2_k4v4 \
LAYERS='8-19' UNIT_SIZE=1560 \
nohup bash scripts/analysis/run_conditional_innovation_queue.sh \
  > /tmp/conditional_innovation_queue.log 2>&1 &
```

The atomic `QUEUE_STATE` JSON records `WAITING_E1`, `RUNNING_E2`, a scientific
stop, or `WAITING_E2_SUBGATES`. GPU video stages are intentionally not
started by this CPU queue.
