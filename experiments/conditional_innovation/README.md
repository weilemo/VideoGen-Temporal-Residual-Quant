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
- `manifest.json`: exact dump paths, arguments, and source Git SHA.
- `report.md`: compact human-readable result with explicit evidence boundary.

The shuffled negative control takes the prior innovation from a different
validation prompt while preserving layer/head/unit geometry. The experiment
uses only complete units so the final partial unit cannot create an alignment
artifact.

## Interpretation

- `STOPPED_CROSS_GATE`: current K is not a validated V base predictor; do not
  use correction to rescue the story.
- `FAIL_CORRECTION_GATE`: Cross-KV passed, but the K-unexplained V innovation
  did not show sufficiently broad held-out persistence.
- `PASS_E1`: the information hypothesis passed. This only authorizes the next
  reconstructed-state fixed-bit experiment; it is not evidence of video or
  system benefit.
