# Importance Top-K Head-Wise Quantization

This note describes the first fixed-policy version of importance-based
head-wise KV-cache quantization.

## Idea

Use the focused-forcing head-ablation sweep to assign an importance score to
each attention head:

```text
global_head_id = layer_idx * num_heads + head_idx
importance = mean DMD loss after masking this head
```

For Wan2.1-T2V-1.3B, the current defaults are:

```text
num_layers = 30
num_heads = 12
total_heads = 360
```

The first policy is deliberately simple:

- fixed for the whole run
- per-layer top-k heads use high precision
- all remaining heads use low precision
- K and V share the same head policy
- all prompts/chunks share the same policy

For the packed-naive branch, the intended default is:

```text
top-k heads: packed-naive-int4
other heads: packed-naive-int2
```

## Build A Policy From Focused-Forcing Outputs

`temporalresidualkvquant` now contains the full analysis chain.  The vendored
Self-Forcing backend can mask one global attention head per sample, compute DMD
loss, save the focused-forcing-style JSON files, and aggregate them into a
top-k policy.

Run the complete calibration / policy-generation step:

```bash
cd /data2/moweile-20251213/workspace/videoquant/temporalresidualkvquant

bash scripts/self_forcing/run_head_importance_analysis.sh
```

The launcher is split internally into two independent phases to avoid holding
the Self-Forcing inference KV cache and DMD scoring models in VRAM at the same
time:

```bash
# Phase 1 only: mask heads and save generated latents
PHASE=inference bash scripts/self_forcing/run_head_importance_analysis.sh

# Phase 2 only: load saved latents, compute DMD loss, aggregate policy
PHASE=scoring bash scripts/self_forcing/run_head_importance_analysis.sh
```

Useful smoke-test / resume knobs:

```bash
HEAD_START=0 HEAD_END=6 NUM_OUTPUT_FRAMES=42 HEADS_PER_BATCH=1 \
  PHASE=inference bash scripts/self_forcing/run_head_importance_analysis.sh

ALLOW_INCOMPLETE=1 PHASE=scoring \
  bash scripts/self_forcing/run_head_importance_analysis.sh
```

Set `DELETE_LATENTS_AFTER_SCORING=1` to remove `.pt` latent files after scoring
when disk space matters.  Set `SKIP_EXISTING=0` to recompute existing chunk JSON
entries.

Defaults:

```text
analysis output: results/head_importance/focused_forcing_dmd/
policy output: assets/head_importance/top4_dmd_loss.json
heads per inference batch: 3
num layers / heads: 30 / 12
top-k: 4
```

Then run packed-naive top-k HWQ:

```bash
HEAD_IMPORTANCE_PATH=assets/head_importance/top4_dmd_loss.json \
  bash scripts/self_forcing/run_packed_naive_topk_hwq.sh
```

If you already have JSON files from a separate focused-forcing run, collect
them into one top-k policy.  The selection logic lives in the library module
`trq.head_importance`; the script below is a thin CLI wrapper:

```bash
cd /data2/moweile-20251213/workspace/videoquant/temporalresidualkvquant

python scripts/aggregate_head_importance.py \
  --input /path/to/focusedforcing_dm_loss_outputs \
  --output assets/head_importance/top4_dmd_loss.json \
  --num_layers 30 \
  --num_heads 12 \
  --top_k 4
```

Python API:

```python
from trq.head_importance import build_topk_policy_from_focused_forcing, write_topk_policy

policy = build_topk_policy_from_focused_forcing(
    "/path/to/focusedforcing_dm_loss_outputs",
    num_layers=30,
    num_heads=12,
    top_k=4,
)
write_topk_policy(policy, "assets/head_importance/top4_dmd_loss.json")
```

The input can be either one JSON file or a directory containing JSON files like:

```json
{
  "0": 0.123,
  "1": 0.117,
  "2": 0.151
}
```

Keys are global head ids.  Values are DMD losses from masking that head.  By
default, larger loss means the head is more important.

The output policy contains both raw scores and selected heads:

```json
{
  "format": "headwise-topk-policy-v1",
  "num_heads": 12,
  "top_k": 4,
  "score_direction": "higher",
  "top_heads_by_layer": {
    "0": [1, 4, 7, 10],
    "1": [0, 2, 5, 11]
  }
}
```

You can also hand-write a policy JSON with only `top_heads_by_layer` when you
want to test a specific selection.

## Run Packed-Naive Top-K HWQ

On this machine:

```bash
cd /data2/moweile-20251213/workspace/videoquant/temporalresidualkvquant

HEAD_IMPORTANCE_PATH=assets/head_importance/top4_dmd_loss.json \
  bash scripts/self_forcing/run_packed_naive_topk_hwq.sh
```

Equivalent explicit form:

```bash
HEADWISE_MODE=topk \
HEAD_IMPORTANCE_PATH=assets/head_importance/top4_dmd_loss.json \
NUM_HIGH_PRECISION_HEADS=4 \
HIGH_PRECISION_QUANT_TYPE=packed-naive-int4 \
LOW_PRECISION_QUANT_TYPE=packed-naive-int2 \
QUANT_TYPE=packed-naive-int2 \
  bash scripts/self_forcing/run_packed_naive_hwq.sh
```

Outputs are written under:

```text
temporalresidualkvquant/results/selfforcing/topk_<policy-name>_hi_4_packed-naive-int4_lo_packed-naive-int2_64/
```

## Path Handling On Another Machine

The launchers infer paths in this order:

1. `SELF_FORCING_CKPT_ROOT`, if set
2. `temporalresidualkvquant/ckpts/Self-Forcing`, if present
3. `QVG_ROOT/ckpts/Self-Forcing`, if `QVG_ROOT` is set

For a different checkout path, use absolute paths:

```bash
cd /mnt/workspace/caipeiliang/code/moweile/videoquant/temporalresidualkvquant

SELF_FORCING_CKPT_ROOT=/mnt/workspace/caipeiliang/code/moweile/videoquant/temporalresidualkvquant/ckpts/Self-Forcing \
bash scripts/self_forcing/run_head_importance_analysis.sh

SELF_FORCING_CKPT_ROOT=/mnt/workspace/caipeiliang/code/moweile/videoquant/temporalresidualkvquant/ckpts/Self-Forcing \
HEAD_IMPORTANCE_PATH=/mnt/workspace/caipeiliang/code/moweile/videoquant/temporalresidualkvquant/assets/head_importance/top4_dmd_loss.json \
  bash scripts/self_forcing/run_packed_naive_topk_hwq.sh
```

If checkpoints already live in the old QVG tree:

```bash
QVG_ROOT=/mnt/workspace/caipeiliang/code/moweile/videoquant/Quant-VideoGen \
bash scripts/self_forcing/run_head_importance_analysis.sh

QVG_ROOT=/mnt/workspace/caipeiliang/code/moweile/videoquant/Quant-VideoGen \
HEAD_IMPORTANCE_PATH=/mnt/workspace/caipeiliang/code/moweile/videoquant/temporalresidualkvquant/assets/head_importance/top4_dmd_loss.json \
  bash scripts/self_forcing/run_packed_naive_topk_hwq.sh
```

Prompt and output paths can also be overridden:

```bash
PROMPTS_PATH=/path/to/t2v.txt \
OUTPUT_FOLDER=/path/to/results/topk_packed_naive \
HEAD_IMPORTANCE_PATH=/path/to/top4_dmd_loss.json \
  bash scripts/self_forcing/run_packed_naive_topk_hwq.sh
```
