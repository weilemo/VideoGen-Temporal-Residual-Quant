# HeadWiseKVQuant

`HeadWiseKVQuant` is a standalone research codebase for low-bit KV-cache
quantization in long video generation.

The first version extracts the reusable KV-cache quantization framework from
`Quant-VideoGen` and makes the head-wise mixed precision policy explicit.  The
goal is to keep the paper-facing method code modular instead of coupling it to
one experiment directory.

## Scope

This repository contains:

- low-bit KV-cache compression and decompression
- chunked KV-cache storage for mixed BF16 / quantized spans
- KMeans / PRQ based quantization kernels
- random head-group mixed precision as the first `HWQ` baseline
- fixed per-layer importance top-k head-group mixed precision
- a small compatibility layer for Self-Forcing style KV tensors
- a vendored Self-Forcing inference backend

This repository intentionally does not contain:

- Wan / Self-Forcing model weights

The recommended research workflow is to work from `HeadWiseKVQuant`.  The
Self-Forcing model and pipeline code are vendored under
`backends/self_forcing/`, while large checkpoints stay outside git.

Expected workspace layout:

```text
HeadWiseKVQuant/
├── src/hwq/                 # method code
├── backends/self_forcing/   # Self-Forcing model and pipeline code
├── scripts/self_forcing/    # launchers
├── assets/t2v.txt           # default prompts
└── ckpts/Self-Forcing/      # optional local checkpoint path, ignored by git
```

If your checkpoints live elsewhere, set
`SELF_FORCING_CKPT_ROOT=/path/to/ckpts/Self-Forcing`.

The local checkpoint directory is ignored by git.  It is safe to keep large
weights under `HeadWiseKVQuant/ckpts/Self-Forcing/` without pushing them.

## Install

From this directory:

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Minimal API

```python
from hwq import QuantizeConfig
from hwq.headwise import RandomHeadPolicy, compress_headwise_kv_cache

policy = RandomHeadPolicy(
    num_heads=12,
    num_high_precision_heads=4,
    high_precision_quant_type="triton-nstages-kmeans-int4",
    low_precision_quant_type="triton-nstages-kmeans-int2",
    seed=0,
)

quant_config = QuantizeConfig(
    quant_type="triton-nstages-kmeans-int2",
    quant_block_size=64,
    cache_num_k_centroids=256,
    cache_num_v_centroids=256,
    kmeans_max_iters=2,
    num_prq_stages=1,
)

# k, v: [B, H, S, D]
k_cache, v_cache = compress_headwise_kv_cache(k, v, quant_config, policy)
```

## Current Baseline

The first research baseline is `R-HWQ`:

- random head-wise quantization
- all layers share one randomly selected high-precision head group
- high group: INT4
- low group: INT2

This is a sanity-check baseline.  It verifies that mixed head precision works
before adding importance-based head selection.

The packed naive branch is available as:

- `packed-naive-int2`
- `packed-naive-int4`
- `packed-naive-int8`

Unlike `naive-int2/int4`, these quant types store packed low-bit codes plus
per-block min/scale metadata, so they are real KV-cache compression baselines.

The first importance-based policy is `headwise_mode=topk`: each layer keeps the
top-k important heads at high precision and quantizes the remaining heads with
the low-precision type.  Head selection from focused-forcing ablation JSON is
implemented in `hwq.head_importance`; see `docs/head_importance_topk.md`.

## Self-Forcing Experiments

From this directory:

```bash
bash scripts/self_forcing/run_random_hwq.sh
```

Useful baselines:

```bash
bash scripts/self_forcing/run_bf16.sh
bash scripts/self_forcing/run_int2_all.sh
bash scripts/self_forcing/run_random_hwq.sh
bash scripts/self_forcing/run_packed_naive_hwq.sh
bash scripts/self_forcing/run_head_importance_analysis.sh
HEAD_IMPORTANCE_PATH=assets/head_importance/top4_dmd_loss.json \
  bash scripts/self_forcing/run_packed_naive_topk_hwq.sh
```

By default, outputs are written under:

```text
HeadWiseKVQuant/results/selfforcing/
```

On a machine with a different QVG path:

```bash
SELF_FORCING_CKPT_ROOT=/mnt/workspace/caipeiliang/code/moweile/videoquant/Quant-VideoGen/ckpts/Self-Forcing \
  bash scripts/self_forcing/run_random_hwq.sh
```

See `docs/checkpoint_sync.md` for the full cross-machine setup workflow.
See `docs/head_importance_topk.md` for building and running an importance
top-k packed-naive policy.

## VBench Evaluation Results (2026-05-17)

Five experiment lines evaluated with VBench-Long (8 dimensions, 2 videos each, 180 frames):

| Experiment | Final Score | vs BF16 | Peak GPU Mem | KV Cache | Compression |
|---|---|---|---|---|---|
| BF16 Baseline | 0.6486 | — | ~78 GB* | ~64 GB* | 1× |
| QVG INT2 (PRQ) | 0.6469 | ↓0.26% | ~26 GB* | ~12 GB* | ~6× |
| R-HWQ-4h (PRQ) | 0.6416 | ↓1.07% | ~30 GB* | ~16 GB* | ~4-5× |
| R-HWQ-4h Packed (int8+int4) | 0.6479 | ↓0.10% | 33.6 GB | 19.6 GB | 3.3× |
| R-HWQ-4h Packed (int4+int2) | 0.6279 | ↓3.19% | 26.1 GB | 12.1 GB | 5.3× |

\* Estimated. Bold values are measured from inference logs.

Key findings:
- PRQ-based quantization barely degrades quality (↓0.3%-1.1%) with ~5-6× compression
- Packed-naive int8+int4 is nearly lossless (↓0.10%) at 3.3× compression — viable lightweight option
- Packed-naive int4+int2 provides 5.3× compression at ↓3.19% quality cost

Evaluation scripts: `scripts/eval/evaluate_experiments.sh`, `scripts/eval/aggregate_results.py`.
Full results: `results/selfforcing/vbench_eval/comparison_summary.json`.

## Attribution

The low-bit quantization kernels and PRQ implementation are derived from the
`Quant-VideoGen` codebase.  This repository reorganizes that implementation
into a standalone method framework for head-wise KV-cache quantization research.
