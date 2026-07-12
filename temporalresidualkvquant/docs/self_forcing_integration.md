# Self-Forcing Integration Notes

This repository is the method workspace.  It keeps quantization policy,
compression metadata, and decompression in `trq`, while vendoring the
Self-Forcing model backend under `backends/self_forcing/`.

The expected layouts are:

- Self-Forcing cache span: `[B, S, H, D]`
- HWQ quantizer input: `[B, H, S, D]`
- HWQ decompressed output: `[B, H, S, D]`

The adapter helper is:

```python
from trq import QuantizeConfig
from trq.headwise import RandomHeadPolicy
from trq.self_forcing import compress_self_forcing_cache_span

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

k_cache, v_cache = compress_self_forcing_cache_span(k_bshd, v_bshd, quant_config, policy)
```

For the current `Quant-VideoGen` Self-Forcing code, the integration point is:

- `experiments/Self-Forcing/pipeline/causal_inference.py`
- `quantize_kv_cache()`

The clean integration path is:

1. Read `layer["k"].read(...)` and `layer["v"].read(...)`, which returns BSHD.
2. Call `compress_self_forcing_cache_span(...)`.
3. Store the packed result back with `ChunkedKVCache.store_quantized(...)`.
4. Let `ChunkedKVCache.read(...)` call `uncompress_single_cache(...)` when a quantized span is accessed.

This keeps the video model repository responsible for inference scheduling and
keeps this repository responsible for quantization policy, compression metadata,
and decompression.

## Running From temporalresidualkvquant

The main launcher is:

```bash
cd /path/to/videoquant/temporalresidualkvquant
bash scripts/self_forcing/run_random_hwq.sh
```

The script uses the vendored backend by default:

```text
temporalresidualkvquant/
├── src/trq/
├── backends/self_forcing/
├── scripts/self_forcing/
└── assets/t2v.txt
```

Large checkpoints are intentionally not copied into git.  Put them at:

```text
temporalresidualkvquant/ckpts/Self-Forcing/
```

or point to an existing checkpoint directory:

```bash
SELF_FORCING_CKPT_ROOT=/mnt/workspace/caipeiliang/code/moweile/videoquant/Quant-VideoGen/ckpts/Self-Forcing \
  bash scripts/self_forcing/run_random_hwq.sh
```

The launcher sets:

```bash
PYTHONPATH="${HWQ_ROOT}/src:${HWQ_ROOT}/backends/self_forcing"
SELF_FORCING_CKPT_ROOT="${CKPT_ROOT}"
```

This makes `temporalresidualkvquant` self-contained for code development.  The only
external runtime dependency is the checkpoint directory.

Recommended first experiment order:

```bash
bash scripts/self_forcing/run_random_hwq.sh
bash scripts/self_forcing/run_bf16.sh
bash scripts/self_forcing/run_int2_all.sh
bash scripts/self_forcing/run_packed_naive_hwq.sh
```

`packed-naive-int2/int4/int8` are real packed baselines.  They store uint8
packed low-bit codes plus per-block min/scale metadata and are decompressed on
cache read.  The older `naive-int2/int4` path remains a fake-quant quality
control because it returns BF16 tensors.

TRQ is selected with `trq-int2`, `trq-int4`, or `trq-int8`. Legacy
`hrq-*`/`s2pp-*` names resolve to the same codec, but new launchers should use
`trq-*`. Stable TRQ v1 accepts `identity` and `affine_channel`; RoPE remains an
explicit experiment and is not enabled by the production codec.

## Importance Top-K Policy

`headwise_mode=topk` replaces the random high-precision group with a fixed
per-layer policy loaded from `HEAD_IMPORTANCE_PATH`.

The packed-naive top-k launcher is:

```bash
bash scripts/self_forcing/run_head_importance_analysis.sh

HEAD_IMPORTANCE_PATH=assets/head_importance/top4_dmd_loss.json \
  bash scripts/self_forcing/run_packed_naive_topk_hwq.sh
```

It defaults to:

```text
top-k heads: packed-naive-int4
remaining heads: packed-naive-int2
top-k count: NUM_HIGH_PRECISION_HEADS, default 4
```

The policy file can contain explicit heads:

```json
{
  "num_heads": 12,
  "top_heads_by_layer": {
    "0": [1, 4, 7, 10],
    "1": [0, 2, 5, 11]
  }
}
```

or scores per layer / global head id.  The focused-forcing JSON-to-policy
selection logic is implemented in `trq.head_importance`; see
`docs/head_importance_topk.md` for the aggregation command, Python API, and
cross-machine path examples.
