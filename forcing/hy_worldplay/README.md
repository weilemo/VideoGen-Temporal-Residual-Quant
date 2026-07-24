# HY-WorldPlay quantization experiments

This directory is the experiment adapter for HY-WorldPlay. It keeps the
upstream implementation in `../qvg/Quant-VideoGen` and records only the
configuration, action controls, manifests, and evaluation protocol here.

## Why an adapter

The BF16 and quantized runs must differ only in the cache quantizer. The
upstream example scripts use different memory windows and rollout lengths, so
they are useful demos but not a controlled quantization comparison.

This adapter fixes the following variables across variants:

- prompt, conditioning image, seed, and action sequence;
- 48 memory frames, 44 recent context frames, and 4 predicted latent frames;
- 12 rollout chunks and one GPU process.

The executable matrix keeps the rollout fixed and changes only the KV-cache
codec:

| Variant | Upstream quant type | Purpose |
| --- | --- | --- |
| `bf16` | `none` | reference trajectory |
| `trq_int4` | `trq-int4` | temporal-residual packed INT4 |
| `trq_int2` | `trq-int2` | temporal-residual packed INT2 |
| `naive_int4` | `packed-naive-int4` | non-residual packed INT4 control |
| `naive_int2` | `packed-naive-int2` | non-residual packed INT2 control |

The HY pipeline stores each layer's K/V cache in frame-aligned BHSD
`ChunkedKVCache` objects. Once a completed frame range leaves the recent
context window, `quantize_kv_cache()` encodes that range with
`compress_kv_cache()` and replaces it with a packed span. Attention reads
decode only the requested causal range, so quantized runs use the packed cache
rather than an auxiliary quality-only simulation.

## Usage

From the repository root:

```bash
python forcing/hy_worldplay/runner.py --matrix --dry-run
python forcing/hy_worldplay/runner.py --variant bf16 --action canonical
python forcing/hy_worldplay/runner.py --variant trq_int2 --action turn_left
```

Set `HY_WORLDPLAY_SOURCE` if Quant-VideoGen is not located at the default
sibling path:

```bash
HY_WORLDPLAY_SOURCE=/path/to/Quant-VideoGen \+  python forcing/hy_worldplay/runner.py --matrix --dry-run
```

Real runs require these upstream assets:

- `assets/hyworld.png`
- `ckpts/HY-WorldPlay/wan_transformer`
- `ckpts/HY-WorldPlay/wan_distilled_model/model.pt`

Each real invocation writes a `run_manifest.json` beside its output. The
manifest is the source of truth for pairing outputs; directory names alone
must not be used to infer experimental settings.

## Action-controllability protocol

`actions.json` defines a canonical mixed-action rollout and two
counterfactual pairs. For every action ID, run all quantization variants with
the same seed. Evaluate:

1. visual/trajectory fidelity against BF16 for the same action;
2. whether left-vs-right and forward-vs-backward branches remain separated;
3. long-horizon degradation after each action boundary.

A quantized rollout is not action-controllable merely because it moves. It
must preserve the direction and timing of the BF16 response while remaining
visually stable.
