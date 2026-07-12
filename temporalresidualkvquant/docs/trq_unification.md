# TRQ Unification

`temporalresidualkvquant` is the canonical implementation workspace for temporal
residual KV-cache quantization. Stable capabilities migrated from the former
HRQ implementation and QVG S2++ are exposed through one public codec contract:
TRQ. The QVG repository itself remains an unchanged reference source.

For installation and experiment commands, start with
[`getting_started.md`](getting_started.md).

## Stable v1 boundary

- Codec: CPU PyTorch reference implementation in `src/trq/real/trq.py`.
- Input/output layout: `[B, H, S, D]`.
- Predictors: `identity` and per-channel `affine_channel`.
- Affine parameter shapes: `[D]`, `[H,D]`, or `[L,H,D]` with `layer_idx`.
- Quantization: packed 2/4/8-bit anchor and asymmetric zero-point residuals.
- K/V may override residual bit width independently.
- Head-wise Random/Top-K policies remain outside the codec and only choose
  head groups and quantization types.
- Predictor assets live under `assets/trq_predictors/`.

Every new state contains:

```text
format = "trq"
version = 1
layout = "BHSD"
shape, padded_dim, unit_lengths
anchor payload + quantization metadata
residual payload + scales + zero points
predictor kind, version, id, and required parameters
dependency = "previous_reconstruction"
```

Decoder inputs are self-contained. Missing affine parameters are an error;
there is no silent fallback to identity.

## Compatibility

- `hrq-int*` and `s2pp-int*` are accepted aliases for `trq-int*`.
- `trq.real.hrq` forwards to TRQ.
- `trq.real.s2pp` adapts stable identity/affine S2++ calls and `.npz` files.
- Legacy identity HRQ states and stable identity/affine S2++ states remain
  decodable by the TRQ decoder.

New experiments and serialized outputs must use the TRQ name. PRQ continues to
mean multi-stage K-Means residual quantization and is a separate codec.

## Experimental backlog

The following modes are intentionally excluded from v1:

- RoPE-aware affine prediction: pending controlled comparison against identity
  and affine on quality, motion metrics, latency, and actual state bytes.
- Cross-KV prediction: requires an explicit reconstructed-K dependency in the
  cache API before it can be decoded through single-cache reads.
- Tiny MLP, block VAR, AR(2), and error feedback: require encoder/decoder state
  parity tests and a demonstrated quality-memory benefit.
- Lloyd-Max residuals: require real 2/4-bit packing before memory claims are
  comparable.
- Triton decode: port only predictor/quantizer combinations that pass CPU parity;
  unsupported combinations must use an explicit fallback.

## Required gates

1. Encoder reconstruction equals standalone decoder reconstruction.
2. CPU and Triton outputs meet a documented numerical tolerance.
3. Test short spans, partial final units, non-divisible `D`, K/V bit overrides,
   mixed head groups, missing parameters, and cache offload/onload.
4. Report physical bytes for all tensors in the encoded state.
5. Run a two-prompt smoke test before MovieGenBench-32 or larger evaluation.

The offline distribution and chain-drift framework is documented in
`docs/trq_diagnostic_experiments.md`.
