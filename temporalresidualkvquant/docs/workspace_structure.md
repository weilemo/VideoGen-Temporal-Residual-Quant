# Workspace Structure

`temporalresidualkvquant` is the active method workspace inside the local
`videoquant-trq` repository.

## Current Local Layout

```text
/Users/moweile/Code/LAB/
  videoquant-trq/      # active unified repository
  qvg/                 # junior S2++ repository, reference only
```


```text
videoquant-trq/
├── temporalresidualkvquant/
│   ├── src/trq/                  # paper-facing quantization method
│   ├── backends/self_forcing/    # vendored Self-Forcing model and pipeline
│   ├── scripts/self_forcing/     # experiment launchers
│   ├── assets/t2v.txt            # default prompts
│   ├── ckpts/Self-Forcing/       # optional local checkpoints, ignored by git
│   ├── outputs/self_forcing/     # generated videos and logs, ignored by git
│   └── docs/
└── Quant-VideoGen/
    ├── quant_videogen/           # original QVG paper code, retained as reference
    └── experiments/Self-Forcing/ # original backend source, copied into HWQ
```

The split is intentional:

- `temporalresidualkvquant` owns head-wise quantization research code.
- `temporalresidualkvquant/backends/self_forcing` owns the active Self-Forcing backend used by HWQ launchers.
- `Quant-VideoGen` is retained as original source/reference.
- QVG's original `quant_videogen` package is kept for reference and baseline comparison.
- New head-wise policies should be added under `temporalresidualkvquant/src/trq/`, not under `Quant-VideoGen/quant_videogen/`.
- Temporal residual quantization is owned by `src/trq/real/trq.py`. Within the
  active HWQ package, `real/hrq.py` and `real/s2pp.py` are compatibility
  adapters only. QVG's original S2++ stays reference-only; do not develop a
  second active codec implementation under either legacy name.

Run the CPU smoke test from `temporalresidualkvquant` first:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Then follow [`getting_started.md`](getting_started.md) for Self-Forcing BF16
and TRQ generation.

Use `SELF_FORCING_CKPT_ROOT` when checkpoints are not under
`temporalresidualkvquant/ckpts/Self-Forcing`:

```bash
SELF_FORCING_CKPT_ROOT=/mnt/workspace/caipeiliang/code/moweile/videoquant/Quant-VideoGen/ckpts/Self-Forcing \
  bash scripts/self_forcing/run_random_hwq.sh
```
