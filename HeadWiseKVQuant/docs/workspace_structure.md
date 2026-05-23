# Workspace Structure

`HeadWiseKVQuant` should be treated as the main research workspace inside each branch-specific worktree.

## Repository Worktrees

The parent `videoquant` checkout is now only a detached-HEAD management entrypoint. Use one sibling worktree per research direction:

```text
/mnt/workspace/caipeiliang/code/moweile/
  videoquant/          # worktree management only
  videoquant-main/     # main
  videoquant-prompt/   # HWQ_prompt_router
  videoquant-online/   # hwq_online_calibration
  videoquant-hrq/      # feature/hwq-residual-quant
```

Do not develop by repeatedly switching branches in `videoquant/`; enter the matching `videoquant-*` directory first.


```text
videoquant/
├── HeadWiseKVQuant/
│   ├── src/hwq/                  # paper-facing quantization method
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

- `HeadWiseKVQuant` owns head-wise quantization research code.
- `HeadWiseKVQuant/backends/self_forcing` owns the active Self-Forcing backend used by HWQ launchers.
- `Quant-VideoGen` is retained as original source/reference.
- QVG's original `quant_videogen` package is kept for reference and baseline comparison.
- New head-wise policies should be added under `HeadWiseKVQuant/src/hwq/`, not under `Quant-VideoGen/quant_videogen/`.

Run the current smoke test from `HeadWiseKVQuant`:

```bash
bash scripts/self_forcing/run_random_hwq.sh
```

Use `SELF_FORCING_CKPT_ROOT` when checkpoints are not under
`HeadWiseKVQuant/ckpts/Self-Forcing`:

```bash
SELF_FORCING_CKPT_ROOT=/mnt/workspace/caipeiliang/code/moweile/videoquant/Quant-VideoGen/ckpts/Self-Forcing \
  bash scripts/self_forcing/run_random_hwq.sh
```
