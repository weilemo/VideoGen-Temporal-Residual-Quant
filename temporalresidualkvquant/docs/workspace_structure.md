# Workspace Structure

The repository separates method code, baseline glue, experiment protocols, and
upstream provenance so a backend can change without creating a second TRQ
implementation.

```text
videoquant-trq/
├── temporalresidualkvquant/
│   ├── src/trq/                 # sole owner of TRQ algorithms
│   ├── src/trq/backends/        # backend-neutral runtime adapters
│   ├── backends/self_forcing/   # vendored Self-Forcing runtime
│   ├── scripts/                 # method and Self-Forcing launchers
│   └── tests/
├── integrations/
│   ├── causal_forcing/          # setup patch and MovieGen10 launcher
│   ├── hy_worldplay/            # action config, runner, and tests
│   ├── longcat_video/           # clone/setup patch and launcher
│   ├── rolling_forcing/         # MovieGen10 launcher
│   └── evaluation/              # shared prompts and VBench runner
├── experiments/
│   └── paired_quality/           # PSNR, SSIM, and LPIPS orchestration
├── references/
│   ├── quant-videogen/           # read-only QVG snapshot
│   └── focused-forcing-code/     # read-only forcing snapshots
├── forcing/                      # legacy runtime snapshots pending migration
└── docs/
```

## Placement Rules

- Put quantizers, cache data structures, policies, and diagnostics in
  `temporalresidualkvquant/src/trq/`.
- Put a baseline's argument mapping, monkey patch, source patch, and launcher in
  `integrations/<baseline>/`.
- Put protocols shared by multiple baselines in `experiments/`.
- Keep upstream snapshots unchanged under `references/`; record necessary
  upstream edits as patches under `integrations/`.
- Keep checkpoints and generated results outside tracked source directories.

Self-Forcing remains vendored under `temporalresidualkvquant/backends/` for now.
The other large upstream runtimes are installed on the remote GPU runner and
are connected through the integration contracts.

## Validation Order

```bash
cd temporalresidualkvquant
python -m unittest discover -s tests -v
```

Then run the selected integration's dry run or smoke test remotely. A backend is
not considered ready merely because its adapter imports locally.
