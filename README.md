# VideoQuant TRQ

This repository studies KV-cache quantization for autoregressive long-video and
action-conditioned world models. `temporalresidualkvquant/` is the method
workspace; baseline integration code is kept separate from upstream snapshots
and experiment outputs.

## Repository Map

```text
videoquant-trq/
├── temporalresidualkvquant/  # TRQ algorithms, tests, Self-Forcing backend
├── integrations/             # Causal, HY, LongCat, Rolling contracts/patches
├── experiments/              # baseline-neutral evaluation orchestration
├── references/               # read-only upstream snapshots
├── forcing/                  # legacy upstream/runtime snapshots
└── docs/
    ├── project/              # status, decisions, handoff, durable facts
    └── history/              # dated task and experiment records
```

The ownership rules and local/remote execution policy are in [AGENTS.md](AGENTS.md).

## Start Here

1. Read [current status](docs/project/STATUS.md) and [handoff](docs/project/HANDOFF.md).
2. Read [method documentation](temporalresidualkvquant/README.md).
3. Choose a baseline from [integrations](integrations/README.md).
4. Use [workspace structure](temporalresidualkvquant/docs/workspace_structure.md)
   when adding a backend or moving code.

## Local Validation

Local work is CPU/static only. Model weights and GPU experiments stay on the
remote runner.

```bash
cd temporalresidualkvquant
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Project Records

- `docs/project/STATUS.md`: current progress and open work.
- `docs/project/HANDOFF.md`: concise next-agent context.
- `docs/project/MEMORY.md`: stable project facts.
- `docs/project/DECISIONS.md`: architecture and protocol decisions.
- `docs/project/TASK_TEMPLATE.md`: template for a dated record.
- `docs/history/`: individual experiment, debug, and migration records.

Historical absolute paths in older records are evidence, not current commands.
