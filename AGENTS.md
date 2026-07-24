# Repository Working Agreement

## Ownership

- `temporalresidualkvquant/src/trq/`: core TRQ algorithms and backend-neutral contracts.
- `temporalresidualkvquant/backends/self_forcing/`: current vendored Self-Forcing runtime.
- `integrations/`: baseline-specific launchers, adapters, patches, and smoke tests.
- `experiments/`: baseline-neutral experiment and evaluation orchestration.
- `references/`: read-only upstream snapshots; do not add method logic here.
- `forcing/`: legacy remote-runtime snapshots. Treat these as upstream code while
  their integrations are migrated to patch-based setup.
- `docs/project/`: current status, decisions, memory, and handoff.
- `docs/history/`: dated execution records.

## Execution Policy

This local checkout is for code review and adapter development. Do not download
model weights or run GPU experiments locally. After a coherent code change:

1. run CPU/static checks locally;
2. synchronize the same change to the remote code-server checkout;
3. verify checksums for maintained code paths;
4. run remote smoke tests before reporting the backend as usable.

Do not push to GitHub unless the user explicitly asks. Preserve remote-only
weights, outputs, environments, and upstream clones during synchronization.

## Change Discipline

- Create a checkpoint commit before broad path migrations.
- Use `git mv` for tracked moves and update every executable path in the same change.
- Keep upstream repositories pinned and express local modifications as patches.
- Never commit weights, generated videos, caches, or Python packaging artifacts.
- Read `docs/project/STATUS.md` and `docs/project/HANDOFF.md` before experiment work.
