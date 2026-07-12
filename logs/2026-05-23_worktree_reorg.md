# 2026-05-23 Git worktree reorganization

## Task

Separate concurrent `videoquant` research branches into independent Git worktrees and update project handoff documentation.

## Final layout

```text
/mnt/workspace/caipeiliang/code/moweile/
  videoquant/          # detached HEAD management checkout
  videoquant-main/     # main
  videoquant-prompt/   # HWQ_prompt_router
  videoquant-online/   # hwq_online_calibration
  videoquant-hrq/      # feature/hwq-residual-quant
```

## Branch roles

- `main`: stable project docs and baseline handoff.
- `HWQ_prompt_router`: Proposal 1, prompt router with multiple offline policies.
- `hwq_online_calibration`: Proposal 2, first-chunks online calibration.
- `feature/hwq-residual-quant`: HRQ residual quant backend.

## Checks

- All worktrees were clean after creation.
- `HWQ_prompt_router` and `hwq_online_calibration` were checked with `git merge-tree`; after rearranging adjacent insertions and removing branch-specific `STATUS.md` edits, the dry-run merge had no conflict markers.
- Local exclude ignores generated artifacts: `HeadWiseKVQuant/results/`, `HeadWiseKVQuant/tmp/`, `**/._*`, and `HeadWiseKVQuant/docs/*.pdf`.

## Operating rule

Agents should not develop in `/mnt/workspace/caipeiliang/code/moweile/videoquant`. They should enter the matching sibling worktree and keep each direction isolated.
