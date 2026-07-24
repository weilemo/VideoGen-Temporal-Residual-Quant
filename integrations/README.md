# Baseline Integrations

This directory owns the executable contracts between TRQ and each video/world
model baseline. It contains launchers, thin adapters, pinned patches, and smoke
tests. Core quantization logic remains in `temporalresidualkvquant/src/trq/`.

| Baseline | Integration | Upstream source |
| --- | --- | --- |
| Causal Forcing | `causal_forcing/` | cloned on the remote runner, then patched |
| HY-WorldPlay | `hy_worldplay/` | `references/quant-videogen/experiments/HY-WorldPlay` |
| LongCat-Video-13B | `longcat_video/` | cloned on the remote runner, then patched |
| Rolling Forcing | `rolling_forcing/` | remote `forcing/rollingforcing` checkout |

Shared MovieGen10 prompts and VBench launchers live in `evaluation/`. Paired
PSNR, SSIM, and LPIPS evaluation lives in `experiments/paired_quality/`.
The reproducible Causal/LongCat/HY matrix, GPU allocation, gates, and one-shot
remote pull workflow live in `experiments/world_model_quant/`.

Do not put checkpoints or generated videos here. Those belong on the remote
runner under its model and result roots.
