# Self-Forcing Checkpoint Download

Large model files are kept locally under:

```text
temporalresidualkvquant/ckpts/Self-Forcing/
```

This directory is ignored by git, so checkpoints are never pushed to GitHub.

## Download Sources

The required model files come from two Hugging Face repositories:

- Wan base model: `Wan-AI/Wan2.1-T2V-1.3B`
- Self-Forcing DMD checkpoint: `gdhe17/Self-Forcing`

Reference pages:

- https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
- https://huggingface.co/gdhe17/Self-Forcing

## Set Up Another Machine

First make sure the code is up to date:

```bash
cd /mnt/workspace/caipeiliang/code/moweile/videoquant
git pull
git rev-parse HEAD
```

Then download the model files into `temporalresidualkvquant`:

```bash
cd /mnt/workspace/caipeiliang/code/moweile/videoquant/temporalresidualkvquant
mkdir -p ckpts/Self-Forcing

huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir ckpts/Self-Forcing/Wan2.1-T2V-1.3B

huggingface-cli download gdhe17/Self-Forcing \
  checkpoints/self_forcing_dmd.pt \
  --local-dir ckpts/Self-Forcing

mv ckpts/Self-Forcing/checkpoints/self_forcing_dmd.pt \
  ckpts/Self-Forcing/self_forcing_dmd.pt
rmdir ckpts/Self-Forcing/checkpoints
```

Expected local layout:

```text
temporalresidualkvquant/ckpts/Self-Forcing/
├── self_forcing_dmd.pt
└── Wan2.1-T2V-1.3B/
    ├── Wan2.1_VAE.pth
    ├── diffusion_pytorch_model.safetensors
    ├── models_t5_umt5-xxl-enc-bf16.pth
    └── google/umt5-xxl/
```

After this, run:

```bash
bash scripts/self_forcing/run_random_hwq.sh
```

## If Hugging Face Is Slow

You can also copy from an existing machine that already has the files:

```bash
rsync -a --info=progress2 \
  /data2/moweile-20251213/workspace/videoquant/temporalresidualkvquant/ckpts/Self-Forcing/ \
  USER@HOST:/mnt/workspace/caipeiliang/code/moweile/videoquant/temporalresidualkvquant/ckpts/Self-Forcing/
```

## Run With External Checkpoints

If the other machine already has a trusted shared checkpoint directory, you can
avoid copying into `temporalresidualkvquant/ckpts`:

```bash
SELF_FORCING_CKPT_ROOT=/path/to/ckpts/Self-Forcing \
  bash scripts/self_forcing/run_random_hwq.sh
```

That external directory must have the same layout:

```text
/path/to/ckpts/Self-Forcing/
├── self_forcing_dmd.pt
└── Wan2.1-T2V-1.3B/
```

For packed-naive importance top-k, set the policy path independently from the
checkpoint path:

```bash
SELF_FORCING_CKPT_ROOT=/path/to/ckpts/Self-Forcing \
HEAD_IMPORTANCE_PATH=/path/to/temporalresidualkvquant/assets/head_importance/top4_dmd_loss.json \
  bash scripts/self_forcing/run_packed_naive_topk_hwq.sh
```

If checkpoints live in an existing QVG checkout:

```bash
QVG_ROOT=/path/to/videoquant/Quant-VideoGen \
HEAD_IMPORTANCE_PATH=/path/to/videoquant/temporalresidualkvquant/assets/head_importance/top4_dmd_loss.json \
  bash scripts/self_forcing/run_packed_naive_topk_hwq.sh
```
