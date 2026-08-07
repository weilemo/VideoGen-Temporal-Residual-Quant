# Paired PSNR / SSIM / LPIPS

This framework measures **quantization trajectory fidelity**. Each quantized
video must be generated from the same prompt, seed, frame count, resolution,
sampler, and scheduler as its BF16 reference. The BF16 video is not ground
truth, so these metrics do not replace VBench.

Expected directory layout for each baseline:

```text
ROOT/
  bf16/
  trq_int4/
  trq_int2/
  naive_int4/
  naive_int2/
```

Video names must begin with `PROMPT_INDEX-SAMPLE_INDEX` or
`PROMPT_INDEX_SAMPLE_INDEX`. Use `--start-frame 13` for LongCat continuation so
the shared conditioning prefix is not counted as quantization fidelity. Run the
legacy two-forcing wrapper with:

```bash
pip install -e 'temporalresidualkvquant[analysis]'
bash experiments/paired_quality/run_both_forcing_metrics.sh \
  /path/to/selfforcing/moviegen10 \
  /path/to/rollingforcing/moviegen10 \
  10
```

Each baseline receives per-variant JSON plus `summary.json` and `summary.csv`.
The command fails on missing/extra pairs, shape mismatches, missing LPIPS, an
unexpected video count, or invalid aggregate values.

The Causal Forcing, LongCat, and HY-WorldPlay orchestration is documented in
`experiments/world_model_quant/README.md`.
