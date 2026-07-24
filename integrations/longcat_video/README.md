# LongCat-Video 13.6B KV-cache experiment

LongCat's text-to-video path does not use a persistent KV cache. Therefore the
quantization baseline targets the official **video continuation** path:

1. Generate one BF16 T2V prefix per MovieGen prompt.
2. Reuse that exact prefix for BF16, TRQ INT4/INT2, and packed-naive INT4/INT2
   continuation runs.
3. Reset the continuation seed to `42 + prompt_index` for every precision.
4. Compare quantized continuations to BF16 with the paired metric framework and
   evaluate each precision independently with VBench.

Setup and run:

```bash
bash integrations/longcat_video/download.sh \
  "$HOME/storage/models/LongCat-Video" forcing/longcatvideo

bash integrations/longcat_video/run_moviegen10.sh bf16 prefix
for mode in bf16 trq_int4 trq_int2 naive_int4 naive_int2; do
  bash integrations/longcat_video/run_moviegen10.sh "$mode" continuation
done

python experiments/paired_quality/run_forcing_paired_metrics.py \
  --baseline longcat \
  --root results/world_model_quant/longcat/moviegen10 \
  --expected-videos 10
```

VBench remains the quality metric; PSNR/SSIM/LPIPS measure drift from the
same-seed BF16 continuation.
