# Causal Forcing KV-cache experiment

This backend targets the official chunk-wise Causal Forcing model, not the
later Rolling Forcing long-video extension. The pinned upstream revision is
`1fc7bbc19a503c1bce80ecef08158b20e702f386`, and the checkpoint is
`zhuhz22/Causal-Forcing/chunkwise/causal_forcing.pt`.

Setup with the Hugging Face mirror:

```bash
bash experiments/world_model_quant/setup_causal_forcing.sh
```

Run the five paired MovieGen-10 modes:

```bash
for mode in bf16 trq_int4 trq_int2 naive_int4 naive_int2; do
  bash experiments/world_model_quant/run_causal_forcing_moviegen10.sh "$mode"
done
```

The adapter keeps Causal Forcing's BSHD cache slicing contract. BF16 uses the
native tensor cache; quantized modes store frame-aligned packed spans and
decode only the range requested by causal attention.
