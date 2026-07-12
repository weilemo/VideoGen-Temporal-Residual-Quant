#!/bin/bash

prompts_path=assets/t2v.txt
ckpt_id=official
local_attn_size=180
num_output_frames=180
ckpt_path=ckpts/Self-Forcing/self_forcing_dmd.pt

output_folder=results/selfforcing/bf16

echo "Running inference with checkpoint $ckpt_path and prompts from $prompts_path"
echo "Output will be saved to $output_folder"

export PYTHONPATH=../temporalresidualkvquant/src:experiments/Self-Forcing:.

DUMP_KV_LEVEL=0 torchrun --nproc_per_node=1 --standalone experiments/Self-Forcing/inference.py \
  --config_path experiments/Self-Forcing/configs/self_forcing_dmd.yaml \
  --checkpoint_path $ckpt_path \
  --data_path $prompts_path \
  --output_folder $output_folder \
  --num_samples 1 \
  --num_output_frames $num_output_frames \
  --local_attn_size $local_attn_size \
  --use_ema \
  --save_with_index \
  --quant_type none
