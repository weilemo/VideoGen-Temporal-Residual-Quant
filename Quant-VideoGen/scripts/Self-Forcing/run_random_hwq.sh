#!/bin/bash

prompts_path=assets/t2v.txt
local_attn_size=180
num_output_frames=180
ckpt_path=ckpts/Self-Forcing/self_forcing_dmd.pt

#########################################################
# Base Quantization Configuration
#########################################################
quant_type="triton-nstages-kmeans-int2"
cache_num_k_centroids=256
cache_num_v_centroids=256
kmeans_max_iters=2
quant_block_size=64
num_prq_stages=1

#########################################################
# Random Head-Wise Quantization Configuration
#########################################################
headwise_mode="random"
headwise_seed=0
num_high_precision_heads=4
high_precision_quant_type="triton-nstages-kmeans-int4"
low_precision_quant_type="triton-nstages-kmeans-int2"

quant_dir="rhwq_seed_${headwise_seed}_hi_${num_high_precision_heads}_${high_precision_quant_type}_lo_${low_precision_quant_type}_${quant_block_size}/kc_${cache_num_k_centroids}_vc_${cache_num_v_centroids}_nstages_${num_prq_stages}"
output_folder=results/selfforcing/${quant_dir}

echo "Running random head-wise quant inference with checkpoint $ckpt_path and prompts from $prompts_path"
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
  --quant_type $quant_type \
  --cache_num_k_centroids $cache_num_k_centroids \
  --cache_num_v_centroids $cache_num_v_centroids \
  --kmeans_max_iters $kmeans_max_iters \
  --quant_block_size $quant_block_size \
  --num_prq_stages $num_prq_stages \
  --headwise_mode $headwise_mode \
  --headwise_seed $headwise_seed \
  --num_high_precision_heads $num_high_precision_heads \
  --high_precision_quant_type $high_precision_quant_type \
  --low_precision_quant_type $low_precision_quant_type
