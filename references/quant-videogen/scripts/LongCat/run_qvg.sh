#!/bin/bash

prompt_source="text_to_video_from_file"
prompt_file=assets/t2v.txt
prompt_idx=1
seed=0
num_cond_frames=73
num_segments=10

#########################################################
# Quantization Configuration
#########################################################
# quant_type="triton-nstages-kmeans-int4"
quant_type="triton-nstages-kmeans-int2"
quant_block_size=64
cache_num_k_centroids=256
cache_num_v_centroids=256
kmeans_max_iters=100
num_prq_stages=1

quant_dir=${quant_type}_${quant_block_size}/kc_${cache_num_k_centroids}_vc_${cache_num_v_centroids}/nstages_${num_prq_stages}_iters_${kmeans_max_iters}
output_path=results/longcat/${quant_dir}
init_video_path=results/longcat/base/${prompt_idx}-${seed}.mp4

export PYTHONPATH=experiments/LongCat

torchrun --nproc_per_node=1 --standalone experiments/LongCat/run_long_t2v.py --checkpoint_dir=ckpts/LongCat-Video \
    --workload 480p_long_gen \
    --init_video_path ${init_video_path} \
    --output_dir ${output_path} \
    --num_segments ${num_segments} \
    --num_cond_frames ${num_cond_frames} \
    --seed ${seed} \
    --prompt_source ${prompt_source} \
    --prompt ${prompt_file} \
    --prompt_idx ${prompt_idx} \
    --quant_type ${quant_type} \
    --quant_block_size ${quant_block_size} \
    --cache_num_k_centroids ${cache_num_k_centroids} \
    --cache_num_v_centroids ${cache_num_v_centroids} \
    --kmeans_max_iters ${kmeans_max_iters} \
    --num_prq_stages ${num_prq_stages}
