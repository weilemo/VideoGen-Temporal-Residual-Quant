#!/bin/bash

prompt_source="text_to_video_from_file"
prompt_file=assets/t2v.txt
prompt_idx=1
seed=0
num_cond_frames=73
num_segments=10

output_path=results/longcat/bf16
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
    --quant_type none
