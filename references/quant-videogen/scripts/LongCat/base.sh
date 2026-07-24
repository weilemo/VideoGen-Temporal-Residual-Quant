#!/bin/bash

prompt_source="text_to_video_from_file"
prompt_file=assets/t2v.txt
prompt_idx=1
seed=0

output_dirname=results/longcat/base

export PYTHONPATH=experiments/LongCat

torchrun --nproc_per_node=1 --standalone experiments/LongCat/run_long_t2v.py --checkpoint_dir=ckpts/LongCat-Video \
    --workload 480p_init \
    --output_dir ${output_dirname} \
    --seed ${seed} \
    --prompt_source ${prompt_source} \
    --prompt ${prompt_file} \
    --prompt_idx ${prompt_idx}
