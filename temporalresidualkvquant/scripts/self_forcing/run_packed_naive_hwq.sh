#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hwq_root="$(cd "${script_dir}/../.." && pwd)"
self_forcing_root="${SELF_FORCING_ROOT:-${hwq_root}/backends/self_forcing}"
if [ -n "${SELF_FORCING_CKPT_ROOT:-}" ]; then
  ckpt_root="${SELF_FORCING_CKPT_ROOT}"
elif [ -d "${hwq_root}/ckpts/Self-Forcing" ]; then
  ckpt_root="${hwq_root}/ckpts/Self-Forcing"
elif [ -n "${QVG_ROOT:-}" ]; then
  ckpt_root="${QVG_ROOT}/ckpts/Self-Forcing"
else
  ckpt_root="${hwq_root}/ckpts/Self-Forcing"
fi

prompts_path="${PROMPTS_PATH:-${hwq_root}/assets/t2v.txt}"
local_attn_size="${LOCAL_ATTN_SIZE:-180}"
num_output_frames="${NUM_OUTPUT_FRAMES:-180}"
ckpt_path="${CKPT_PATH:-${ckpt_root}/self_forcing_dmd.pt}"

quant_type="${QUANT_TYPE:-packed-naive-int2}"
cache_num_k_centroids="${CACHE_NUM_K_CENTROIDS:-256}"
cache_num_v_centroids="${CACHE_NUM_V_CENTROIDS:-256}"
kmeans_max_iters="${KMEANS_MAX_ITERS:-2}"
quant_block_size="${QUANT_BLOCK_SIZE:-64}"
num_prq_stages="${NUM_PRQ_STAGES:-1}"

headwise_mode="${HEADWISE_MODE:-random}"
headwise_seed="${HEADWISE_SEED:-0}"
num_high_precision_heads="${NUM_HIGH_PRECISION_HEADS:-4}"
high_precision_quant_type="${HIGH_PRECISION_QUANT_TYPE:-packed-naive-int4}"
low_precision_quant_type="${LOW_PRECISION_QUANT_TYPE:-packed-naive-int2}"
head_importance_path="${HEAD_IMPORTANCE_PATH:-}"
head_importance_score_direction="${HEAD_IMPORTANCE_SCORE_DIRECTION:-higher}"

if [ "${headwise_mode}" = "topk" ]; then
  if [ -z "${head_importance_path}" ]; then
    echo "HEAD_IMPORTANCE_PATH is required when HEADWISE_MODE=topk" >&2
    exit 1
  fi
  policy_name="$(basename "${head_importance_path}")"
  policy_name="${policy_name%.*}"
  quant_dir="topk_${policy_name}_hi_${num_high_precision_heads}_${high_precision_quant_type}_lo_${low_precision_quant_type}_${quant_block_size}/kc_${cache_num_k_centroids}_vc_${cache_num_v_centroids}_nstages_${num_prq_stages}"
else
  quant_dir="rhwq_seed_${headwise_seed}_hi_${num_high_precision_heads}_${high_precision_quant_type}_lo_${low_precision_quant_type}_${quant_block_size}/kc_${cache_num_k_centroids}_vc_${cache_num_v_centroids}_nstages_${num_prq_stages}"
fi
output_folder="${OUTPUT_FOLDER:-${hwq_root}/results/selfforcing/${quant_dir}}"

echo "temporalresidualkvquant root: ${hwq_root}"
echo "Self-Forcing backend: ${self_forcing_root}"
echo "Self-Forcing ckpt root: ${ckpt_root}"
echo "Running Self-Forcing packed-naive head-wise quant inference"
echo "Head-wise mode: ${headwise_mode}"
if [ -n "${head_importance_path}" ]; then
  echo "Head importance path: ${head_importance_path}"
fi
echo "Checkpoint: ${ckpt_path}"
echo "Prompts: ${prompts_path}"
echo "Output: ${output_folder}"

export PYTHONPATH="${hwq_root}/src:${self_forcing_root}:${PYTHONPATH:-}"
export SELF_FORCING_CKPT_ROOT="${ckpt_root}"

DUMP_KV_LEVEL="${DUMP_KV_LEVEL:-0}" torchrun --nproc_per_node=1 --standalone "${self_forcing_root}/inference.py" \
  --config_path "${self_forcing_root}/configs/self_forcing_dmd.yaml" \
  --checkpoint_path "${ckpt_path}" \
  --data_path "${prompts_path}" \
  --output_folder "${output_folder}" \
  --num_samples "${NUM_SAMPLES:-1}" \
  --num_output_frames "${num_output_frames}" \
  --local_attn_size "${local_attn_size}" \
  --use_ema \
  --save_with_index \
  --quant_type "${quant_type}" \
  --cache_num_k_centroids "${cache_num_k_centroids}" \
  --cache_num_v_centroids "${cache_num_v_centroids}" \
  --kmeans_max_iters "${kmeans_max_iters}" \
  --quant_block_size "${quant_block_size}" \
  --num_prq_stages "${num_prq_stages}" \
  --headwise_mode "${headwise_mode}" \
  --headwise_seed "${headwise_seed}" \
  --num_high_precision_heads "${num_high_precision_heads}" \
  --high_precision_quant_type "${high_precision_quant_type}" \
  --low_precision_quant_type "${low_precision_quant_type}" \
  --head_importance_path "${head_importance_path}" \
  --head_importance_score_direction "${head_importance_score_direction}"
