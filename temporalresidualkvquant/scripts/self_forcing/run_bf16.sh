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
output_folder="${OUTPUT_FOLDER:-${hwq_root}/outputs/self_forcing/bf16}"
attention_sink_frames="${ATTENTION_SINK_FRAMES:-0}"
runtime_args=()
if [ "${PROFILE_RUNTIME:-0}" = "1" ]; then
  runtime_args+=(--profile)
fi
if [ "${SAVE_ROLLOUT_LATENTS:-0}" = "1" ]; then
  runtime_args+=(--save_rollout_latents)
fi
if [ -n "${ROLLOUT_METRICS_DIR:-}" ]; then
  runtime_args+=(--rollout_metrics_dir "${ROLLOUT_METRICS_DIR}")
fi
if [ -n "${PROMPT_INDICES:-}" ]; then
  runtime_args+=(--prompt_indices "${PROMPT_INDICES}")
fi

echo "temporalresidualkvquant root: ${hwq_root}"
echo "Self-Forcing backend: ${self_forcing_root}"
echo "Self-Forcing ckpt root: ${ckpt_root}"
echo "Running Self-Forcing BF16 baseline"
echo "Output: ${output_folder}"

export PYTHONPATH="${hwq_root}/src:${self_forcing_root}:${PYTHONPATH:-}"
export SELF_FORCING_CKPT_ROOT="${ckpt_root}"

DUMP_KV_LEVEL="${DUMP_KV_LEVEL:-0}" torchrun --nproc_per_node=1 --standalone "${self_forcing_root}/inference.py" \
  --config_path "${self_forcing_root}/configs/self_forcing_dmd.yaml" \
  --checkpoint_path "${ckpt_path}" \
  --data_path "${prompts_path}" \
  --output_folder "${output_folder}" \
  --num_samples "${NUM_SAMPLES:-1}" \
  --seed "${SEED:-0}" \
  --num_output_frames "${num_output_frames}" \
  --local_attn_size "${local_attn_size}" \
  --attention_sink_frames "${attention_sink_frames}" \
  --use_ema \
  --save_with_index \
  "${runtime_args[@]}" \
  --quant_type none
