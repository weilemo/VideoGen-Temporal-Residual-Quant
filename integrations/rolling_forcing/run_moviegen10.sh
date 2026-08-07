#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
model_root="${MODEL_ROOT:-${HOME}/storage/models}"
result_root="${RESULT_ROOT:-${repo_root}/results/world_model_quant/rollingforcing/moviegen10}"
log_root="${LOG_ROOT:-${repo_root}/results/world_model_quant/logs}"
checkpoint="${model_root}/RollingForcing/checkpoints/rolling_forcing_dmd.pt"
prompt_file="${repo_root}/integrations/evaluation/moviegen10.txt"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate videoquant
export WAN_MODEL_ROOT="${model_root}/Wan2.1-T2V-1.3B"
export PYTHONPATH="${repo_root}/temporalresidualkvquant/src:${repo_root}/forcing/rollingforcing${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${result_root}" "${log_root}"

while pgrep -f "hf download TencentARC/RollingForcing" >/dev/null; do
  sleep 30
done
test -s "${checkpoint}"
test -s "${WAN_MODEL_ROOT}/Wan2.1_VAE.pth"

variants=(bf16 trq_int4 trq_int2 naive_int4 naive_int2)
declare -A quant_types=(
  [bf16]=none
  [trq_int4]=trq-int4
  [trq_int2]=trq-int2
  [naive_int4]=packed-naive-int4
  [naive_int2]=packed-naive-int2
)

status_file="${result_root}/queue_status.tsv"
touch "${status_file}"
for variant in "${variants[@]}"; do
  output_dir="${result_root}/${variant}"
  mkdir -p "${output_dir}"
  count="$(find "${output_dir}" -maxdepth 1 -name '*.mp4' | wc -l)"
  if [ "${count}" -ge 10 ]; then
    printf '%s\t%s\t%s\n' "$(date -Iseconds)" "${variant}" skipped >> "${status_file}"
    continue
  fi
  printf '%s\t%s\t%s\n' "$(date -Iseconds)" "${variant}" running >> "${status_file}"
  python "${repo_root}/forcing/rollingforcing/inference.py" \
    --prompt_path "${prompt_file}" \
    --output_path "${output_dir}" \
    --checkpoint_path "${checkpoint}" \
    --kv_quant_type "${quant_types[${variant}]}" \
    --kv_quant_block_size 64 \
    --trq_anchor_bits 4 \
    --trq_predictor_stride 1560 \
    --trq_predictor_mode identity \
    --seed 0 \
    --num_samples 1 \
    --num_latent_frames 126 \
    2>&1 | tee "${log_root}/rolling_${variant}.log"
  printf '%s\t%s\t%s\n' "$(date -Iseconds)" "${variant}" complete >> "${status_file}"
done

printf '%s\n' "Rolling Forcing MovieGen10 queue complete"
