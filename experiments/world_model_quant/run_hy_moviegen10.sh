#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
model_root="${MODEL_ROOT:-${HOME}/storage/models}"
result_root="${RESULT_ROOT:-${repo_root}/results/world_model_quant/hy_worldplay/moviegen10/canonical}"
log_root="${LOG_ROOT:-${repo_root}/results/world_model_quant/logs}"
ar_model="${model_root}/HY-WorldPlay/wan_transformer/diffusion_pytorch_model.safetensors"
distilled_model="${model_root}/HY-WorldPlay/wan_distilled_model/model.pt"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate videoquant
export PYTHONPATH="${repo_root}/temporalresidualkvquant/src:${repo_root}/Quant-VideoGen:${repo_root}/Quant-VideoGen/experiments/HY-WorldPlay${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${result_root}" "${log_root}"

while pgrep -f "hf download tencent/HY-WorldPlay" >/dev/null; do
  sleep 30
done
test -s "${ar_model}"
test -s "${distilled_model}"

variants=(bf16 trq_int4 trq_int2 naive_int4 naive_int2)
status_file="${result_root}/queue_status.tsv"
touch "${status_file}"
for variant in "${variants[@]}"; do
  output_dir="${result_root}/${variant}"
  mkdir -p "${output_dir}"
  count="$(find "${output_dir}" -type f -name '*.mp4' | wc -l)"
  if [ "${count}" -ge 10 ]; then
    printf '%s\t%s\t%s\n' "$(date -Iseconds)" "${variant}" skipped >> "${status_file}"
    continue
  fi
  printf '%s\t%s\t%s\n' "$(date -Iseconds)" "${variant}" running >> "${status_file}"
  python "${repo_root}/forcing/hy_worldplay/runner.py" \
    --variant "${variant}" \
    --action canonical \
    2>&1 | tee "${log_root}/hy_${variant}.log"
  count="$(find "${output_dir}" -type f -name '*.mp4' | wc -l)"
  if [ "${count}" -lt 10 ]; then
    printf '%s\t%s\t%s\n' "$(date -Iseconds)" "${variant}" "incomplete:${count}" >> "${status_file}"
    exit 1
  fi
  printf '%s\t%s\t%s\n' "$(date -Iseconds)" "${variant}" complete >> "${status_file}"
done

printf '%s\n' "HY-WorldPlay MovieGen10 queue complete"
