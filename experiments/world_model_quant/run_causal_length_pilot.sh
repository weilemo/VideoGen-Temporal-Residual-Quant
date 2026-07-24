#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
prompts="$(prepare_prompt_subset "${PILOT_PROMPTS:-3}" causal_length_pilot)"
read -r -a lengths <<< "${CAUSAL_LENGTHS:-21 42 84}"

for frames in "${lengths[@]}"; do
  output_root="${REPO_ROOT}/results/world_model_quant/causal_forcing/length_pilot/frames_${frames}"
  log_root="${REPO_ROOT}/results/world_model_quant/logs/length_pilot/frames_${frames}"
  mkdir -p "${log_root}"
  for variant in "${VARIANTS[@]}"; do
    PROMPTS="${prompts}" OUTPUT_ROOT="${output_root}" NUM_OUTPUT_FRAMES="${frames}" \
      bash "${REPO_ROOT}/integrations/causal_forcing/run_moviegen10.sh" "${variant}" \
      2>&1 | tee "${log_root}/${variant}.log"
  done
done
