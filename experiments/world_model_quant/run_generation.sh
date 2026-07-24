#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"

baseline="${1:?usage: run_generation.sh BASELINE smoke|full}"
stage="${2:?usage: run_generation.sh BASELINE smoke|full}"
require_baseline "${baseline}"
require_stage "${stage}"

gpu="${GPU:-0}"
if [[ "${stage}" == smoke ]]; then
  limit=1
else
  limit=10
fi
export CUDA_VISIBLE_DEVICES="${gpu}"

log_root="${REPO_ROOT}/results/world_model_quant/logs/${stage}"
mkdir -p "${log_root}"

case "${baseline}" in
  causal_forcing)
    prompts="$(prepare_prompt_subset "${limit}" "causal_${stage}")"
    output_root="${REPO_ROOT}/results/world_model_quant/causal_forcing/${stage}"
    for variant in "${VARIANTS[@]}"; do
      PROMPTS="${prompts}" \
      OUTPUT_ROOT="${output_root}" \
      NUM_OUTPUT_FRAMES="${CAUSAL_FRAMES:-21}" \
        bash "${REPO_ROOT}/integrations/causal_forcing/run_moviegen10.sh" "${variant}" \
        2>&1 | tee "${log_root}/causal_forcing_${variant}.log"
    done
    ;;
  longcat)
    output_root="${REPO_ROOT}/results/world_model_quant/longcat/${stage}"
    LIMIT="${limit}" OUTPUT_ROOT="${output_root}" \
      bash "${REPO_ROOT}/integrations/longcat_video/run_moviegen10.sh" bf16 prefix \
      2>&1 | tee "${log_root}/longcat_prefix_bf16.log"
    for variant in "${VARIANTS[@]}"; do
      LIMIT="${limit}" OUTPUT_ROOT="${output_root}" \
        bash "${REPO_ROOT}/integrations/longcat_video/run_moviegen10.sh" \
          "${variant}" continuation \
        2>&1 | tee "${log_root}/longcat_${variant}.log"
    done
    ;;
  hy_worldplay)
    scenes="${HY_SCENES:-${script_dir}/hy_scenes.example.json}"
    args=(
      --scenes "${scenes}"
      --experiment-id "action_control_${stage}"
      --limit-scenes "${HY_LIMIT_SCENES:-${limit}}"
    )
    if [[ "${stage}" == smoke ]]; then
      args+=(--action canonical)
    else
      args+=(--minimum-scenes "${HY_MIN_SCENES:-5}")
      args+=(--action turn_left --action turn_right --action forward --action backward)
    fi
    python "${script_dir}/run_hy_action_matrix.py" "${args[@]}" \
      2>&1 | tee "${log_root}/hy_worldplay_${stage}.log"
    ;;
esac
