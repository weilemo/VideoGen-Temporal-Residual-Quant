#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"
configure_videoquant_runtime

baseline="${1:?usage: run_generation.sh BASELINE smoke|full}"
stage="${2:?usage: run_generation.sh BASELINE smoke|full}"
require_baseline "${baseline}"
require_stage "${stage}"

gpu="${GPU:-0}"
if [[ "${stage}" == smoke ]]; then
  default_limit=1
else
  default_limit=10
fi
limit="${LIMIT:-${default_limit}}"
start_index="${START_INDEX:-0}"
run_id="${RUN_ID:-${stage}}"
[[ "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "invalid RUN_ID: ${run_id}" >&2
  exit 2
}
export CUDA_VISIBLE_DEVICES="${gpu}"

log_root="${REPO_ROOT}/results/world_model_quant/logs/${run_id}"
mkdir -p "${log_root}"

case "${baseline}" in
  causal_forcing)
    output_root="${REPO_ROOT}/results/world_model_quant/causal_forcing/${run_id}"
    for variant in "${VARIANTS[@]}"; do
      prompts="$(prepare_missing_causal_prompts \
        "${start_index}" "${limit}" \
        "causal_${run_id}_${variant}_${start_index}_${limit}" \
        "${output_root}/${variant}")"
      if [[ ! -s "${prompts}" ]]; then
        printf 'skip complete Causal generation: variant=%s start=%s count=%s\n' \
          "${variant}" "${start_index}" "${limit}"
        continue
      fi
      PROMPTS="${prompts}" \
      OUTPUT_ROOT="${output_root}" \
      NUM_OUTPUT_FRAMES="${CAUSAL_FRAMES:-21}" \
        bash "${REPO_ROOT}/integrations/causal_forcing/run_moviegen10.sh" "${variant}" \
        2>&1 | tee "${log_root}/causal_forcing_${variant}_${start_index}_${limit}.log"
    done
    ;;
  longcat)
    output_root="${REPO_ROOT}/results/world_model_quant/longcat/${run_id}"
    LIMIT="${limit}" START_INDEX="${start_index}" OUTPUT_ROOT="${output_root}" \
      bash "${REPO_ROOT}/integrations/longcat_video/run_moviegen10.sh" bf16 prefix \
      2>&1 | tee "${log_root}/longcat_prefix_bf16_${start_index}_${limit}.log"
    for variant in "${VARIANTS[@]}"; do
      LIMIT="${limit}" START_INDEX="${start_index}" OUTPUT_ROOT="${output_root}" \
        bash "${REPO_ROOT}/integrations/longcat_video/run_moviegen10.sh" \
          "${variant}" continuation \
        2>&1 | tee "${log_root}/longcat_${variant}_${start_index}_${limit}.log"
    done
    ;;
  hy_worldplay)
    scenes="${HY_SCENES:-${script_dir}/hy_scenes.example.json}"
    args=(
      --scenes "${scenes}"
      --experiment-id "action_control_${run_id}"
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
