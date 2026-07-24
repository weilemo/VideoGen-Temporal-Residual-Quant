#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"

baseline="${1:?usage: run_evaluation.sh BASELINE smoke|full}"
stage="${2:?usage: run_evaluation.sh BASELINE smoke|full}"
require_baseline "${baseline}"
require_stage "${stage}"
expected=10
[[ "${stage}" == smoke ]] && expected=1

case "${baseline}" in
  causal_forcing)
    root="${REPO_ROOT}/results/world_model_quant/causal_forcing/${stage}"
    python "${REPO_ROOT}/experiments/paired_quality/run_forcing_paired_metrics.py" \
      --baseline causal_forcing --root "${root}" --expected-videos "${expected}" \
      --device "${METRIC_DEVICE:-cuda}"
    PROMPT_FILE="$(prepare_prompt_subset "${expected}" "causal_${stage}")" \
    VIDEO_ROOT="${root}" EXPECTED_VIDEOS="${expected}" \
    OUTPUT_ROOT="${REPO_ROOT}/results/world_model_quant/vbench/causal_forcing_${stage}" \
      bash "${REPO_ROOT}/integrations/evaluation/run_vbench.sh" causal_forcing
    ;;
  longcat)
    root="${REPO_ROOT}/results/world_model_quant/longcat/${stage}"
    python "${REPO_ROOT}/experiments/paired_quality/run_forcing_paired_metrics.py" \
      --baseline longcat --root "${root}" --expected-videos "${expected}" \
      --start-frame "${LONGCAT_COND_FRAMES:-13}" --device "${METRIC_DEVICE:-cuda}"
    PROMPT_FILE="$(prepare_prompt_subset "${expected}" "longcat_${stage}")" \
    VIDEO_ROOT="${root}" EXPECTED_VIDEOS="${expected}" \
    OUTPUT_ROOT="${REPO_ROOT}/results/world_model_quant/vbench/longcat_${stage}" \
      bash "${REPO_ROOT}/integrations/evaluation/run_vbench.sh" longcat
    ;;
  hy_worldplay)
    root="${REPO_ROOT}/results/world_model_quant/hy_worldplay/action_control_${stage}"
    while IFS= read -r action_root; do
      python "${REPO_ROOT}/experiments/paired_quality/run_forcing_paired_metrics.py" \
        --baseline hy_worldplay --root "${action_root}" --expected-videos 1 \
        --device "${METRIC_DEVICE:-cuda}"
    done < <(find "${root}" -mindepth 2 -maxdepth 2 -type d | sort)
    if [[ "${stage}" == full ]]; then
      python "${script_dir}/evaluate_hy_actions.py" \
        --root "${root}" \
        --actions "${REPO_ROOT}/integrations/hy_worldplay/actions.json" \
        --output "${root}/action_metrics"
    fi
    ;;
esac
