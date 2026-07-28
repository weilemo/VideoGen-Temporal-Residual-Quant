#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"
configure_videoquant_runtime

causal_gpu="${CAUSAL_GPU:-6}"
longcat_gpu="${LONGCAT_GPU:-7}"
run_id="${RUN_ID:-b1_moviegen32_$(date +%Y%m%d_%H%M%S)}"
prompts="${PROMPTS_SOURCE:-${REPO_ROOT}/temporalresidualkvquant/assets/moviegenbench_resume_32.txt}"
index_root="${INDEX_ROOT:-${REPO_ROOT}/results/world_model_quant/indexes/${run_id}}"
eval_root="${EVAL_ROOT:-${REPO_ROOT}/results/world_model_quant/evaluation/${run_id}}"
status_root="${REPO_ROOT}/results/world_model_quant/orchestration/${run_id}_evaluation"
causal_legacy="${CAUSAL_LEGACY_ROOT:?Set CAUSAL_LEGACY_ROOT to the original MovieGen10 result root}"
causal_b1="${CAUSAL_B1_ROOT:?Set CAUSAL_B1_ROOT to the B1 Causal result root}"
longcat_legacy="${LONGCAT_LEGACY_ROOT:?Set LONGCAT_LEGACY_ROOT to the original MovieGen10 result root}"
longcat_b1="${LONGCAT_B1_ROOT:?Set LONGCAT_B1_ROOT to the B1 LongCat result root}"
hy_root="${HY_ROOT:-}"
run_hy="${RUN_HY:-1}"
run_vbench="${RUN_VBENCH:-1}"
dry_run="${DRY_RUN:-0}"

[[ "${causal_gpu}" != "${longcat_gpu}" ]] || {
  echo "CAUSAL_GPU and LONGCAT_GPU must differ" >&2
  exit 2
}
[[ -s "${prompts}" ]] || { echo "missing prompt file: ${prompts}" >&2; exit 2; }
if [[ "${run_hy}" == 1 && -z "${hy_root}" ]]; then
  echo "Set HY_ROOT to the HY holdout action-control result root, or RUN_HY=0" >&2
  exit 2
fi

mkdir -p "${status_root}" "${eval_root}"

manifest_command() {
  local baseline="$1"
  local legacy="$2"
  local b1="$3"
  python "${script_dir}/prepare_moviegen32_manifest.py" \
    --baseline "${baseline}" --prompts "${prompts}" --expected-prompts 32 \
    --source-root "legacy=${legacy}" --source-root "b1=${b1}" \
    --output-root "${index_root}/${baseline}"
}

if [[ "${dry_run}" == 1 ]]; then
  cat <<EOF
RUN_ID=${run_id}
GPU ${causal_gpu}: Causal MovieGen32 paired metrics + VBench, then HY holdout metrics
GPU ${longcat_gpu}: LongCat MovieGen32 paired metrics + VBench
CAUSAL sources: ${causal_legacy} + ${causal_b1}
LONGCAT sources: ${longcat_legacy} + ${longcat_b1}
HY_ROOT=${hy_root:-disabled}
INDEX_ROOT=${index_root}
EVAL_ROOT=${eval_root}
No manifests, metrics, or GPU jobs were started.
EOF
  exit 0
fi

manifest_command causal_forcing "${causal_legacy}" "${causal_b1}"
manifest_command longcat "${longcat_legacy}" "${longcat_b1}"

record_stage() {
  local name="$1"
  shift
  if [[ -f "${status_root}/${name}.done" ]]; then
    echo "skip completed stage: ${name}"
    return 0
  fi
  printf 'running\n' > "${status_root}/${name}.status"
  if "$@" > "${status_root}/${name}.log" 2>&1; then
    printf 'done\n' > "${status_root}/${name}.status"
    touch "${status_root}/${name}.done"
    return 0
  fi
  printf 'failed\n' > "${status_root}/${name}.status"
  touch "${status_root}/${name}.failed"
  return 1
}

run_paired() {
  local baseline="$1"
  local root="$2"
  local output="$3"
  shift 3
  python "${REPO_ROOT}/experiments/paired_quality/run_forcing_paired_metrics.py" \
    --baseline "${baseline}" --root "${root}" --output-dir "${output}" \
    --expected-videos 32 --device cuda --bootstrap-resamples 2000 "$@"
}

run_vbench_for() {
  local baseline="$1"
  local root="$2"
  local output="$3"
  [[ "${run_vbench}" == 1 ]] || return 0
  PROMPT_FILE="${prompts}" VIDEO_ROOT="${root}" EXPECTED_VIDEOS=32 \
    OUTPUT_ROOT="${output}" \
    bash "${REPO_ROOT}/integrations/evaluation/run_vbench.sh" "${baseline}"
}

causal_hy_lane() {
  export CUDA_VISIBLE_DEVICES="${causal_gpu}"
  record_stage causal_paired run_paired causal_forcing \
    "${index_root}/causal_forcing" "${eval_root}/causal_forcing/paired_metrics"
  record_stage causal_vbench run_vbench_for causal_forcing \
    "${index_root}/causal_forcing" "${eval_root}/causal_forcing/vbench"
  if [[ "${run_hy}" == 1 ]]; then
    local action_root
    while IFS= read -r action_root; do
      record_stage "hy_paired_$(basename "$(dirname "${action_root}")")_$(basename "${action_root}")" \
        python "${REPO_ROOT}/experiments/paired_quality/run_forcing_paired_metrics.py" \
          --baseline hy_worldplay --root "${action_root}" --expected-videos 1 \
          --device cuda --bootstrap-resamples 2000
    done < <(find "${hy_root}" -mindepth 2 -maxdepth 2 -type d | sort)
    record_stage hy_action_metrics python "${script_dir}/evaluate_hy_actions.py" \
      --root "${hy_root}" \
      --actions "${REPO_ROOT}/integrations/hy_worldplay/actions.json" \
      --output "${eval_root}/hy_worldplay/action_metrics"
  fi
}

longcat_lane() {
  export CUDA_VISIBLE_DEVICES="${longcat_gpu}"
  record_stage longcat_paired run_paired longcat \
    "${index_root}/longcat" "${eval_root}/longcat/paired_metrics" \
    --start-frame "${LONGCAT_COND_FRAMES:-13}"
  record_stage longcat_vbench run_vbench_for longcat \
    "${index_root}/longcat" "${eval_root}/longcat/vbench"
}

set +e
causal_hy_lane &
causal_pid=$!
longcat_lane &
longcat_pid=$!
wait "${causal_pid}"
causal_status=$?
wait "${longcat_pid}"
longcat_status=$?
set -e

printf 'causal_hy=%s longcat=%s\n' "${causal_status}" "${longcat_status}" |
  tee "${status_root}/summary.txt"
if [[ "${causal_status}" -eq 0 && "${longcat_status}" -eq 0 ]]; then
  touch "${status_root}/evaluation.done"
  echo "B1 automatic evaluation complete: ${run_id}"
  exit 0
fi
touch "${status_root}/evaluation.partial"
exit 1
