#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"
configure_videoquant_runtime

mode="${1:-orchestrator}"
gpu_a="${GPU_A:-6}"
gpu_b="${GPU_B:-7}"
run_id="${RUN_ID:-b2_moviegen128_$(date +%Y%m%d)}"
prompts_source="${PROMPTS_SOURCE:-${REPO_ROOT}/temporalresidualkvquant/assets/moviegenbench_resume_128.txt}"
prefix_source="${PREFIX_SOURCE:-${REPO_ROOT}/temporalresidualkvquant/assets/moviegenbench_resume_32.txt}"
start_index="${START_INDEX:-32}"
total_count="${LIMIT:-96}"
shard_a_count="${SHARD_A_COUNT:-48}"
causal_frames="${CAUSAL_FRAMES:-84}"
dry_run="${DRY_RUN:-0}"
min_free_gb="${MIN_FREE_GB:-100}"
status_root="${WORLD_MODEL_RESULTS_ROOT}/orchestration/${run_id}_generation"
log_root="${WORLD_MODEL_RESULTS_ROOT}/logs/${run_id}/orchestrator"

[[ "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid RUN_ID: ${run_id}" >&2; exit 2; }
[[ "${gpu_a}" != "${gpu_b}" ]] || { echo "GPU_A and GPU_B must differ" >&2; exit 2; }
[[ "${start_index}" =~ ^[0-9]+$ && "${total_count}" =~ ^[1-9][0-9]*$ ]] || {
  echo "START_INDEX and LIMIT must be non-negative/positive integers" >&2
  exit 2
}
[[ "${shard_a_count}" =~ ^[1-9][0-9]*$ && "${shard_a_count}" -lt "${total_count}" ]] || {
  echo "SHARD_A_COUNT must be positive and smaller than LIMIT" >&2
  exit 2
}
[[ "$((start_index + total_count))" -eq 128 ]] || {
  echo "B2 must cover the canonical missing slice ending at index 127" >&2
  exit 2
}

python "${script_dir}/validate_moviegen_prompts.py" \
  --prompts "${prompts_source}" --expected 128 \
  --prefix "${prefix_source}" --prefix-count 32 >/dev/null

shard_b_start=$((start_index + shard_a_count))
shard_b_count=$((total_count - shard_a_count))
mkdir -p "${status_root}" "${log_root}"

if [[ "${dry_run}" == 1 ]]; then
  cat <<EOF
RUN_ID=${run_id}
PROMPTS_SOURCE=${prompts_source}
Phase A Causal: GPU ${gpu_a} indices ${start_index}-$((shard_b_start - 1)); GPU ${gpu_b} indices ${shard_b_start}-127
Phase B LongCat: GPU ${gpu_a} indices ${start_index}-$((shard_b_start - 1)); GPU ${gpu_b} indices ${shard_b_start}-127
Five variants: ${VARIANTS[*]}
CAUSAL_FRAMES=${causal_frames}
MIN_FREE_GB=${min_free_gb}
WORLD_MODEL_RESULTS_ROOT=${WORLD_MODEL_RESULTS_ROOT}
Existing decodable outputs in this RUN_ID are reused.
No GPU commands executed.
EOF
  exit 0
fi

if [[ "${mode}" == orchestrator ]]; then
  free_kb="$(df -Pk "${WORLD_MODEL_RESULTS_ROOT}" | awk 'NR == 2 {print $4}')"
  required_kb=$((min_free_gb * 1024 * 1024))
  [[ "${free_kb}" -ge "${required_kb}" ]] || {
    echo "insufficient free space: need at least ${min_free_gb} GiB" >&2
    exit 2
  }

  exec 9>"${status_root}/orchestrator.lock"
  flock -n 9 || { echo "another B2 orchestrator owns ${status_root}" >&2; exit 2; }
fi

is_video() {
  ffprobe -v error -show_entries stream=codec_type -of csv=p=0 "$1" 2>/dev/null |
    grep -qx video
}

validate_causal_shard() {
  local root="$1"
  local shard_start="$2"
  local shard_count="$3"
  local variant missing
  for variant in "${VARIANTS[@]}"; do
    missing="${status_root}/missing_causal_${variant}_${shard_start}_${shard_count}.txt"
    python "${script_dir}/prepare_causal_prompts.py" \
      --prompts "${prompts_source}" --start "${shard_start}" --count "${shard_count}" \
      --output-dir "${root}/${variant}" --destination "${missing}" >/dev/null
    [[ ! -s "${missing}" ]] || return 1
  done
}

validate_longcat_shard() {
  local root="$1"
  local shard_start="$2"
  local shard_count="$3"
  local directory index
  for directory in prefix_bf16 "${VARIANTS[@]}"; do
    for ((index=shard_start; index < shard_start + shard_count; index++)); do
      is_video "${root}/${directory}/${index}-0.mp4" || return 1
    done
  done
}

record_stage() {
  local name="$1"
  shift
  rm -f "${status_root}/${name}.done" "${status_root}/${name}.failed"
  printf 'running\n' > "${status_root}/${name}.status"
  set +e
  "$@" > "${log_root}/${name}.log" 2>&1
  local exit_code=$?
  set -e
  printf '%s\n' "${exit_code}" > "${status_root}/${name}.exit"
  if [[ "${exit_code}" -eq 0 ]]; then
    printf 'done\n' > "${status_root}/${name}.status"
    touch "${status_root}/${name}.done"
    return 0
  fi
  printf 'failed\n' > "${status_root}/${name}.status"
  touch "${status_root}/${name}.failed"
  return "${exit_code}"
}

run_lane() {
  local baseline="$1"
  local lane_gpu="$2"
  local shard_start="$3"
  local shard_count="$4"
  local name="${baseline}_${shard_start}_$((shard_start + shard_count - 1))"
  local result_root
  result_root="${WORLD_MODEL_RESULTS_ROOT}/${baseline}/${run_id}"
  [[ "${baseline}" != longcat ]] || result_root="${WORLD_MODEL_RESULTS_ROOT}/longcat/${run_id}"

  if ! record_stage "${name}" env \
    RUN_ID="${run_id}" GPU="${lane_gpu}" PROMPTS_SOURCE="${prompts_source}" \
    WORLD_MODEL_RESULTS_ROOT="${WORLD_MODEL_RESULTS_ROOT}" \
    START_INDEX="${shard_start}" LIMIT="${shard_count}" CAUSAL_FRAMES="${causal_frames}" \
    bash "${script_dir}/run_generation.sh" "${baseline}" full; then
    return 1
  fi
  if [[ "${baseline}" == causal_forcing ]]; then
    validate_causal_shard "${result_root}" "${shard_start}" "${shard_count}"
  else
    validate_longcat_shard "${result_root}" "${shard_start}" "${shard_count}"
  fi || {
    printf 'failed:validation\n' > "${status_root}/${name}.status"
    rm -f "${status_root}/${name}.done"
    touch "${status_root}/${name}.failed"
    return 1
  }
}

if [[ "${mode}" == lane ]]; then
  [[ "$#" -eq 5 ]] || { echo "lane mode requires BASELINE GPU START COUNT" >&2; exit 2; }
  run_lane "$2" "$3" "$4" "$5"
  exit $?
fi
[[ "${mode}" == orchestrator ]] || { echo "unknown mode: ${mode}" >&2; exit 2; }

lane_a_pid=""
lane_b_pid=""
cleanup_groups() {
  local pid
  for pid in "${lane_a_pid}" "${lane_b_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || true
    fi
  done
}
trap 'cleanup_groups; exit 130' INT TERM

run_phase() {
  local baseline="$1"
  local phase_a phase_b
  setsid env GPU_A="${gpu_a}" GPU_B="${gpu_b}" RUN_ID="${run_id}" \
    PROMPTS_SOURCE="${prompts_source}" PREFIX_SOURCE="${prefix_source}" \
    WORLD_MODEL_RESULTS_ROOT="${WORLD_MODEL_RESULTS_ROOT}" \
    CAUSAL_FRAMES="${causal_frames}" MIN_FREE_GB="${min_free_gb}" \
    bash "${BASH_SOURCE[0]}" lane "${baseline}" "${gpu_a}" "${start_index}" "${shard_a_count}" &
  lane_a_pid=$!
  setsid env GPU_A="${gpu_a}" GPU_B="${gpu_b}" RUN_ID="${run_id}" \
    PROMPTS_SOURCE="${prompts_source}" PREFIX_SOURCE="${prefix_source}" \
    WORLD_MODEL_RESULTS_ROOT="${WORLD_MODEL_RESULTS_ROOT}" \
    CAUSAL_FRAMES="${causal_frames}" MIN_FREE_GB="${min_free_gb}" \
    bash "${BASH_SOURCE[0]}" lane "${baseline}" "${gpu_b}" "${shard_b_start}" "${shard_b_count}" &
  lane_b_pid=$!
  set +e
  wait "${lane_a_pid}"; phase_a=$?
  wait "${lane_b_pid}"; phase_b=$?
  set -e
  lane_a_pid=""
  lane_b_pid=""
  printf '%s_a=%s %s_b=%s\n' "${baseline}" "${phase_a}" "${baseline}" "${phase_b}" |
    tee "${status_root}/${baseline}_summary.txt"
  [[ "${phase_a}" -eq 0 && "${phase_b}" -eq 0 ]]
}

set +e
run_phase causal_forcing
causal_status=$?
run_phase longcat
longcat_status=$?
set -e

printf 'causal_forcing=%s longcat=%s\n' "${causal_status}" "${longcat_status}" |
  tee "${status_root}/summary.txt"
if [[ "${causal_status}" -eq 0 && "${longcat_status}" -eq 0 ]]; then
  touch "${status_root}/generation.done"
  echo "MovieGen128 B2 generation complete: ${run_id}"
  exit 0
fi
touch "${status_root}/generation.partial"
echo "MovieGen128 B2 generation finished with partial failures: ${run_id}" >&2
exit 1
