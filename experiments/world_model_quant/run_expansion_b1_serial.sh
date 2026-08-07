#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"
configure_videoquant_runtime

lane_mode="${1:-orchestrator}"
causal_hy_gpu="${CAUSAL_HY_GPU:-6}"
longcat_gpu="${LONGCAT_GPU:-7}"
run_id="${RUN_ID:-expansion_a_20260724}"
stage_id="${STAGE_ID:-expansion_b1_$(date +%Y%m%d_%H%M%S)}"
prompts_source="${PROMPTS_SOURCE:-${REPO_ROOT}/temporalresidualkvquant/assets/moviegenbench_resume_32.txt}"
start_index="${START_INDEX:-10}"
limit="${LIMIT:-22}"
causal_frames="${CAUSAL_FRAMES:-84}"
hy_dataset_root="${HY_DATASET_ROOT:-${HOME}/storage/datasets/hy_worldplay_official}"
hy_holdout="${HY_SCENES:-${hy_dataset_root}/hy_scenes_holdout.json}"
hy_source="${HY_WORLDPLAY_SOURCE:-${REPO_ROOT}/references/quant-videogen}"
dry_run="${DRY_RUN:-0}"
log_root="${REPO_ROOT}/results/world_model_quant/logs/${stage_id}/orchestrator"
status_root="${REPO_ROOT}/results/world_model_quant/orchestration/${stage_id}"
mkdir -p "${log_root}" "${status_root}"

[[ "${causal_hy_gpu}" != "${longcat_gpu}" ]] || {
  echo "CAUSAL_HY_GPU and LONGCAT_GPU must differ" >&2
  exit 2
}
[[ "${start_index}" =~ ^[0-9]+$ && "${limit}" =~ ^[1-9][0-9]*$ ]] || {
  echo "START_INDEX and LIMIT must be non-negative/positive integers" >&2
  exit 2
}
[[ -s "${prompts_source}" ]] || {
  echo "prompt source is missing or empty: ${prompts_source}" >&2
  exit 2
}
[[ "$(grep -cve '^[[:space:]]*$' "${prompts_source}")" -ge $((start_index + limit)) ]] || {
  echo "prompt source does not cover indices ${start_index}-$((start_index + limit - 1))" >&2
  exit 2
}
[[ -s "${hy_holdout}" ]] || {
  echo "HY holdout manifest is missing: ${hy_holdout}" >&2
  exit 2
}

if [[ "${dry_run}" == 1 ]]; then
  cat <<EOF
STAGE_ID=${stage_id}
RUN_ID=${run_id} (resume existing decodable outputs)
GPU ${causal_hy_gpu} serial lane: Causal prompts ${start_index}-$((start_index + limit - 1)) -> HY official cases 6-10
GPU ${longcat_gpu} serial lane: LongCat prompts ${start_index}-$((start_index + limit - 1))
PROMPTS_SOURCE=${prompts_source}
CAUSAL_FRAMES=${causal_frames}
HY_SCENES=${hy_holdout}
No GPU commands executed.
EOF
  exit 0
fi

if [[ "${lane_mode}" == orchestrator ]]; then
  exec > >(tee -a "${log_root}/orchestrator.log") 2>&1
fi
lane_causal_hy_pid=""
lane_longcat_pid=""

cleanup_groups() {
  local pid
  for pid in "${lane_causal_hy_pid}" "${lane_longcat_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || true
    fi
  done
}
trap 'cleanup_groups; exit 130' INT TERM

record_stage() {
  local name="$1"
  shift
  rm -f "${status_root}/${name}.done" "${status_root}/${name}.failed"
  printf 'running\n' > "${status_root}/${name}.status"
  set +e
  "$@" > "${log_root}/${name}.log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "${status}" > "${status_root}/${name}.exit"
  if [[ "${status}" -eq 0 ]]; then
    printf 'done\n' > "${status_root}/${name}.status"
    touch "${status_root}/${name}.done"
    return 0
  fi
  printf 'failed\n' > "${status_root}/${name}.status"
  touch "${status_root}/${name}.failed"
  return "${status}"
}

is_video() {
  ffprobe -v error -show_entries stream=codec_type -of csv=p=0 "$1" 2>/dev/null |
    grep -qx video
}

validate_causal_total() {
  local root="$1"
  local expected="$2"
  local variant destination
  for variant in "${VARIANTS[@]}"; do
    destination="${status_root}/causal_missing_${variant}.txt"
    python "${script_dir}/prepare_causal_prompts.py" \
      --prompts "${prompts_source}" --start 0 --count "${expected}" \
      --output-dir "${root}/${variant}" --destination "${destination}" >/dev/null
    [[ ! -s "${destination}" ]] || return 1
  done
}

validate_longcat_total() {
  local root="$1"
  local expected="$2"
  local directory index
  for directory in prefix_bf16 "${VARIANTS[@]}"; do
    for ((index=0; index < expected; index++)); do
      is_video "${root}/${directory}/${index}-0.mp4" || return 1
    done
  done
}

validate_hy_manifest() {
  local root="$1"
  local manifest="$2"
  local scene action variant directory video
  while IFS= read -r scene; do
    for action in turn_left turn_right forward backward; do
      for variant in "${VARIANTS[@]}"; do
        directory="${root}/${scene}/${action}/${variant}"
        video="$(find "${directory}" -maxdepth 1 -type f -name '*.mp4' -print -quit 2>/dev/null || true)"
        [[ -n "${video}" ]] && is_video "${video}" || return 1
      done
    done
  done < <(python -c 'import json,sys; print("\n".join(x["id"] for x in json.load(open(sys.argv[1]))["scenes"]))' "${manifest}")
}

causal_hy_lane() {
  local failed=0
  if ! record_stage causal_moviegen32 env \
    RUN_ID="${run_id}" GPU="${causal_hy_gpu}" PROMPTS_SOURCE="${prompts_source}" \
    START_INDEX="${start_index}" LIMIT="${limit}" CAUSAL_FRAMES="${causal_frames}" \
    bash "${script_dir}/run_generation.sh" causal_forcing full; then
    failed=1
  elif ! validate_causal_total \
    "${REPO_ROOT}/results/world_model_quant/causal_forcing/${run_id}" \
    "$((start_index + limit))"; then
    printf 'failed:validation\n' > "${status_root}/causal_moviegen32.status"
    failed=1
  fi

  if ! record_stage hy_wan_holdout env \
    RUN_ID="${run_id}" GPU="${causal_hy_gpu}" HY_WORLDPLAY_SOURCE="${hy_source}" \
    HY_SCENES="${hy_holdout}" HY_LIMIT_SCENES=5 HY_MIN_SCENES=5 \
    bash "${script_dir}/run_generation.sh" hy_worldplay full; then
    failed=1
  elif ! validate_hy_manifest \
    "${REPO_ROOT}/results/world_model_quant/hy_worldplay/action_control_${run_id}" \
    "${hy_holdout}"; then
    printf 'failed:validation\n' > "${status_root}/hy_wan_holdout.status"
    failed=1
  fi
  return "${failed}"
}

longcat_lane() {
  if ! record_stage longcat_moviegen32 env \
    RUN_ID="${run_id}" GPU="${longcat_gpu}" PROMPTS_SOURCE="${prompts_source}" \
    START_INDEX="${start_index}" LIMIT="${limit}" \
    bash "${script_dir}/run_generation.sh" longcat full; then
    return 1
  fi
  if ! validate_longcat_total \
    "${REPO_ROOT}/results/world_model_quant/longcat/${run_id}" \
    "$((start_index + limit))"; then
    printf 'failed:validation\n' > "${status_root}/longcat_moviegen32.status"
    return 1
  fi
}

if [[ "${lane_mode}" == lane-causal-hy ]]; then
  causal_hy_lane
  exit $?
fi
if [[ "${lane_mode}" == lane-longcat ]]; then
  longcat_lane
  exit $?
fi
[[ "${lane_mode}" == orchestrator ]] || {
  echo "unknown lane mode: ${lane_mode}" >&2
  exit 2
}

setsid env \
  CAUSAL_HY_GPU="${causal_hy_gpu}" LONGCAT_GPU="${longcat_gpu}" \
  RUN_ID="${run_id}" STAGE_ID="${stage_id}" PROMPTS_SOURCE="${prompts_source}" \
  START_INDEX="${start_index}" LIMIT="${limit}" CAUSAL_FRAMES="${causal_frames}" \
  HY_DATASET_ROOT="${hy_dataset_root}" HY_SCENES="${hy_holdout}" \
  HY_WORLDPLAY_SOURCE="${hy_source}" \
  bash "${BASH_SOURCE[0]}" lane-causal-hy &
lane_causal_hy_pid=$!
setsid env \
  CAUSAL_HY_GPU="${causal_hy_gpu}" LONGCAT_GPU="${longcat_gpu}" \
  RUN_ID="${run_id}" STAGE_ID="${stage_id}" PROMPTS_SOURCE="${prompts_source}" \
  START_INDEX="${start_index}" LIMIT="${limit}" CAUSAL_FRAMES="${causal_frames}" \
  HY_DATASET_ROOT="${hy_dataset_root}" HY_SCENES="${hy_holdout}" \
  HY_WORLDPLAY_SOURCE="${hy_source}" \
  bash "${BASH_SOURCE[0]}" lane-longcat &
lane_longcat_pid=$!

set +e
wait "${lane_causal_hy_pid}"
causal_hy_status=$?
wait "${lane_longcat_pid}"
longcat_status=$?
set -e
lane_causal_hy_pid=""
lane_longcat_pid=""

printf 'causal_hy=%s longcat=%s\n' "${causal_hy_status}" "${longcat_status}" > "${status_root}/summary.txt"
if [[ "${causal_hy_status}" -eq 0 && "${longcat_status}" -eq 0 ]]; then
  touch "${status_root}/generation.done"
  echo "Expansion B1 generation complete: ${stage_id}"
  exit 0
fi
touch "${status_root}/generation.partial"
echo "Expansion B1 finished with partial failures: ${stage_id}" >&2
exit 1
