#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"
configure_videoquant_runtime

longcat_gpu="${LONGCAT_GPU:-2}"
secondary_gpu="${SECONDARY_GPU:-4}"
run_id="${RUN_ID:-expanded_$(date +%Y%m%d_%H%M%S)}"
dry_run="${DRY_RUN:-0}"
log_root="${REPO_ROOT}/results/world_model_quant/logs/${run_id}/orchestrator"
status_root="${REPO_ROOT}/results/world_model_quant/orchestration/${run_id}"
hy_dataset_root="${HY_DATASET_ROOT:-${HOME}/storage/datasets/hy_worldplay_official}"
hy_asset_source="${HY_ASSET_SOURCE:-}"
hy_source="${HY_WORLDPLAY_SOURCE:-${REPO_ROOT}/references/quant-videogen}"
longcat_smoke_root="${LONGCAT_SMOKE_ROOT:-}"
causal_smoke_root="${CAUSAL_SMOKE_ROOT:-}"
mkdir -p "${log_root}" "${status_root}"

[[ "${longcat_gpu}" != "${secondary_gpu}" ]] || {
  echo "LONGCAT_GPU and SECONDARY_GPU must differ" >&2
  exit 2
}

if [[ "${dry_run}" == 1 ]]; then
  cat <<EOF
RUN_ID=${run_id}
GPU ${longcat_gpu}: adopt/reuse LongCat prompts 0-4, then fill missing outputs
GPU ${secondary_gpu}: resumable Causal 21/42/84 pilot -> Causal full -> HY dev -> LongCat prompts 5-9
HY dataset: ${hy_dataset_root}
Independent stages continue after a sibling failure.
No GPU commands executed.
EOF
  exit 0
fi

exec > >(tee -a "${log_root}/orchestrator.log") 2>&1
secondary_pid=""
longcat_pid=""

cleanup_groups() {
  local pid
  for pid in "${secondary_pid}" "${longcat_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || true
    fi
  done
}
trap 'cleanup_groups; exit 130' INT TERM

run_stage() {
  local name="$1"
  local log="$2"
  shift 2
  rm -f "${status_root}/${name}.done" "${status_root}/${name}.failed"
  printf 'running\n' > "${status_root}/${name}.status"
  setsid "$@" > "${log}" 2>&1 &
  secondary_pid=$!
  set +e
  wait "${secondary_pid}"
  local status=$?
  set -e
  secondary_pid=""
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
  ffprobe -v error -show_entries stream=codec_type -of csv=p=0 "$1" 2>/dev/null | grep -qx video
}

validate_variant_tree() {
  local root="$1"
  local expected="$2"
  local variant video count
  command -v ffprobe >/dev/null
  for variant in "${VARIANTS[@]}"; do
    count="$(find -L "${root}/${variant}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l)"
    [[ "${count}" -eq "${expected}" ]] || return 1
    while IFS= read -r video; do
      is_video "${video}" || return 1
    done < <(find -L "${root}/${variant}" -maxdepth 1 -type f -name '*.mp4' | sort)
  done
}

validate_longcat_range() {
  local root="$1"
  local start="$2"
  local count="$3"
  local directory index
  for directory in prefix_bf16 "${VARIANTS[@]}"; do
    for ((index=start; index < start + count; index++)); do
      is_video "${root}/${directory}/${index}-0.mp4" || return 1
    done
  done
}

seed_variant_tree() {
  local source="$1"
  local destination="$2"
  local variant video
  [[ -d "${source}" ]] || return 1
  for variant in "${VARIANTS[@]}"; do
    mkdir -p "${destination}/${variant}"
    while IFS= read -r video; do
      ln -sfn "${video}" "${destination}/${variant}/$(basename "${video}")"
    done < <(find -L "${source}/${variant}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | sort)
  done
}

seed_longcat_smoke() {
  local source="$1"
  local destination="$2"
  local directory video
  [[ -d "${source}" ]] || return 1
  for directory in prefix_bf16 "${VARIANTS[@]}"; do
    video="$(find "${source}/${directory}" -maxdepth 1 -type f -name '0-0.mp4' -print -quit 2>/dev/null || true)"
    [[ -n "${video}" ]] || return 1
    mkdir -p "${destination}/${directory}"
    ln -sfn "${video}" "${destination}/${directory}/0-0.mp4"
  done
}

longcat_is_running() {
  pgrep -f "moviegen10.py.*--output-root ${longcat_root}" >/dev/null 2>&1
}

start_longcat_first_shard() {
  printf 'running\n' > "${status_root}/longcat_0_4.status"
  setsid env RUN_ID="${run_id}" GPU="${longcat_gpu}" START_INDEX=0 LIMIT=5 \
    bash "${script_dir}/run_generation.sh" longcat full \
    > "${log_root}/longcat_0_4.log" 2>&1 &
  longcat_pid=$!
}

finish_longcat_first_shard() {
  local status=0
  if [[ -n "${longcat_pid}" ]]; then
    set +e
    wait "${longcat_pid}"
    status=$?
    set -e
    longcat_pid=""
  else
    while longcat_is_running; do
      sleep 30
    done
  fi
  if [[ "${status}" -ne 0 ]] || ! validate_longcat_range "${longcat_root}" 0 5; then
    echo "LongCat prompts 0-4 are incomplete; filling only missing videos"
    start_longcat_first_shard
    set +e
    wait "${longcat_pid}"
    status=$?
    set -e
    longcat_pid=""
  fi
  printf '%s\n' "${status}" > "${status_root}/longcat_0_4.exit"
  if [[ "${status}" -eq 0 ]] && validate_longcat_range "${longcat_root}" 0 5; then
    printf 'done\n' > "${status_root}/longcat_0_4.status"
    touch "${status_root}/longcat_0_4.done"
    return 0
  fi
  printf 'failed\n' > "${status_root}/longcat_0_4.status"
  touch "${status_root}/longcat_0_4.failed"
  return 1
}

hy_prepare_args=(--dataset-root "${hy_dataset_root}")
if [[ -n "${hy_asset_source}" ]]; then
  hy_prepare_args+=(--source-dir "${hy_asset_source}")
fi
python "${script_dir}/prepare_hy_official_scenes.py" "${hy_prepare_args[@]}" \
  2>&1 | tee "${log_root}/prepare_hy_scenes.log"

longcat_root="${REPO_ROOT}/results/world_model_quant/longcat/${run_id}"
seed_longcat_smoke "${longcat_smoke_root}" "${longcat_root}" || true
if longcat_is_running; then
  printf 'adopted\n' > "${status_root}/longcat_0_4.status"
  echo "adopted existing LongCat process for ${run_id}"
else
  start_longcat_first_shard
fi

causal_pilot_status=0
if ! run_stage causal_length_pilot "${log_root}/causal_length_pilot.log" \
  env RUN_ID="${run_id}" GPU="${secondary_gpu}" CAUSAL_SMOKE_ROOT="${causal_smoke_root}" \
  bash "${script_dir}/run_causal_length_pilot.sh"; then
  causal_pilot_status=1
  echo "Causal pilot has failed variants; selecting only a fully validated length"
fi

selected_frames=""
read -r -a candidate_lengths <<< "${CAUSAL_LENGTHS:-21 42 84}"
for ((index=${#candidate_lengths[@]} - 1; index >= 0; index--)); do
  frames="${candidate_lengths[index]}"
  candidate_root="${REPO_ROOT}/results/world_model_quant/causal_forcing/${run_id}/length_pilot/frames_${frames}"
  if validate_variant_tree "${candidate_root}" "${PILOT_PROMPTS:-3}"; then
    selected_frames="${frames}"
    break
  fi
done

causal_status=0
if [[ -n "${selected_frames}" ]]; then
  printf '%s\n' "${selected_frames}" > "${status_root}/causal_selected_frames.txt"
  echo "selected Causal length: ${selected_frames}"
  pilot_root="${REPO_ROOT}/results/world_model_quant/causal_forcing/${run_id}/length_pilot/frames_${selected_frames}"
  causal_root="${REPO_ROOT}/results/world_model_quant/causal_forcing/${run_id}"
  seed_variant_tree "${pilot_root}" "${causal_root}"
  if ! run_stage causal_full "${log_root}/causal_full.log" \
    env RUN_ID="${run_id}" GPU="${secondary_gpu}" START_INDEX=3 LIMIT=7 CAUSAL_FRAMES="${selected_frames}" \
    bash "${script_dir}/run_generation.sh" causal_forcing full; then
    causal_status=1
  elif ! validate_variant_tree "${causal_root}" 10; then
    causal_status=1
    printf 'failed\n' > "${status_root}/causal_full.status"
  fi
else
  causal_status=1
  printf 'failed:no_validated_length\n' > "${status_root}/causal_full.status"
fi

hy_status=0
if ! run_stage hy_dev "${log_root}/hy_dev.log" \
  env HY_WORLDPLAY_SOURCE="${hy_source}" HY_SCENES="${hy_dataset_root}/hy_scenes_dev.json" \
  RUN_ID="${run_id}" GPU="${secondary_gpu}" HY_LIMIT_SCENES=5 HY_MIN_SCENES=5 \
  bash "${script_dir}/run_generation.sh" hy_worldplay full; then
  hy_status=1
fi
hy_root="${REPO_ROOT}/results/world_model_quant/hy_worldplay/action_control_${run_id}"
if [[ "$(find "${hy_root}" -type f -name '*.mp4' 2>/dev/null | wc -l)" -ne 100 ]]; then
  hy_status=1
fi

longcat_status=0
finish_longcat_first_shard || longcat_status=1
if ! run_stage longcat_5_9 "${log_root}/longcat_5_9.log" \
  env RUN_ID="${run_id}" GPU="${secondary_gpu}" START_INDEX=5 LIMIT=5 \
  bash "${script_dir}/run_generation.sh" longcat full; then
  longcat_status=1
fi
if ! validate_longcat_range "${longcat_root}" 0 10; then
  longcat_status=1
fi

printf 'causal_pilot=%s causal=%s hy=%s longcat=%s\n' \
  "${causal_pilot_status}" "${causal_status}" "${hy_status}" "${longcat_status}" \
  > "${status_root}/summary.txt"
if [[ "${causal_status}" -eq 0 && "${hy_status}" -eq 0 && "${longcat_status}" -eq 0 ]]; then
  touch "${status_root}/generation.done"
  printf '%s\n' "three-baseline expansion A complete: ${run_id}"
  exit 0
fi
touch "${status_root}/generation.partial"
printf '%s\n' "three-baseline expansion A finished with partial failures: ${run_id}" >&2
exit 1
