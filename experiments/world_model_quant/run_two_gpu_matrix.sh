#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"

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
GPU ${longcat_gpu}: reuse LongCat smoke prompt 0, then prompts 1-4
GPU ${secondary_gpu}: Causal 21/42/84 pilot -> Causal prompts 3-9 -> HY dev -> LongCat prompts 5-9
HY dataset: ${hy_dataset_root}
No GPU commands executed.
EOF
  exit 0
fi

exec > >(tee -a "${log_root}/orchestrator.log") 2>&1

validate_variant_tree() {
  local root="$1"
  local expected="$2"
  local variant video count
  command -v ffprobe >/dev/null
  for variant in "${VARIANTS[@]}"; do
    count="$(find -L "${root}/${variant}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l)"
    [[ "${count}" -eq "${expected}" ]] || return 1
    while IFS= read -r video; do
      ffprobe -v error -show_entries stream=codec_type -of csv=p=0 "${video}" | grep -qx video
    done < <(find -L "${root}/${variant}" -maxdepth 1 -type f -name '*.mp4' | sort)
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

hy_prepare_args=(--dataset-root "${hy_dataset_root}")
if [[ -n "${hy_asset_source}" ]]; then
  hy_prepare_args+=(--source-dir "${hy_asset_source}")
fi
python "${script_dir}/prepare_hy_official_scenes.py" "${hy_prepare_args[@]}" \
  2>&1 | tee "${log_root}/prepare_hy_scenes.log"

longcat_root="${REPO_ROOT}/results/world_model_quant/longcat/${run_id}"
longcat_start=0
longcat_limit=5
if seed_longcat_smoke "${longcat_smoke_root}" "${longcat_root}"; then
  longcat_start=1
  longcat_limit=4
  echo "reused LongCat smoke prompt 0"
fi

(
  set +e
  RUN_ID="${run_id}" GPU="${longcat_gpu}" START_INDEX="${longcat_start}" LIMIT="${longcat_limit}" \
    bash "${script_dir}/run_generation.sh" longcat full
  status=$?
  printf '%s\n' "${status}" > "${status_root}/longcat_0_4.exit"
  exit "${status}"
) > "${log_root}/longcat_0_4.log" 2>&1 &
longcat_pid=$!

cleanup() {
  if kill -0 "${longcat_pid}" 2>/dev/null; then
    kill "${longcat_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

check_longcat_background() {
  if [[ -f "${status_root}/longcat_0_4.exit" ]] &&
     [[ "$(cat "${status_root}/longcat_0_4.exit")" -ne 0 ]]; then
    echo "LongCat prompts 0-4 failed; stopping secondary queue" >&2
    exit 1
  fi
}

RUN_ID="${run_id}" GPU="${secondary_gpu}" CAUSAL_SMOKE_ROOT="${causal_smoke_root}" \
  bash "${script_dir}/run_causal_length_pilot.sh" \
  2>&1 | tee "${log_root}/causal_length_pilot.log"
check_longcat_background

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
[[ -n "${selected_frames}" ]] || {
  echo "no Causal length passed the engineering gate" >&2
  exit 1
}
printf '%s\n' "${selected_frames}" > "${status_root}/causal_selected_frames.txt"
echo "selected Causal length: ${selected_frames}"

pilot_root="${REPO_ROOT}/results/world_model_quant/causal_forcing/${run_id}/length_pilot/frames_${selected_frames}"
causal_root="${REPO_ROOT}/results/world_model_quant/causal_forcing/${run_id}"
seed_variant_tree "${pilot_root}" "${causal_root}"
RUN_ID="${run_id}" GPU="${secondary_gpu}" START_INDEX=3 LIMIT=7 CAUSAL_FRAMES="${selected_frames}" \
  bash "${script_dir}/run_generation.sh" causal_forcing full \
  2>&1 | tee "${log_root}/causal_full.log"
validate_variant_tree "${causal_root}" 10
touch "${status_root}/causal.done"
check_longcat_background

HY_WORLDPLAY_SOURCE="${hy_source}" HY_SCENES="${hy_dataset_root}/hy_scenes_dev.json" \
  RUN_ID="${run_id}" GPU="${secondary_gpu}" HY_LIMIT_SCENES=5 HY_MIN_SCENES=5 \
  bash "${script_dir}/run_generation.sh" hy_worldplay full \
  2>&1 | tee "${log_root}/hy_dev.log"
hy_root="${REPO_ROOT}/results/world_model_quant/hy_worldplay/action_control_${run_id}"
[[ "$(find "${hy_root}" -type f -name '*.mp4' | wc -l)" -eq 100 ]]
touch "${status_root}/hy_dev.done"
check_longcat_background

RUN_ID="${run_id}" GPU="${secondary_gpu}" START_INDEX=5 LIMIT=5 \
  bash "${script_dir}/run_generation.sh" longcat full \
  2>&1 | tee "${log_root}/longcat_5_9.log"

wait "${longcat_pid}"
validate_variant_tree "${longcat_root}" 10
[[ "$(find -L "${longcat_root}/prefix_bf16" -maxdepth 1 -type f -name '*.mp4' | wc -l)" -eq 10 ]]
touch "${status_root}/longcat.done" "${status_root}/generation.done"
trap - EXIT INT TERM
printf '%s\n' "three-baseline expansion A complete: ${run_id}"
