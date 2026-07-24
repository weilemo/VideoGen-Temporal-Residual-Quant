#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.sh"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
pilot_prompts="${PILOT_PROMPTS:-3}"
run_id="${RUN_ID:-length_pilot}"
reuse_smoke_root="${CAUSAL_SMOKE_ROOT:-}"
read -r -a lengths <<< "${CAUSAL_LENGTHS:-21 42 84}"
failed=0

for frames in "${lengths[@]}"; do
  output_root="${REPO_ROOT}/results/world_model_quant/causal_forcing/${run_id}/length_pilot/frames_${frames}"
  log_root="${REPO_ROOT}/results/world_model_quant/logs/${run_id}/length_pilot/frames_${frames}"
  mkdir -p "${log_root}"
  start_index=0
  count="${pilot_prompts}"
  if [[ "${frames}" == 21 && -n "${reuse_smoke_root}" ]]; then
    reused=1
    for variant in "${VARIANTS[@]}"; do
      source_video="$(find "${reuse_smoke_root}/${variant}" -maxdepth 1 -type f -name '*.mp4' -print -quit 2>/dev/null || true)"
      if [[ -z "${source_video}" ]]; then
        reused=0
        break
      fi
      mkdir -p "${output_root}/${variant}"
      ln -sfn "${source_video}" "${output_root}/${variant}/$(basename "${source_video}")"
    done
    if [[ "${reused}" == 1 ]]; then
      start_index=1
      count=$((pilot_prompts - 1))
      printf 'reused Causal smoke prompt 0 for 21-frame pilot\n'
    fi
  fi
  if [[ "${count}" -eq 0 ]]; then
    continue
  fi
  for variant in "${VARIANTS[@]}"; do
    prompts="$(prepare_missing_causal_prompts \
      "${start_index}" "${count}" \
      "causal_${run_id}_frames_${frames}_${variant}_${start_index}_${count}" \
      "${output_root}/${variant}")"
    if [[ ! -s "${prompts}" ]]; then
      printf 'skip complete Causal pilot: frames=%s variant=%s\n' "${frames}" "${variant}"
      continue
    fi
    if ! PROMPTS="${prompts}" OUTPUT_ROOT="${output_root}" NUM_OUTPUT_FRAMES="${frames}" \
      bash "${REPO_ROOT}/integrations/causal_forcing/run_moviegen10.sh" "${variant}" \
      2>&1 | tee "${log_root}/${variant}.log"; then
      failed=1
      printf 'Causal pilot failed: frames=%s variant=%s\n' "${frames}" "${variant}" >&2
    fi
  done
done

exit "${failed}"
