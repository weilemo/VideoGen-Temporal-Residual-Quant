#!/bin/bash
# Frozen Smoke4 comparison: BF16 vs temporal identity S2++ vs Cross-KV.
#
# This script never launches MB32. It writes a decision whose MB32 authorization
# remains false until automatic gates and manual failure tags both pass.
#
# Usage:
#   GPU_ID=<leased physical GPU> CROSS_PARAMS=/path/to/cross_codec.npz \
#     bash scripts/analysis/run_cross_kv_smoke.sh [all|generate|analyze]
#
# A local orchestration check does not need CUDA:
#   DRY_RUN=1 CROSS_PARAMS=/remote/path/cross_codec.npz \
#     bash scripts/analysis/run_cross_kv_smoke.sh all

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trq_root="$(cd "${script_dir}/../.." && pwd)"
action="${1:-all}"
case "${action}" in
  all|generate|analyze) ;;
  *) echo "Usage: $0 [all|generate|analyze]" >&2; exit 2 ;;
esac

run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_cross_kv_smoke4}"
run_root="${RUN_ROOT:-${trq_root}/results/cross_kv_smoke/${run_id}}"
generation_root="${run_root}/generation"
analysis_root="${run_root}/analysis"
logs_root="${run_root}/logs"
bf16_root="${BF16_ROOT:-${generation_root}/bf16}"
temporal_root="${generation_root}/temporal"
cross_root="${generation_root}/cross"
prompts="${PROMPTS_PATH:-${trq_root}/assets/mb32_paired_smoke4.txt}"
cross_params="${CROSS_PARAMS:-}"
prompt_indices_text="${PROMPT_INDICES_LIST:-0 1 2 3}"
seeds_text="${SEEDS:-0 1}"
dry_run="${DRY_RUN:-0}"
mkdir -p "${logs_root}" "${analysis_root}"

read -r -a prompt_indices <<<"${prompt_indices_text}"
read -r -a seeds <<<"${seeds_text}"
expected_pairs=$(( ${#prompt_indices[@]} * ${#seeds[@]} ))

if [[ "${dry_run}" != "1" ]]; then
  : "${cross_params:?Set CROSS_PARAMS to the frozen Cross-KV .npz checkpoint}"
  [[ -f "${cross_params}" ]] || { echo "Missing CROSS_PARAMS: ${cross_params}" >&2; exit 2; }
  [[ -f "${prompts}" ]] || { echo "Missing prompts: ${prompts}" >&2; exit 2; }
  python - "${cross_params}" <<'PY'
import numpy as np
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
with np.load(path, allow_pickle=False) as payload:
    missing = {"cross_weight", "cross_bias"}.difference(payload.files)
    if missing:
        raise SystemExit(f"Cross checkpoint lacks arrays: {sorted(missing)}")
PY
fi

print_command() {
  printf 'COMMAND:'
  printf ' %q' "$@"
  printf '\n'
}

run_logged() {
  local log_file="$1"
  shift
  print_command "$@"
  if [[ "${dry_run}" == "1" ]]; then
    return 0
  fi
  "$@" >"${log_file}" 2>&1
}

run_generation() {
  if [[ "${dry_run}" != "1" ]]; then
    : "${GPU_ID:?Set GPU_ID only after verifying the current physical-to-logical lease mapping}"
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  fi
  export PYTHONPATH="${trq_root}/src:${trq_root}/backends/self_forcing:${PYTHONPATH:-}"

  local prompt seed method output marker log_file
  for prompt in "${prompt_indices[@]}"; do
    for seed in "${seeds[@]}"; do
      for method in bf16 temporal cross; do
        output="${generation_root}/${method}/p${prompt}/s${seed}"
        if [[ "${method}" == "bf16" ]]; then
          output="${bf16_root}/p${prompt}/s${seed}"
        fi
        marker="${output}/generation.complete"
        log_file="${logs_root}/${method}_p${prompt}_s${seed}.log"
        if [[ -f "${marker}" ]]; then
          echo "SKIP complete: ${method} prompt=${prompt} seed=${seed}"
          continue
        fi
        mkdir -p "${output}"
        env_args=(
          "PROMPTS_PATH=${prompts}"
          "PROMPT_INDICES=${prompt}"
          "NUM_OUTPUT_FRAMES=${NUM_OUTPUT_FRAMES:-180}"
          "LOCAL_ATTN_SIZE=${LOCAL_ATTN_SIZE:-180}"
          "SEED=${seed}"
          "OUTPUT_FOLDER=${output}"
          "PROFILE_RUNTIME=1"
          "SAVE_ROLLOUT_LATENTS=1"
          "ROLLOUT_METRICS_DIR=${output}/rollout_metrics"
        )
        if [[ "${method}" == "bf16" ]]; then
          run_logged "${log_file}" env "${env_args[@]}" \
            bash "${trq_root}/scripts/self_forcing/run_bf16.sh"
        else
          env_args+=(
            "TRQ_QUANT_TYPE=s2pp-int4"
            "HEADWISE_MODE=none"
            "TRQ_BITS=4"
            "TRQ_K_BITS=4"
            "TRQ_V_BITS=4"
            "TRQ_ANCHOR_BITS=4"
            "TRQ_GROUP_SIZE=64"
            "TRQ_FIRST_QUANT_FRAME=24"
            "TRQ_CACHE_ROLES=both"
            "TRQ_QUANTIZED_LAYERS=${LAYERS:-8-19}"
            "TRQ_ATTENTION_TRACE_DIR=${output}/attention_trace"
            "TRQ_ATTENTION_TRACE_LAYERS=${LAYERS:-8-19}"
            "TRQ_PARITY_CAPTURE_DIR=${output}/parity_snapshots"
            "TRQ_PARITY_CAPTURE_LAYERS=${LAYERS:-8-19}"
          )
          if [[ "${method}" == "cross" ]]; then
            env_args+=("TRQ_V_PREDICTOR_PARAMS_PATH=${cross_params}")
            run_logged "${log_file}" env "${env_args[@]}" \
              bash "${trq_root}/scripts/self_forcing/run_hrq_predictor_ablation.sh" cross_kv
          else
            run_logged "${log_file}" env "${env_args[@]}" \
              bash "${trq_root}/scripts/self_forcing/run_hrq_predictor_ablation.sh" identity
          fi
        fi
        if [[ "${dry_run}" != "1" ]]; then
          touch "${marker}"
        fi
      done
    done
  done
}

run_analysis() {
  if [[ "${dry_run}" == "1" ]]; then
    echo "DRY_RUN: analysis requires completed videos, latents, traces, and runtime JSON."
    return 0
  fi
  local prompt seed method ref_dir cmp_dir quality_json
  local -a quality_args=()
  local -a review_args=()
  local -a lpips_args=()
  if [[ "${REQUIRE_LPIPS:-1}" == "1" ]]; then
    lpips_args+=(--require-lpips)
  fi
  for prompt in "${prompt_indices[@]}"; do
    for seed in "${seeds[@]}"; do
      ref_dir="${bf16_root}/p${prompt}/s${seed}"
      review_args+=(--run "bf16_a,${seed},${ref_dir}")
      for method in temporal cross; do
        cmp_dir="${generation_root}/${method}/p${prompt}/s${seed}"
        quality_json="${analysis_root}/quality/${method}/p${prompt}_s${seed}.json"
        mkdir -p "$(dirname "${quality_json}")"
        if [[ ! -f "${quality_json}" ]]; then
          python "${trq_root}/scripts/eval/eval_ref_metrics.py" \
            --ref-dir "${ref_dir}" --cmp-dir "${cmp_dir}" --out "${quality_json}" \
            --device "${EVAL_DEVICE:-cuda}" --match-by-index --strict-shape \
            "${lpips_args[@]}"
        fi
        quality_args+=(--quality "${method},${seed},${quality_json}")
        review_args+=(--run "${method},${seed},${cmp_dir}")
      done
    done
  done

  if [[ ! -f "${analysis_root}/review/failure_tags.csv" ]]; then
    python "${trq_root}/scripts/eval/create_paired_review.py" \
      "${review_args[@]}" --output-dir "${analysis_root}/review"
  fi

  python "${script_dir}/analyze_cross_kv_smoke.py" \
    --bf16 "${bf16_root}" \
    --temporal "${temporal_root}" \
    --cross "${cross_root}" \
    --temporal-traces "${temporal_root}" \
    --cross-traces "${cross_root}" \
    --runtime-root "bf16=${bf16_root}" \
    --runtime-root "temporal=${temporal_root}" \
    --runtime-root "cross=${cross_root}" \
    "${quality_args[@]}" \
    --failure-tags "${analysis_root}/review/failure_tags.csv" \
    --cross-params "${cross_params}" \
    --expected-pairs "${expected_pairs}" \
    --bytes-tolerance "${BYTES_TOLERANCE:-0.005}" \
    --output-dir "${analysis_root}/decision"
}

if [[ "${action}" == "all" || "${action}" == "generate" ]]; then
  run_generation
fi
if [[ "${action}" == "all" || "${action}" == "analyze" ]]; then
  run_analysis
fi

echo "Cross-KV Smoke workflow complete for action=${action}: ${run_root}"
