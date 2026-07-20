#!/bin/bash
# E2/E3: online schedule, K/V role, and layer-group causal interventions.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trq_root="$(cd "${script_dir}/../.." && pwd)"
gpu_id="${GPU_ID:-0}"
run_root="${RUN_ROOT:-${HOME}/storage/runs/trq_causal_20260720}"
prompts="${PROMPTS_PATH:-${trq_root}/assets/mb32_paired_smoke4.txt}"
stage="${STAGE:-all}"

case "${gpu_id}" in
  0|1) ;;
  *) echo "ERROR: GPU_ID must be 0 or 1 for this lease, got ${gpu_id}" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="${gpu_id}"
mkdir -p "${run_root}/status"

run_trq() {
  local name="$1" seed="$2" indices="$3" first="$4" schedule="$5" gradual="$6"
  local protected_sink="$7" attention_sink="$8" roles="$9" layers="${10}"
  local output="${run_root}/${name}/s${seed}"
  local trace_dir=""
  if [[ "${name}" = e3_* ]]; then
    trace_dir="${output}/attention_trace"
  fi
  echo "RUN ${name} seed=${seed} prompts=${indices} GPU=${gpu_id}"
  PROMPTS_PATH="${prompts}" PROMPT_INDICES="${indices}" NUM_OUTPUT_FRAMES=180 LOCAL_ATTN_SIZE=180 \
  SEED="${seed}" HEADWISE_MODE=none TRQ_BITS=4 TRQ_K_BITS=4 TRQ_V_BITS=4 \
  TRQ_FIRST_QUANT_FRAME="${first}" TRQ_QUANT_INTERVAL_FRAMES=24 \
  TRQ_QUANT_SCHEDULE="${schedule}" TRQ_GRADUAL_FRAMES="${gradual}" \
  TRQ_PROTECTED_SINK_FRAMES="${protected_sink}" ATTENTION_SINK_FRAMES="${attention_sink}" \
  TRQ_CACHE_ROLES="${roles}" TRQ_QUANTIZED_LAYERS="${layers}" \
  TRQ_ATTENTION_TRACE_DIR="${trace_dir}" TRQ_ATTENTION_TRACE_LAYERS="${layers}" \
  PROFILE_RUNTIME=1 SAVE_ROLLOUT_LATENTS=1 OUTPUT_FOLDER="${output}" \
  ROLLOUT_METRICS_DIR="${output}/rollout_metrics" \
    bash "${trq_root}/scripts/self_forcing/run_hrq_predictor_ablation.sh" identity
}

run_e2() {
  for seed in 0 1; do
    run_trq e2_delay48 "${seed}" 0,2 48 bulk 3 0 0 both all
    run_trq e2_delay72 "${seed}" 0,2 72 bulk 3 0 0 both all
    run_trq e2_sink24 "${seed}" 0,2 24 bulk 3 24 0 both all
    run_trq e2_sink48 "${seed}" 0,2 48 bulk 3 48 0 both all
    run_trq e2_gradual3 "${seed}" 0,2 24 gradual 3 0 0 both all
  done
}

run_e3() {
  run_trq e3_k_only 0 0 24 bulk 3 0 0 k all
  run_trq e3_v_only 0 0 24 bulk 3 0 0 v all
  run_trq e3_layers_early 0 0 24 bulk 3 0 0 both 0-7
  run_trq e3_layers_middle 0 0 24 bulk 3 0 0 both 8-19
  run_trq e3_layers_late 0 0 24 bulk 3 0 0 both 20-29
}

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY RUN: E2 uses 5 configs x 2 prompts x 2 seeds = 20 videos"
  echo "DRY RUN: E3 adds K-only, V-only, and three layer groups on prompt 0 / seed 0"
  echo "DRY RUN: GPU is restricted to physical ${gpu_id}"
  exit 0
fi

case "${stage}" in
  e2) run_e2 ;;
  e3) run_e3 ;;
  all) run_e2; run_e3 ;;
  *) echo "ERROR: STAGE must be e2, e3, or all" >&2; exit 2 ;;
esac

touch "${run_root}/status/${stage}.done"
echo "Causal diagnosis stage complete: ${stage}"
