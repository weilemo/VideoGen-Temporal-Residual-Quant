#!/bin/bash
# E4: run at most four preselected protection candidates from a TSV manifest.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trq_root="$(cd "${script_dir}/../.." && pwd)"
gpu_id="${GPU_ID:-0}"
manifest="${E4_CANDIDATES_TSV:?Set E4_CANDIDATES_TSV}"
run_root="${RUN_ROOT:-${HOME}/storage/runs/trq_causal_20260720/e4}"
prompts="${PROMPTS_PATH:-${trq_root}/assets/mb32_paired_smoke4.txt}"

case "${gpu_id}" in 0|1) ;; *) echo "ERROR: GPU_ID must be 0 or 1" >&2; exit 2 ;; esac
export CUDA_VISIBLE_DEVICES="${gpu_id}"

candidate_count="$(awk -F '\t' 'NF && $1 !~ /^#/ {count++} END {print count+0}' "${manifest}")"
if [ "${candidate_count}" -lt 1 ] || [ "${candidate_count}" -gt 4 ]; then
  echo "ERROR: E4 manifest must contain 1-4 candidates, got ${candidate_count}" >&2
  exit 2
fi

# Columns: name k_bits v_bits first schedule gradual protected_sink attention_sink roles layers
while IFS=$'\t' read -r name k_bits v_bits first schedule gradual protected_sink attention_sink roles layers; do
  if [ -z "${name}" ] || [[ "${name}" = \#* ]]; then continue; fi
  for seed in 0 1; do
    output="${run_root}/${name}/s${seed}"
    PROMPTS_PATH="${prompts}" NUM_OUTPUT_FRAMES=180 LOCAL_ATTN_SIZE=180 SEED="${seed}" \
    HEADWISE_MODE=none TRQ_BITS=2 TRQ_K_BITS="${k_bits}" TRQ_V_BITS="${v_bits}" \
    TRQ_FIRST_QUANT_FRAME="${first}" TRQ_QUANT_INTERVAL_FRAMES=24 \
    TRQ_QUANT_SCHEDULE="${schedule}" TRQ_GRADUAL_FRAMES="${gradual}" \
    TRQ_PROTECTED_SINK_FRAMES="${protected_sink}" ATTENTION_SINK_FRAMES="${attention_sink}" \
    TRQ_CACHE_ROLES="${roles}" TRQ_QUANTIZED_LAYERS="${layers}" \
    PROFILE_RUNTIME=1 SAVE_ROLLOUT_LATENTS=1 OUTPUT_FOLDER="${output}" \
    ROLLOUT_METRICS_DIR="${output}/rollout_metrics" \
      bash "${trq_root}/scripts/self_forcing/run_hrq_predictor_ablation.sh" identity
  done
done < "${manifest}"

echo "E4 candidates complete: ${run_root}"
