#!/bin/bash
# Run the conditional-increment structure test on raw and decoder-visible states.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trq_root="$(cd "${script_dir}/../.." && pwd)"

: "${CALIBRATION_DUMPS:?Set CALIBRATION_DUMPS to raw BF16 calibration dumps}"
: "${VALIDATION_DUMPS:?Set VALIDATION_DUMPS to prompt-disjoint raw BF16 validation dumps}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new Gate R result directory}"

mkdir -p "${OUTPUT_ROOT}"
for state_source in raw reconstructed_k full_decoder_state; do
  output_dir="${OUTPUT_ROOT}/${state_source}"
  if [[ -f "${output_dir}/analysis.complete" ]]; then
    echo "SKIP complete state source: ${state_source}"
    continue
  fi
  echo "=== Conditional increment: ${state_source} ==="
  CALIBRATION_DUMPS="${CALIBRATION_DUMPS}" \
  VALIDATION_DUMPS="${VALIDATION_DUMPS}" \
  OUTPUT_DIR="${output_dir}" \
  STATE_SOURCE="${state_source}" \
  LAYERS="${LAYERS:-8-19}" \
  UNIT_SIZE="${UNIT_SIZE:-0}" \
  RIDGE="${RIDGE:-1e-4}" \
  BOOTSTRAP_RESAMPLES="${BOOTSTRAP_RESAMPLES:-2000}" \
  CODEC_BITS="${CODEC_BITS:-4}" \
  CODEC_ANCHOR_BITS="${CODEC_ANCHOR_BITS:-4}" \
  CODEC_BLOCK_SIZE="${CODEC_BLOCK_SIZE:-64}" \
  CODEC_PREDICTOR_STRIDE="${CODEC_PREDICTOR_STRIDE:-0}" \
  CODEC_SCALE_PRECISION="${CODEC_SCALE_PRECISION:-bf16}" \
  CODEC_RESIDUAL_QUANT_MODE="${CODEC_RESIDUAL_QUANT_MODE:-asym_zero_point}" \
    bash "${script_dir}/run_conditional_increment.sh"
done

export PYTHONPATH="${trq_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
summary_dir="${OUTPUT_ROOT}/gate"
mkdir -p "${summary_dir}"
command=(
  python "${script_dir}/summarize_conditional_increment_states.py"
  --raw "${OUTPUT_ROOT}/raw"
  --reconstructed-k "${OUTPUT_ROOT}/reconstructed_k"
  --full-decoder-state "${OUTPUT_ROOT}/full_decoder_state"
  --output-dir "${summary_dir}"
  --minimum-retention "${MINIMUM_RETENTION:-0.80}"
  --bootstrap-resamples "${BOOTSTRAP_RESAMPLES:-2000}"
  --seed "${SEED:-0}"
)
if [[ "${REQUIRE_PASS:-0}" == "1" ]]; then
  command+=(--require-pass)
fi
printf '%q ' "${command[@]}" > "${summary_dir}/command.txt"
printf '\n' >> "${summary_dir}/command.txt"
"${command[@]}" 2>&1 | tee "${summary_dir}/run.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "${summary_dir}/analysis.complete"
