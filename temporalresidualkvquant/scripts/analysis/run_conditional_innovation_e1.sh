#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trq_root="$(cd "${script_dir}/../.." && pwd)"

: "${CALIBRATION_DUMPS:?Set CALIBRATION_DUMPS to raw BF16 calibration dumps}"
: "${VALIDATION_DUMPS:?Set VALIDATION_DUMPS to prompt-disjoint raw BF16 validation dumps}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new result directory}"

LAYERS="${LAYERS:-all}"
UNIT_SIZE="${UNIT_SIZE:-0}"
MAX_CALIBRATION_DUMPS="${MAX_CALIBRATION_DUMPS:-0}"
MAX_VALIDATION_DUMPS="${MAX_VALIDATION_DUMPS:-0}"
CROSS_RIDGE="${CROSS_RIDGE:-1e-4}"
GAMMA_RIDGE="${GAMMA_RIDGE:-1e-6}"
GAMMA_RHO="${GAMMA_RHO:-0.95}"
BOOTSTRAP_RESAMPLES="${BOOTSTRAP_RESAMPLES:-2000}"
SEED="${SEED:-0}"
REQUIRE_PASS="${REQUIRE_PASS:-0}"

output_dir="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${OUTPUT_DIR}")"
mkdir -p "${output_dir}"
lock_dir="${output_dir}/.running"
if ! mkdir "${lock_dir}" 2>/dev/null; then
  echo "Refusing to overlap an existing run: ${lock_dir}" >&2
  exit 1
fi
cleanup() {
  rmdir "${lock_dir}" 2>/dev/null || true
}
trap cleanup EXIT

export PYTHONPATH="${trq_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
command=(
  python "${script_dir}/analyze_conditional_innovation.py"
  --calibration-dumps "${CALIBRATION_DUMPS}"
  --validation-dumps "${VALIDATION_DUMPS}"
  --output-dir "${output_dir}"
  --layers "${LAYERS}"
  --unit-size "${UNIT_SIZE}"
  --max-calibration-dumps "${MAX_CALIBRATION_DUMPS}"
  --max-validation-dumps "${MAX_VALIDATION_DUMPS}"
  --cross-ridge "${CROSS_RIDGE}"
  --gamma-ridge "${GAMMA_RIDGE}"
  --gamma-rho "${GAMMA_RHO}"
  --bootstrap-resamples "${BOOTSTRAP_RESAMPLES}"
  --seed "${SEED}"
)
if [[ "${REQUIRE_PASS}" == "1" ]]; then
  command+=(--require-pass)
fi

printf '%q ' "${command[@]}" > "${output_dir}/command.txt"
printf '\n' >> "${output_dir}/command.txt"
"${command[@]}" 2>&1 | tee "${output_dir}/run.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "${output_dir}/analysis.complete"
