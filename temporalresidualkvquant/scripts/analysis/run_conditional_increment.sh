#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trq_root="$(cd "${script_dir}/../.." && pwd)"

: "${CALIBRATION_DUMPS:?Set CALIBRATION_DUMPS to raw BF16 calibration dumps}"
: "${VALIDATION_DUMPS:?Set VALIDATION_DUMPS to prompt-disjoint raw BF16 validation dumps}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new result directory}"

LAYERS="${LAYERS:-8-19}"
UNIT_SIZE="${UNIT_SIZE:-0}"
RIDGE="${RIDGE:-1e-4}"
WRONG_SPACE_SHIFT="${WRONG_SPACE_SHIFT:-1}"
SAMPLE_CHUNK="${SAMPLE_CHUNK:-4096}"
BOOTSTRAP_RESAMPLES="${BOOTSTRAP_RESAMPLES:-2000}"
SEED="${SEED:-0}"
MINIMUM_PARTIAL_R2="${MINIMUM_PARTIAL_R2:-0.01}"
MINIMUM_CONTROL_MARGIN="${MINIMUM_CONTROL_MARGIN:-0.005}"
MINIMUM_IMPROVED_GROUPS="${MINIMUM_IMPROVED_GROUPS:-0.60}"
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
  python "${script_dir}/analyze_conditional_increment.py"
  --calibration-dumps "${CALIBRATION_DUMPS}"
  --validation-dumps "${VALIDATION_DUMPS}"
  --output-dir "${output_dir}"
  --layers "${LAYERS}"
  --unit-size "${UNIT_SIZE}"
  --ridge "${RIDGE}"
  --wrong-space-shift "${WRONG_SPACE_SHIFT}"
  --sample-chunk "${SAMPLE_CHUNK}"
  --bootstrap-resamples "${BOOTSTRAP_RESAMPLES}"
  --seed "${SEED}"
  --minimum-partial-r2 "${MINIMUM_PARTIAL_R2}"
  --minimum-control-margin "${MINIMUM_CONTROL_MARGIN}"
  --minimum-improved-groups "${MINIMUM_IMPROVED_GROUPS}"
)
if [[ "${REQUIRE_PASS}" == "1" ]]; then
  command+=(--require-pass)
fi

printf '%q ' "${command[@]}" > "${output_dir}/command.txt"
printf '\n' >> "${output_dir}/command.txt"
"${command[@]}" 2>&1 | tee "${output_dir}/run.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "${output_dir}/analysis.complete"
