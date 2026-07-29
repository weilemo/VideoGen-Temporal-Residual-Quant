#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trq_root="$(cd "${script_dir}/../.." && pwd)"

: "${E1_DIR:?Set E1_DIR to a completed PASS_E1 result directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the E2 result directory}"

LAYERS="${LAYERS:-all}"
UNIT_SIZE="${UNIT_SIZE:-0}"
KEY_BITS="${KEY_BITS:-4}"
VALUE_BITS="${VALUE_BITS:-4}"
ANCHOR_BITS="${ANCHOR_BITS:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
RESET_SPANS="${RESET_SPANS:-2,4,8}"
MAX_VALIDATION_DUMPS="${MAX_VALIDATION_DUMPS:-0}"
HYBRID_RATIO_THRESHOLD="${HYBRID_RATIO_THRESHOLD:-1.0}"
SHUFFLED_RATIO_THRESHOLD="${SHUFFLED_RATIO_THRESHOLD:-0.98}"
MAXIMUM_INCREASE_FRACTION="${MAXIMUM_INCREASE_FRACTION:-1.0}"
REQUIRE_CORE_PASS="${REQUIRE_CORE_PASS:-0}"

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
  python "${script_dir}/analyze_conditional_codec.py"
  --e1-dir "${E1_DIR}"
  --output-dir "${output_dir}"
  --layers "${LAYERS}"
  --unit-size "${UNIT_SIZE}"
  --key-bits "${KEY_BITS}"
  --value-bits "${VALUE_BITS}"
  --anchor-bits "${ANCHOR_BITS}"
  --block-size "${BLOCK_SIZE}"
  --reset-spans "${RESET_SPANS}"
  --max-validation-dumps "${MAX_VALIDATION_DUMPS}"
  --hybrid-ratio-threshold "${HYBRID_RATIO_THRESHOLD}"
  --shuffled-ratio-threshold "${SHUFFLED_RATIO_THRESHOLD}"
  --maximum-increase-fraction "${MAXIMUM_INCREASE_FRACTION}"
)
if [[ "${REQUIRE_CORE_PASS}" == "1" ]]; then
  command+=(--require-core-pass)
fi

printf '%q ' "${command[@]}" > "${output_dir}/command.txt"
printf '\n' >> "${output_dir}/command.txt"
"${command[@]}" 2>&1 | tee "${output_dir}/run.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "${output_dir}/analysis.complete"
