#!/bin/bash
# Compare TRQ and the student's QVG S2++ identity codec on identical raw KV dumps.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trq_root="$(cd "${script_dir}/../.." && pwd)"
repo_root="$(cd "${trq_root}/.." && pwd)"

qvg_root="${QVG_ROOT:-$(cd "${repo_root}/.." && pwd)/qvg}"
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_identity_codec_parity}"
output_dir="${OUTPUT_DIR:-${trq_root}/results/identity_codec_parity/${run_id}}"
dumps_glob="${DUMPS_GLOB:-${trq_root}/kv_dumps/trq_diagnostics/*/heldout/*.pt}"

export PYTHONPATH="${trq_root}/src:${PYTHONPATH:-}"

if [ ! -f "${qvg_root}/references/quant-videogen/quant_videogen/real/s2pp.py" ]; then
  echo "ERROR: Student QVG checkout not found at ${qvg_root}" >&2
  echo "Clone it with: git clone https://github.com/jiahui1021/qvg.git ${qvg_root}" >&2
  exit 1
fi
if ! compgen -G "${dumps_glob}" >/dev/null && [[ "${dumps_glob}" != *,* ]]; then
  echo "ERROR: No raw KV dumps matched: ${dumps_glob}" >&2
  echo "Collect dumps first with scripts/self_forcing/collect_kv_dumps.sh" >&2
  exit 1
fi

args=(
  --dumps "${dumps_glob}"
  --qvg-root "${qvg_root}"
  --output-dir "${output_dir}"
  --layers "${LAYERS:-0}"
  --kv "${KV:-both}"
  --max-dumps "${MAX_DUMPS:-1}"
  --device "${DEVICE:-cpu}"
  --codec-dtype "${CODEC_DTYPE:-bf16}"
  --scale-precision "${SCALE_PRECISION:-bf16}"
)

if [ -n "${CONFIGS:-}" ]; then
  IFS=',' read -r -a config_values <<< "${CONFIGS}"
  for config in "${config_values[@]}"; do
    args+=(--config "${config}")
  done
fi
if [ "${STUDENT_TRITON:-0}" = "1" ]; then
  args+=(--student-triton)
fi

mkdir -p "${output_dir}"
echo "Raw dumps:   ${dumps_glob}"
echo "Student QVG: ${qvg_root}"
echo "Output:      ${output_dir}"

python "${script_dir}/compare_identity_codecs.py" "${args[@]}" "$@" \
  2>&1 | tee "${output_dir}/run.log"
