#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${E1_DIR:?Set E1_DIR to the active E1 result directory}"
: "${E2_OUTPUT_DIR:?Set E2_OUTPUT_DIR to the fixed-bit E2 result directory}"

POLL_SECONDS="${POLL_SECONDS:-60}"
QUEUE_STATE="${QUEUE_STATE:-${E2_OUTPUT_DIR}.queue.json}"

write_state() {
  local stage="$1"
  local detail="$2"
  python - "${QUEUE_STATE}" "${stage}" "${detail}" <<'PY'
import datetime
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).expanduser().resolve()
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "stage": sys.argv[2],
            "detail": sys.argv[3],
            "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
temporary.replace(path)
PY
}

read_status() {
  python - "$1" <<'PY'
import json
import pathlib
import sys

print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["status"])
PY
}

write_state "WAITING_E1" "waiting for E1 analysis.complete"
while [[ ! -f "${E1_DIR}/analysis.complete" ]]; do
  if [[ ! -d "${E1_DIR}/.running" ]]; then
    write_state "FAILED_E1" "E1 stopped without analysis.complete"
    echo "E1 stopped without analysis.complete" >&2
    exit 2
  fi
  sleep "${POLL_SECONDS}"
done

e1_status="$(read_status "${E1_DIR}/summary.json")"
if [[ "${e1_status}" != "PASS_E1" ]]; then
  write_state "STOPPED_E1_GATE" "${e1_status}"
  echo "Stopping queue because E1 status is ${e1_status}" >&2
  exit 3
fi

write_state "RUNNING_E2" "E1 passed; starting fixed-bit reconstructed-state analysis"
if ! E1_DIR="${E1_DIR}" \
  OUTPUT_DIR="${E2_OUTPUT_DIR}" \
  LAYERS="${LAYERS:-all}" \
  UNIT_SIZE="${UNIT_SIZE:-0}" \
  KEY_BITS="${KEY_BITS:-4}" \
  VALUE_BITS="${VALUE_BITS:-4}" \
  ANCHOR_BITS="${ANCHOR_BITS:-4}" \
  BLOCK_SIZE="${BLOCK_SIZE:-64}" \
  RESET_SPANS="${RESET_SPANS:-2,4,8}" \
  MAX_VALIDATION_DUMPS="${MAX_VALIDATION_DUMPS:-0}" \
  HYBRID_RATIO_THRESHOLD="${HYBRID_RATIO_THRESHOLD:-1.0}" \
  SHUFFLED_RATIO_THRESHOLD="${SHUFFLED_RATIO_THRESHOLD:-0.98}" \
  MAXIMUM_INCREASE_FRACTION="${MAXIMUM_INCREASE_FRACTION:-1.0}" \
  REQUIRE_CORE_PASS=1 \
  bash "${script_dir}/run_conditional_innovation_e2.sh"; then
  if [[ -f "${E2_OUTPUT_DIR}/summary.json" ]]; then
    write_state "STOPPED_E2_GATE" "$(read_status "${E2_OUTPUT_DIR}/summary.json")"
  else
    write_state "FAILED_E2" "E2 exited before writing summary.json"
  fi
  exit 4
fi

e2_status="$(read_status "${E2_OUTPUT_DIR}/summary.json")"
if [[ "${e2_status}" == "BLOCKED_E2_SUBGATES" ]]; then
  write_state "WAITING_E2_SUBGATES" "capture Q, event labels, and saturation counters before E3"
  exit 0
fi

write_state "STOPPED_E2_GATE" "${e2_status}"
exit 4
