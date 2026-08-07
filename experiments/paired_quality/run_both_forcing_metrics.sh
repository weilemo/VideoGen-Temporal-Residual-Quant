#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 SELF_FORCING_ROOT ROLLING_FORCING_ROOT [EXPECTED_VIDEOS]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_VIDEOS="${3:-0}"

python "${SCRIPT_DIR}/run_forcing_paired_metrics.py" \
  --baseline selfforcing \
  --root "$1" \
  --expected-videos "${EXPECTED_VIDEOS}"

python "${SCRIPT_DIR}/run_forcing_paired_metrics.py" \
  --baseline rollingforcing \
  --root "$2" \
  --expected-videos "${EXPECTED_VIDEOS}"
