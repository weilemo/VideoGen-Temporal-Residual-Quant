#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

longcat_gpu="${LONGCAT_GPU:-2}"
secondary_gpu="${SECONDARY_GPU:-3}"
log_root="$(cd "${script_dir}/../.." && pwd)/results/world_model_quant/logs/two_gpu"
mkdir -p "${log_root}"

GPU="${longcat_gpu}" bash "${script_dir}/run_generation.sh" longcat full \
  > "${log_root}/longcat.log" 2>&1 &
longcat_pid=$!

cleanup() {
  if kill -0 "${longcat_pid}" 2>/dev/null; then
    kill "${longcat_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

GPU="${secondary_gpu}" bash "${script_dir}/run_generation.sh" causal_forcing full \
  2>&1 | tee "${log_root}/causal_forcing.log"
GPU="${secondary_gpu}" bash "${script_dir}/run_generation.sh" hy_worldplay full \
  2>&1 | tee "${log_root}/hy_worldplay.log"

wait "${longcat_pid}"
trap - EXIT INT TERM
printf '%s\n' "three-baseline generation matrix complete"
