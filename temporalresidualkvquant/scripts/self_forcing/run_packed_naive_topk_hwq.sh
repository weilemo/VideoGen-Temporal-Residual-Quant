#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HEADWISE_MODE="${HEADWISE_MODE:-topk}"
export QUANT_TYPE="${QUANT_TYPE:-packed-naive-int2}"
export HIGH_PRECISION_QUANT_TYPE="${HIGH_PRECISION_QUANT_TYPE:-packed-naive-int4}"
export LOW_PRECISION_QUANT_TYPE="${LOW_PRECISION_QUANT_TYPE:-packed-naive-int2}"
export NUM_HIGH_PRECISION_HEADS="${NUM_HIGH_PRECISION_HEADS:-4}"

exec bash "${script_dir}/run_packed_naive_hwq.sh"
