#!/usr/bin/env bash

set -euo pipefail

WORLD_MODEL_EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${WORLD_MODEL_EXPERIMENT_DIR}/../.." && pwd)"
readonly WORLD_MODEL_EXPERIMENT_DIR REPO_ROOT

VARIANTS=(bf16 trq_int4 trq_int2 naive_int4 naive_int2)
readonly VARIANTS

require_baseline() {
  case "$1" in
    causal_forcing|longcat|hy_worldplay) ;;
    *)
      echo "unsupported baseline: $1" >&2
      return 2
      ;;
  esac
}

require_stage() {
  case "$1" in
    smoke|full) ;;
    *)
      echo "unsupported stage: $1 (expected smoke or full)" >&2
      return 2
      ;;
  esac
}

prepare_prompt_subset() {
  local count="$1"
  local name="$2"
  local source="${REPO_ROOT}/integrations/evaluation/moviegen10.txt"
  local destination="${REPO_ROOT}/results/world_model_quant/inputs/${name}.txt"
  mkdir -p "$(dirname "${destination}")"
  head -n "${count}" "${source}" > "${destination}"
  printf '%s\n' "${destination}"
}
