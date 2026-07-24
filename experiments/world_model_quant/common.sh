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
  prepare_prompt_slice "${count}" 0 "${name}"
}

prepare_prompt_slice() {
  local count="$1"
  local start="$2"
  local name="$3"
  local source="${REPO_ROOT}/integrations/evaluation/moviegen10.txt"
  local destination="${REPO_ROOT}/results/world_model_quant/inputs/${name}.txt"
  local first=$((start + 1))
  local last=$((start + count))

  [[ "${count}" =~ ^[1-9][0-9]*$ ]] || {
    echo "count must be a positive integer: ${count}" >&2
    return 2
  }
  [[ "${start}" =~ ^[0-9]+$ ]] || {
    echo "start must be a non-negative integer: ${start}" >&2
    return 2
  }
  mkdir -p "$(dirname "${destination}")"
  sed -n "${first},${last}p" "${source}" > "${destination}"
  [[ "$(wc -l < "${destination}")" -eq "${count}" ]] || {
    echo "requested prompt slice ${start}:${count} exceeds ${source}" >&2
    return 2
  }
  printf '%s\n' "${destination}"
}
