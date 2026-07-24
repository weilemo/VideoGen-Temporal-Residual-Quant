#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: $0 MODE [prefix|continuation]}"
TASK="${2:-continuation}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

LONGCAT_REPO="${LONGCAT_REPO:-${REPO_ROOT}/forcing/longcatvideo}"
LONGCAT_MODEL="${LONGCAT_MODEL:-${HOME}/storage/models/LongCat-Video}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/world_model_quant/longcat/moviegen10}"
PREFIX_DIR="${PREFIX_DIR:-${OUTPUT_ROOT}/prefix_bf16}"
PROMPTS="${PROMPTS:-${REPO_ROOT}/integrations/evaluation/moviegen10.txt}"

export PYTHONPATH="${REPO_ROOT}/temporalresidualkvquant/src:${LONGCAT_REPO}:${PYTHONPATH:-}"

torchrun --standalone --nproc_per_node=1 \
  "${REPO_ROOT}/integrations/longcat_video/moviegen10.py" \
  --longcat-repo "${LONGCAT_REPO}" \
  --checkpoint-dir "${LONGCAT_MODEL}" \
  --prompts "${PROMPTS}" \
  --output-root "${OUTPUT_ROOT}" \
  --prefix-dir "${PREFIX_DIR}" \
  --task "${TASK}" \
  --mode "${MODE}" \
  --limit "${LIMIT:-10}" \
  --start-index "${START_INDEX:-0}" \
  --seed "${SEED:-42}" \
  --num-inference-steps "${STEPS:-50}"
