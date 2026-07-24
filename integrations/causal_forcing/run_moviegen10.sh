#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: $0 MODE (bf16, trq_int4, trq_int2, naive_int4, naive_int2)}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAUSAL_REPO="${CAUSAL_REPO:-${REPO_ROOT}/forcing/causalforcing_official}"
WAN_MODEL_ROOT="${WAN_MODEL_ROOT:-${HOME}/storage/models/Wan2.1-T2V-1.3B}"
CAUSAL_CHECKPOINT="${CAUSAL_CHECKPOINT:-${HOME}/storage/models/Causal-Forcing/chunkwise/causal_forcing.pt}"
PROMPTS="${PROMPTS:-${REPO_ROOT}/integrations/evaluation/moviegen10.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/world_model_quant/causal_forcing/moviegen10}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/videoquant/bin/python}"

case "${MODE}" in
  bf16) QUANT_TYPE="none" ;;
  trq_int4) QUANT_TYPE="trq-int4" ;;
  trq_int2) QUANT_TYPE="trq-int2" ;;
  naive_int4) QUANT_TYPE="packed-naive-int4" ;;
  naive_int2) QUANT_TYPE="packed-naive-int2" ;;
  *) echo "Unsupported mode: ${MODE}" >&2; exit 2 ;;
esac

test -f "${CAUSAL_CHECKPOINT}"
test -f "${WAN_MODEL_ROOT}/Wan2.1_VAE.pth"
test -x "${PYTHON_BIN}"
mkdir -p "${OUTPUT_ROOT}/${MODE}" "${HOME}/storage/tmp"

export WAN_MODEL_ROOT
export TMPDIR="${TMPDIR:-${HOME}/storage/tmp}"
export PYTHONPATH="${REPO_ROOT}/temporalresidualkvquant/src:${CAUSAL_REPO}:${PYTHONPATH:-}"

cd "${CAUSAL_REPO}"
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=1 inference.py \
  --config_path configs/causal_forcing_dmd_chunkwise.yaml \
  --checkpoint_path "${CAUSAL_CHECKPOINT}" \
  --data_path "${PROMPTS}" \
  --output_folder "${OUTPUT_ROOT}/${MODE}" \
  --num_output_frames "${NUM_OUTPUT_FRAMES:-21}" \
  --kv_cache_capacity_frames "${KV_CACHE_CAPACITY_FRAMES:-${NUM_OUTPUT_FRAMES:-21}}" \
  --seed "${SEED:-42}" \
  --kv_quant_type "${QUANT_TYPE}" \
  --kv_quant_block_size "${KV_QUANT_BLOCK_SIZE:-64}" \
  --trq_anchor_bits "${TRQ_ANCHOR_BITS:-4}" \
  --trq_predictor_stride "${TRQ_PREDICTOR_STRIDE:-1560}" \
  --trq_predictor_mode "${TRQ_PREDICTOR_MODE:-identity}" \
  --trq_k_bits "${TRQ_K_BITS:-0}" \
  --trq_v_bits "${TRQ_V_BITS:-0}"
