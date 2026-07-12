#!/bin/bash
# Quick15 TRQ predictor ablation: identity / affine_channel.
#
# Runs topk-8 trq-int4/trq-int2 on moviegenbench_15 for each predictor mode.
# Identity baseline is cited from an existing run if already present;
# affine_channel requires assets/trq_predictors/*.pt to exist.
#
# Usage:
#   bash run_hrq_predictor_ablation.sh [identity|affine_channel|all]
#
# Env overrides:
#   HEADWISE_MODE, HEAD_IMPORTANCE_PATH, NUM_HIGH_PRECISION_HEADS,
#   HIGH_PRECISION_QUANT_TYPE, LOW_PRECISION_QUANT_TYPE, QUANT_BLOCK_SIZE,
#   NUM_OUTPUT_FRAMES, LOCAL_ATTN_SIZE, PROMPTS_PATH
#   TRQ_PREDICTOR_PARAMS_DIR  (default: assets/trq_predictors)

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hwq_root="$(cd "${script_dir}/../.." && pwd)"
self_forcing_root="${SELF_FORCING_ROOT:-${hwq_root}/backends/self_forcing}"

if [ -n "${SELF_FORCING_CKPT_ROOT:-}" ]; then
  ckpt_root="${SELF_FORCING_CKPT_ROOT}"
elif [ -d "${hwq_root}/ckpts/Self-Forcing" ]; then
  ckpt_root="${hwq_root}/ckpts/Self-Forcing"
else
  ckpt_root="${hwq_root}/ckpts/Self-Forcing"
fi
ckpt_path="${CKPT_PATH:-${ckpt_root}/self_forcing_dmd.pt}"

# ── Config ──────────────────────────────────────────────────────────
headwise_mode="${HEADWISE_MODE:-topk}"
head_importance_path="${HEAD_IMPORTANCE_PATH:-${hwq_root}/assets/head_importance/top8_dmd_loss.json}"
num_hp_heads="${NUM_HIGH_PRECISION_HEADS:-8}"
hp_quant="${HIGH_PRECISION_QUANT_TYPE:-trq-int4}"
lp_quant="${LOW_PRECISION_QUANT_TYPE:-trq-int2}"
block_size="${QUANT_BLOCK_SIZE:-64}"
num_output_frames="${NUM_OUTPUT_FRAMES:-180}"
local_attn_size="${LOCAL_ATTN_SIZE:-180}"
prompts_path="${PROMPTS_PATH:-${hwq_root}/tmp/moviegenbench_15.txt}"
predictor_params_dir="${TRQ_PREDICTOR_PARAMS_DIR:-${HRQ_PREDICTOR_PARAMS_DIR:-${hwq_root}/assets/trq_predictors}}"

run_mode="${1:-all}"

export PYTHONPATH="${hwq_root}/src:${self_forcing_root}:${PYTHONPATH:-}"
export SELF_FORCING_CKPT_ROOT="${ckpt_root}"
export TRQ_PREDICTOR_PARAMS_DIR="${predictor_params_dir}"
export HRQ_PREDICTOR_PARAMS_DIR="${predictor_params_dir}"

results_root="${hwq_root}/results/selfforcing/vbench_eval_trq_predictor"
mkdir -p "${results_root}"

# ── Common inference args ────────────────────────────────────────────
common_args=(
  --config_path "${self_forcing_root}/configs/self_forcing_dmd.yaml"
  --checkpoint_path "${ckpt_path}"
  --data_path "${prompts_path}"
  --num_samples 1
  --num_output_frames "${num_output_frames}"
  --local_attn_size "${local_attn_size}"
  --use_ema
  --save_with_index
  --quant_type "trq-int2"
  --quant_block_size "${block_size}"
  --trq_group_size "${block_size}"
  --trq_anchor_bits 4
  --trq_predictor_stride 1560
  --trq_scale_precision bf16
  --trq_residual_quant_mode asym_zero_point
  --headwise_mode "${headwise_mode}"
  --head_importance_path "${head_importance_path}"
  --num_high_precision_heads "${num_hp_heads}"
  --high_precision_quant_type "${hp_quant}"
  --low_precision_quant_type "${lp_quant}"
)

# ── Helper: run one predictor mode ─────────────────────────────────
run_predictor() {
  local mode="$1"
  local out_dir="${results_root}/quick15_topk8_${hp_quant}_${lp_quant}_pred_${mode}"
  local log_file="${out_dir}.log"
  local pid_file="${out_dir}.pid"

  echo ""
  echo "=== Predictor: ${mode} ==="
  echo "Output: ${out_dir}"

  if [ "${mode}" != "identity" ]; then
    params_path="${predictor_params_dir}/${mode}_self_forcing_dmd.pt"
    if [ ! -f "${params_path}" ]; then
      echo "ERROR: Predictor params not found: ${params_path}"
      echo "       Run analyze_hrq_predictor.py first to generate them."
      return 1
    fi
    echo "Params: ${params_path}"
  fi

  mkdir -p "${out_dir}"

  torchrun --nproc_per_node=1 --standalone "${self_forcing_root}/inference.py" \
    "${common_args[@]}" \
    --output_folder "${out_dir}" \
    --trq_predictor_mode "${mode}" \
    2>&1 | tee "${log_file}"

  echo "Done: ${mode} → ${out_dir}"
}

# ── Run selected modes ───────────────────────────────────────────────
if [ "${run_mode}" = "all" ] || [ "${run_mode}" = "identity" ]; then
  run_predictor "identity"
fi

if [ "${run_mode}" = "all" ] || [ "${run_mode}" = "affine_channel" ]; then
  run_predictor "affine_channel"
fi

if [ "${run_mode}" != "all" ] && [ "${run_mode}" != "identity" ] && [ "${run_mode}" != "affine_channel" ]; then
  echo "ERROR: Unsupported stable TRQ predictor: ${run_mode}" >&2
  echo "       RoPE, tiny_mlp, Cross-KV, AR2, and error-feedback remain experimental." >&2
  exit 2
fi

echo ""
echo "=== All requested predictor ablations complete ==="
echo "Results in: ${results_root}"
echo ""
echo "Next steps:"
echo "  1. Run VBench evaluation on each output folder"
echo "  2. Record the resolved TRQ config and physical state bytes with the metrics"
