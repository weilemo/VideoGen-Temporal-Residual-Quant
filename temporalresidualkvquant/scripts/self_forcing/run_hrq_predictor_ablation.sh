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
#   TRQ_BITS, TRQ_GROUP_SIZE, TRQ_ANCHOR_BITS, TRQ_PREDICTOR_STRIDE,
#   TRQ_SCALE_PRECISION, TRQ_RESIDUAL_QUANT_MODE, TRQ_K_BITS, TRQ_V_BITS,
#   TRQ_FIRST_QUANT_FRAME, TRQ_QUANT_INTERVAL_FRAMES, TRQ_QUANT_SCHEDULE,
#   TRQ_GRADUAL_FRAMES, TRQ_PROTECTED_SINK_FRAMES, TRQ_CACHE_ROLES,
#   TRQ_QUANTIZED_LAYERS, ATTENTION_SINK_FRAMES,
#   NUM_OUTPUT_FRAMES, LOCAL_ATTN_SIZE, PROMPTS_PATH, OUTPUT_FOLDER, SEED
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
trq_bits="${TRQ_BITS:-2}"
trq_group_size="${TRQ_GROUP_SIZE:-${block_size}}"
trq_anchor_bits="${TRQ_ANCHOR_BITS:-4}"
trq_predictor_stride="${TRQ_PREDICTOR_STRIDE:-1560}"
trq_scale_precision="${TRQ_SCALE_PRECISION:-bf16}"
trq_residual_quant_mode="${TRQ_RESIDUAL_QUANT_MODE:-asym_zero_point}"
trq_k_bits="${TRQ_K_BITS:-0}"
trq_v_bits="${TRQ_V_BITS:-0}"
trq_first_quant_frame="${TRQ_FIRST_QUANT_FRAME:-24}"
trq_quant_interval_frames="${TRQ_QUANT_INTERVAL_FRAMES:-24}"
trq_quant_schedule="${TRQ_QUANT_SCHEDULE:-bulk}"
trq_gradual_frames="${TRQ_GRADUAL_FRAMES:-3}"
trq_protected_sink_frames="${TRQ_PROTECTED_SINK_FRAMES:-0}"
trq_cache_roles="${TRQ_CACHE_ROLES:-both}"
trq_quantized_layers="${TRQ_QUANTIZED_LAYERS:-all}"
attention_sink_frames="${ATTENTION_SINK_FRAMES:-0}"
seed="${SEED:-0}"
num_output_frames="${NUM_OUTPUT_FRAMES:-180}"
local_attn_size="${LOCAL_ATTN_SIZE:-180}"
prompts_path="${PROMPTS_PATH:-${hwq_root}/tmp/moviegenbench_15.txt}"
predictor_params_dir="${TRQ_PREDICTOR_PARAMS_DIR:-${HRQ_PREDICTOR_PARAMS_DIR:-${hwq_root}/assets/trq_predictors}}"

run_mode="${1:-all}"

export PYTHONPATH="${hwq_root}/src:${self_forcing_root}:${PYTHONPATH:-}"
export SELF_FORCING_CKPT_ROOT="${ckpt_root}"
export TRQ_PREDICTOR_PARAMS_DIR="${predictor_params_dir}"
export HRQ_PREDICTOR_PARAMS_DIR="${predictor_params_dir}"

results_root="${RESULTS_ROOT:-${hwq_root}/results/selfforcing/vbench_eval_trq_predictor}"
mkdir -p "${results_root}"
runtime_args=()
if [ "${PROFILE_RUNTIME:-0}" = "1" ]; then
  runtime_args+=(--profile)
fi
if [ "${SAVE_ROLLOUT_LATENTS:-0}" = "1" ]; then
  runtime_args+=(--save_rollout_latents)
fi
if [ -n "${ROLLOUT_METRICS_DIR:-}" ]; then
  runtime_args+=(--rollout_metrics_dir "${ROLLOUT_METRICS_DIR}")
fi
if [ -n "${PROMPT_INDICES:-}" ]; then
  runtime_args+=(--prompt_indices "${PROMPT_INDICES}")
fi

# ── Common inference args ────────────────────────────────────────────
common_args=(
  --config_path "${self_forcing_root}/configs/self_forcing_dmd.yaml"
  --checkpoint_path "${ckpt_path}"
  --data_path "${prompts_path}"
  --num_samples 1
  --seed "${seed}"
  --num_output_frames "${num_output_frames}"
  --local_attn_size "${local_attn_size}"
  --use_ema
  --save_with_index
  --quant_type "${TRQ_QUANT_TYPE:-trq-int${trq_bits}}"
  --quant_block_size "${block_size}"
  --trq_group_size "${trq_group_size}"
  --trq_anchor_bits "${trq_anchor_bits}"
  --trq_predictor_stride "${trq_predictor_stride}"
  --trq_scale_precision "${trq_scale_precision}"
  --trq_residual_quant_mode "${trq_residual_quant_mode}"
  --trq_k_bits "${trq_k_bits}"
  --trq_v_bits "${trq_v_bits}"
  --trq_first_quant_frame "${trq_first_quant_frame}"
  --trq_quant_interval_frames "${trq_quant_interval_frames}"
  --trq_quant_schedule "${trq_quant_schedule}"
  --trq_gradual_frames "${trq_gradual_frames}"
  --trq_protected_sink_frames "${trq_protected_sink_frames}"
  --trq_cache_roles "${trq_cache_roles}"
  --trq_quantized_layers "${trq_quantized_layers}"
  --attention_sink_frames "${attention_sink_frames}"
  --headwise_mode "${headwise_mode}"
  --head_importance_path "${head_importance_path}"
  --num_high_precision_heads "${num_hp_heads}"
  --high_precision_quant_type "${hp_quant}"
  --low_precision_quant_type "${lp_quant}"
  "${runtime_args[@]}"
)

# ── Helper: run one predictor mode ─────────────────────────────────
run_predictor() {
  local mode="$1"
  local default_name="identity_int${trq_bits}_a${trq_anchor_bits}_g${trq_group_size}_s${trq_predictor_stride}_${mode}"
  local out_dir="${OUTPUT_FOLDER:-${results_root}/${default_name}}"
  local log_file="${out_dir}.log"
  local pid_file="${out_dir}.pid"

  echo ""
  echo "=== Predictor: ${mode} ==="
  echo "Output: ${out_dir}"
  echo "Resolved TRQ: bits=${trq_bits} anchor=${trq_anchor_bits} group=${trq_group_size} stride=${trq_predictor_stride} scale=${trq_scale_precision} residual=${trq_residual_quant_mode} K/V=${trq_k_bits}/${trq_v_bits}"
  echo "Runtime: seed=${seed} frames=${num_output_frames} local_attn=${local_attn_size} headwise=${headwise_mode}"
  echo "Schedule: first=${trq_first_quant_frame} interval=${trq_quant_interval_frames} mode=${trq_quant_schedule} gradual=${trq_gradual_frames} protected_sink=${trq_protected_sink_frames} attention_sink=${attention_sink_frames} roles=${trq_cache_roles} layers=${trq_quantized_layers}"

  if [[ "${mode}" = "cross_kv" || "${mode}" = "hybrid_kv_innovation" ]]; then
    params_path="${TRQ_V_PREDICTOR_PARAMS_PATH:?Set TRQ_V_PREDICTOR_PARAMS_PATH for ${mode}}"
    if [ ! -f "${params_path}" ]; then
      echo "ERROR: Conditional predictor params not found: ${params_path}" >&2
      return 1
    fi
    echo "Params: ${params_path}"
  elif [ "${mode}" != "identity" ]; then
    params_path="${predictor_params_dir}/${mode}_self_forcing_dmd.pt"
    if [ ! -f "${params_path}" ]; then
      echo "ERROR: Predictor params not found: ${params_path}"
      echo "       Run analyze_hrq_predictor.py first to generate them."
      return 1
    fi
    echo "Params: ${params_path}"
  fi

  mkdir -p "${out_dir}"

  predictor_args=(--trq_predictor_mode "${mode}")
  if [[ "${mode}" = "cross_kv" || "${mode}" = "hybrid_kv_innovation" ]]; then
    predictor_args=(
      --trq_predictor_mode identity
      --trq_k_predictor_mode identity
      --trq_v_predictor_mode "${mode}"
      --trq_v_predictor_params_path "${params_path}"
    )
  fi
  torchrun --nproc_per_node=1 --standalone "${self_forcing_root}/inference.py" \
    "${common_args[@]}" \
    --output_folder "${out_dir}" \
    "${predictor_args[@]}" \
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

if [ "${run_mode}" = "cross_kv" ] || [ "${run_mode}" = "hybrid_kv_innovation" ]; then
  run_predictor "${run_mode}"
fi

if [ "${run_mode}" != "all" ] && [ "${run_mode}" != "identity" ] && [ "${run_mode}" != "affine_channel" ] && [ "${run_mode}" != "cross_kv" ] && [ "${run_mode}" != "hybrid_kv_innovation" ]; then
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
