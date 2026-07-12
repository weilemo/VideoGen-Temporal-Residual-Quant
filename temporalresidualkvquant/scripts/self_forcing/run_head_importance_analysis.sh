#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hwq_root="$(cd "${script_dir}/../.." && pwd)"
self_forcing_root="${SELF_FORCING_ROOT:-${hwq_root}/backends/self_forcing}"

if [ -n "${SELF_FORCING_CKPT_ROOT:-}" ]; then
  ckpt_root="${SELF_FORCING_CKPT_ROOT}"
elif [ -d "${hwq_root}/ckpts/Self-Forcing" ]; then
  ckpt_root="${hwq_root}/ckpts/Self-Forcing"
elif [ -n "${QVG_ROOT:-}" ]; then
  ckpt_root="${QVG_ROOT}/ckpts/Self-Forcing"
else
  ckpt_root="${hwq_root}/ckpts/Self-Forcing"
fi

prompts_path="${PROMPTS_PATH:-${hwq_root}/assets/t2v.txt}"
ckpt_path="${CKPT_PATH:-${ckpt_root}/self_forcing_dmd.pt}"
num_layers="${NUM_LAYERS:-30}"
num_heads="${NUM_HEADS:-12}"
top_k="${TOP_K:-4}"
num_output_frames="${NUM_OUTPUT_FRAMES:-126}"
heads_per_batch="${HEADS_PER_BATCH:-3}"
head_start="${HEAD_START:-0}"
head_end="${HEAD_END:--1}"
local_attn_size="${LOCAL_ATTN_SIZE:-180}"
analysis_dir="${ANALYSIS_OUTPUT_DIR:-${hwq_root}/results/head_importance/focused_forcing_dmd}"
policy_output_path="${POLICY_OUTPUT_PATH:-${hwq_root}/assets/head_importance/top${top_k}_dmd_loss.json}"
phase="${PHASE:-all}"
num_loss_chunks="${NUM_LOSS_CHUNKS:-6}"
negative_prompt="${NEGATIVE_PROMPT:-色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形，毁容，杂乱背景}"
seed="${SEED:-0}"
allow_incomplete="${ALLOW_INCOMPLETE:-0}"
delete_latents_after_scoring="${DELETE_LATENTS_AFTER_SCORING:-0}"
skip_existing="${SKIP_EXISTING:-1}"

echo "temporalresidualkvquant root: ${hwq_root}"
echo "Self-Forcing backend: ${self_forcing_root}"
echo "Self-Forcing ckpt root: ${ckpt_root}"
echo "Checkpoint: ${ckpt_path}"
echo "Prompts: ${prompts_path}"
echo "Phase: ${phase}"
echo "Head range: ${head_start} to ${head_end}"
echo "Analysis output: ${analysis_dir}"
echo "Policy output: ${policy_output_path}"

export PYTHONPATH="${hwq_root}/src:${self_forcing_root}:${PYTHONPATH:-}"
export SELF_FORCING_CKPT_ROOT="${ckpt_root}"

run_inference() {
  echo "=== Phase 1: Inference Ablation ==="
  torchrun --nproc_per_node="${NPROC_PER_NODE:-1}" --standalone \
    "${self_forcing_root}/run_inference_ablation.py" \
    --config_path "${self_forcing_root}/configs/self_forcing_dmd.yaml" \
    --checkpoint_path "${ckpt_path}" \
    --data_path "${prompts_path}" \
    --output_folder "${analysis_dir}" \
    --num_layers "${num_layers}" \
    --num_heads "${num_heads}" \
    --num_output_frames "${num_output_frames}" \
    --heads_per_batch "${heads_per_batch}" \
    --head_start "${head_start}" \
    --head_end "${head_end}" \
    --local_attn_size "${local_attn_size}" \
    --num_loss_chunks "${num_loss_chunks}" \
    --seed "${seed}" \
    --use_ema
}

run_scoring() {
  echo "=== Phase 2: DMD Scoring ==="
  scoring_args=(
    --config_path "${self_forcing_root}/configs/self_forcing_dmd.yaml"
    --output_folder "${analysis_dir}"
    --policy_output_path "${policy_output_path}"
    --num_layers "${num_layers}"
    --num_heads "${num_heads}"
    --top_k "${top_k}"
    --negative_prompt "${negative_prompt}"
    --seed "${seed}"
  )
  if [ "${allow_incomplete}" = "1" ]; then
    scoring_args+=(--allow_incomplete)
  fi
  if [ "${delete_latents_after_scoring}" = "1" ]; then
    scoring_args+=(--delete_latents_after_scoring)
  fi
  if [ "${skip_existing}" = "0" ]; then
    scoring_args+=(--no-skip-existing)
  fi

  torchrun --nproc_per_node="${NPROC_PER_NODE:-1}" --standalone \
    "${self_forcing_root}/run_dmd_scoring.py" \
    "${scoring_args[@]}"
}

case "${phase}" in
  all)
    run_inference
    run_scoring
    ;;
  inference)
    run_inference
    ;;
  scoring)
    run_scoring
    ;;
  *)
    echo "Unknown phase: ${phase}. Valid: all, inference, scoring"
    exit 1
    ;;
esac
