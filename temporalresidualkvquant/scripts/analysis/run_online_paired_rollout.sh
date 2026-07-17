#!/bin/bash
# Generate BF16-A/B and TRQ rollouts with matched prompts/seeds, then analyze latents.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trq_root="$(cd "${script_dir}/../.." && pwd)"
prompts_path="${PROMPTS_PATH:-${trq_root}/assets/mb32_paired_smoke4.txt}"
output_root="${OUTPUT_ROOT:-${trq_root}/results/online_paired/smoke4}"
num_frames="${NUM_OUTPUT_FRAMES:-180}"
local_attn_size="${LOCAL_ATTN_SIZE:-180}"
seeds="${SEEDS:-0,1}"
k_bits="${TRQ_K_BITS:-2}"
v_bits="${TRQ_V_BITS:-2}"

if [ ! -f "${prompts_path}" ]; then
  echo "ERROR: prompt file not found: ${prompts_path}" >&2
  exit 1
fi

echo "TRQ root: ${trq_root}"
echo "Prompts: ${prompts_path}"
echo "Output: ${output_root}"
echo "Seeds: ${seeds}"
echo "Frames/local attention: ${num_frames}/${local_attn_size}"
echo "TRQ K/V bits: ${k_bits}/${v_bits}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1; no generation launched"
  exit 0
fi

IFS=',' read -r -a seed_values <<< "${seeds}"
for seed in "${seed_values[@]}"; do
  seed="${seed//[[:space:]]/}"
  for repeat in a b; do
    output_dir="${output_root}/bf16_${repeat}_s${seed}"
    PROMPTS_PATH="${prompts_path}" \
    NUM_OUTPUT_FRAMES="${num_frames}" \
    LOCAL_ATTN_SIZE="${local_attn_size}" \
    SEED="${seed}" \
    PROFILE_RUNTIME=1 \
    SAVE_ROLLOUT_LATENTS=1 \
    OUTPUT_FOLDER="${output_dir}" \
    ROLLOUT_METRICS_DIR="${output_dir}/rollout_metrics" \
      bash "${trq_root}/scripts/self_forcing/run_bf16.sh"
  done

  output_dir="${output_root}/trq_k${k_bits}v${v_bits}_s${seed}"
  PROMPTS_PATH="${prompts_path}" \
  NUM_OUTPUT_FRAMES="${num_frames}" \
  LOCAL_ATTN_SIZE="${local_attn_size}" \
  SEED="${seed}" \
  HEADWISE_MODE=none \
  TRQ_BITS=2 \
  TRQ_K_BITS="${k_bits}" \
  TRQ_V_BITS="${v_bits}" \
  PROFILE_RUNTIME=1 \
  SAVE_ROLLOUT_LATENTS=1 \
  OUTPUT_FOLDER="${output_dir}" \
  ROLLOUT_METRICS_DIR="${output_dir}/rollout_metrics" \
    bash "${trq_root}/scripts/self_forcing/run_hrq_predictor_ablation.sh" identity
done

export PYTHONPATH="${trq_root}/src:${PYTHONPATH:-}"
python "${trq_root}/scripts/analysis/analyze_online_paired_rollout.py" \
  --bf16-a "${output_root}/bf16_a_s*/rollout_metrics/latents/*.pt" \
  --bf16-b "${output_root}/bf16_b_s*/rollout_metrics/latents/*.pt" \
  --trq "${output_root}/trq_k${k_bits}v${v_bits}_s*/rollout_metrics/latents/*.pt" \
  --output-dir "${output_root}/analysis"
