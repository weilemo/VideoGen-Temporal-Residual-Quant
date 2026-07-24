#!/bin/bash
# Run the same identity profiles through TRQ and the student's S2++ integration.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trq_root="$(cd "${script_dir}/../.." && pwd)"
repo_root="$(cd "${trq_root}/.." && pwd)"
qvg_root="${QVG_ROOT:-$(cd "${repo_root}/.." && pwd)/qvg}"
results_root="${OUTPUT_ROOT:-${trq_root}/results/identity_online_matrix}"
prompts_path="${PROMPTS_PATH:-${trq_root}/assets/t2v.txt}"
num_frames="${NUM_OUTPUT_FRAMES:-42}"
local_attn_size="${LOCAL_ATTN_SIZE:-180}"
seed="${SEED:-0}"
gpu_ids="${GPU_IDS:-0}"
implementations="${IMPLEMENTATIONS:-trq,student}"
profiles="${PROFILES:-mine:2:4:64:1560,student:2:4:64:448}"
student_root="${qvg_root}/references/quant-videogen"
student_ckpt="${STUDENT_CKPT_PATH:-${CKPT_PATH:-${SELF_FORCING_CKPT_ROOT:-${trq_root}/ckpts/Self-Forcing}/self_forcing_dmd.pt}}"
student_wan_dir="${FORCING_WAN_MODEL_DIR:-${SELF_FORCING_CKPT_ROOT:-${trq_root}/ckpts/Self-Forcing}/Wan2.1-T2V-1.3B}"

if [ ! -f "${student_root}/experiments/Self-Forcing/inference.py" ]; then
  echo "ERROR: Student QVG checkout not found at ${qvg_root}" >&2
  echo "Clone it with: git clone https://github.com/jiahui1021/qvg.git ${qvg_root}" >&2
  exit 1
fi
if [ ! -f "${prompts_path}" ]; then
  echo "ERROR: Prompt file not found: ${prompts_path}" >&2
  exit 1
fi
mkdir -p "${results_root}"

run_or_print() {
  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

contains_implementation() {
  [[ ",${implementations}," == *",$1,"* ]]
}

IFS=',' read -r -a profile_values <<< "${profiles}"
for profile in "${profile_values[@]}"; do
  IFS=':' read -r label bits anchor group stride extra <<< "${profile}"
  if [ -z "${label}" ] || [ -z "${stride}" ] || [ -n "${extra:-}" ]; then
    echo "ERROR: Invalid profile ${profile}; expected NAME:BITS:ANCHOR_BITS:GROUP_SIZE:STRIDE" >&2
    exit 2
  fi

  if contains_implementation trq; then
    out_dir="${results_root}/trq_${label}_identity"
    mkdir -p "${out_dir}"
    {
      echo "implementation=trq"
      echo "profile=${profile}"
      echo "predictor=identity"
      echo "residual_quant_mode=asym_zero_point"
      echo "scale_precision=bf16"
      echo "seed=${seed}"
      echo "num_output_frames=${num_frames}"
      echo "local_attn_size=${local_attn_size}"
      echo "prompts_path=${prompts_path}"
    } > "${out_dir}/resolved_env.txt"
    run_or_print env \
      PROMPTS_PATH="${prompts_path}" \
      NUM_OUTPUT_FRAMES="${num_frames}" \
      LOCAL_ATTN_SIZE="${local_attn_size}" \
      SEED="${seed}" \
      TRQ_BITS="${bits}" \
      TRQ_ANCHOR_BITS="${anchor}" \
      TRQ_GROUP_SIZE="${group}" \
      QUANT_BLOCK_SIZE="${group}" \
      TRQ_PREDICTOR_STRIDE="${stride}" \
      TRQ_SCALE_PRECISION=bf16 \
      TRQ_RESIDUAL_QUANT_MODE=asym_zero_point \
      HEADWISE_MODE=none \
      NUM_HIGH_PRECISION_HEADS=0 \
      TRQ_PARITY_CAPTURE_DIR="${out_dir}/parity_snapshots" \
      TRQ_PARITY_CAPTURE_LAYERS="${CAPTURE_LAYERS:-0}" \
      OUTPUT_FOLDER="${out_dir}" \
      bash "${trq_root}/scripts/self_forcing/run_hrq_predictor_ablation.sh" identity
  fi

  if contains_implementation student; then
    out_dir="${results_root}/student_${label}_identity"
    mkdir -p "${out_dir}"
    {
      echo "implementation=student_s2pp"
      echo "profile=${profile}"
      echo "predictor=identity"
      echo "residual_quant_mode=asym_zero_point"
      echo "scale_precision=bf16"
      echo "student_triton=${STUDENT_TRITON:-0}"
      echo "seed=${seed}"
      echo "num_output_frames=${num_frames}"
      echo "local_attn_size=${local_attn_size}"
      echo "prompts_path=${prompts_path}"
      echo "checkpoint_path=${student_ckpt}"
      echo "wan_model_dir=${student_wan_dir}"
    } > "${out_dir}/resolved_env.txt"
    run_or_print env \
      CUDA_VISIBLE_DEVICES="${gpu_ids}" \
      PYTHONPATH="${student_root}/experiments/Self-Forcing:${student_root}:${PYTHONPATH:-}" \
      FORCING_WAN_MODEL_DIR="${student_wan_dir}" \
      S2PP_USE_TRITON="${STUDENT_TRITON:-0}" \
      S2PP_TRITON_STRICT="${STUDENT_TRITON:-0}" \
      PYTHONUNBUFFERED=1 \
      torchrun --nproc_per_node=1 --standalone \
      "${student_root}/experiments/Self-Forcing/inference.py" \
      --config_path "${student_root}/experiments/Self-Forcing/configs/self_forcing_dmd.yaml" \
      --checkpoint_path "${student_ckpt}" \
      --data_path "${prompts_path}" \
      --output_folder "${out_dir}" \
      --num_samples 1 \
      --start_index 0 \
      --end_index -1 \
      --num_output_frames "${num_frames}" \
      --local_attn_size "${local_attn_size}" \
      --seed "${seed}" \
      --use_ema \
      --save_with_index \
      --quant_type "s2pp-int${bits}" \
      --quant_block_size "${group}" \
      --s2pp_group_size "${group}" \
      --s2pp_anchor_bits "${anchor}" \
      --s2pp_predictor_stride "${stride}" \
      --s2pp_affine_path "" \
      --s2pp_v_affine_path "" \
      --s2pp_predictor_mode identity \
      --s2pp_v_predictor_mode identity \
      --s2pp_scale_precision bf16 \
      --s2pp_residual_quant_mode asym_zero_point
  fi
done

echo "Identity online matrix complete: ${results_root}"
