#!/bin/bash
# Generate matched BF16 baselines from the unified and student Self-Forcing backends.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trq_root="$(cd "${script_dir}/../.." && pwd)"
repo_root="$(cd "${trq_root}/.." && pwd)"
qvg_root="${QVG_ROOT:-$(cd "${repo_root}/.." && pwd)/qvg}"
student_root="${qvg_root}/references/quant-videogen"
output_root="${OUTPUT_ROOT:-${trq_root}/results/bf16_cross_repo}"
prompts_path="${PROMPTS_PATH:-${trq_root}/assets/t2v.txt}"
num_frames="${NUM_OUTPUT_FRAMES:-42}"
local_attn_size="${LOCAL_ATTN_SIZE:-180}"
seed="${SEED:-0}"
gpu_ids="${GPU_IDS:-0}"
student_ckpt="${STUDENT_CKPT_PATH:-${CKPT_PATH:-${SELF_FORCING_CKPT_ROOT:-${trq_root}/ckpts/Self-Forcing}/self_forcing_dmd.pt}}"
student_wan_dir="${FORCING_WAN_MODEL_DIR:-${SELF_FORCING_CKPT_ROOT:-${trq_root}/ckpts/Self-Forcing}/Wan2.1-T2V-1.3B}"
trq_output="${output_root}/trq_bf16"
student_output="${output_root}/student_bf16"

if [ ! -f "${student_root}/experiments/Self-Forcing/inference.py" ]; then
  echo "ERROR: Student QVG checkout not found at ${qvg_root}" >&2
  exit 1
fi
if [ ! -f "${prompts_path}" ]; then
  echo "ERROR: Prompt file not found: ${prompts_path}" >&2
  exit 1
fi

mkdir -p "${trq_output}" "${student_output}"

run_or_print() {
  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

for implementation in trq student; do
  output="${trq_output}"
  [ "${implementation}" = "student" ] && output="${student_output}"
  {
    echo "implementation=${implementation}"
    echo "quant_type=none"
    echo "seed=${seed}"
    echo "num_output_frames=${num_frames}"
    echo "local_attn_size=${local_attn_size}"
    echo "prompts_path=${prompts_path}"
    echo "checkpoint_path=${student_ckpt}"
    echo "wan_model_dir=${student_wan_dir}"
  } > "${output}/resolved_env.txt"
done

run_or_print env \
  PROMPTS_PATH="${prompts_path}" \
  NUM_OUTPUT_FRAMES="${num_frames}" \
  LOCAL_ATTN_SIZE="${local_attn_size}" \
  SEED="${seed}" \
  OUTPUT_FOLDER="${trq_output}" \
  bash "${trq_root}/scripts/self_forcing/run_bf16.sh"

run_or_print env \
  CUDA_VISIBLE_DEVICES="${gpu_ids}" \
  PYTHONPATH="${student_root}/experiments/Self-Forcing:${student_root}:${PYTHONPATH:-}" \
  FORCING_WAN_MODEL_DIR="${student_wan_dir}" \
  PYTHONUNBUFFERED=1 \
  torchrun --nproc_per_node=1 --standalone \
  "${student_root}/experiments/Self-Forcing/inference.py" \
  --config_path "${student_root}/experiments/Self-Forcing/configs/self_forcing_dmd.yaml" \
  --checkpoint_path "${student_ckpt}" \
  --data_path "${prompts_path}" \
  --output_folder "${student_output}" \
  --num_samples 1 \
  --start_index 0 \
  --end_index -1 \
  --num_output_frames "${num_frames}" \
  --local_attn_size "${local_attn_size}" \
  --seed "${seed}" \
  --use_ema \
  --save_with_index \
  --quant_type none

if [ "${COMPARE_VIDEOS:-1}" = "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  python "${trq_root}/scripts/eval/eval_ref_metrics.py" \
    --ref-dir "${trq_output}" \
    --cmp-dir "${student_output}" \
    --out "${output_root}/video_parity.json" \
    --device "${DEVICE:-cuda}" \
    --match-by-index \
    --strict-shape
fi

echo "BF16 cross-repo outputs: ${output_root}"
