#!/bin/bash
# Evaluate indexed paired videos listed as: config<TAB>seed<TAB>label<TAB>video_dir.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trq_root="$(cd "${script_dir}/../.." && pwd)"
runs_tsv="${E1_RUNS_TSV:?Set E1_RUNS_TSV to the four-column run manifest}"
vbench_root="${VBENCH_ROOT:?Set VBENCH_ROOT to the VBench checkout}"
prompt_file="${PROMPTS_PATH:-${trq_root}/assets/mb32_paired_smoke4.txt}"
output_root="${OUTPUT_ROOT:-${trq_root}/results/online_causal/e1_vbench}"
eval_script="${VBENCH_EVAL_SCRIPT:-${vbench_root}/vbench2_beta_long/eval_long.py}"
gpu_id="${GPU_ID:-0}"

case "${gpu_id}" in
  0|1) ;;
  *) echo "ERROR: GPU_ID must be 0 or 1 for this lease, got ${gpu_id}" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="${gpu_id}"

dimensions=(
  subject_consistency background_consistency motion_smoothness dynamic_degree
  aesthetic_quality imaging_quality overall_consistency clip_score
)

mkdir -p "${output_root}/inputs" "${output_root}/scores"
: > "${output_root}/completed_runs.tsv"
while IFS=$'\t' read -r config seed label video_dir; do
  if [ -z "${config}" ] || [[ "${config}" = \#* ]]; then
    continue
  fi
  named_dir="${output_root}/inputs/${label}"
  python "${script_dir}/prepare_vbench_named_dir.py" --src "${video_dir}" --dst "${named_dir}"
  for dimension in "${dimensions[@]}"; do
    python "${eval_script}" \
      --videos_path "${named_dir}" \
      --name "${label}" \
      --dimension "${dimension}" \
      --mode long_custom_input \
      --prompt_file "${prompt_file}" \
      --output_path "${output_root}/scores"
  done
  printf '%s\t%s\t%s\t%s\n' "${config}" "${seed}" "${label}" "${output_root}/scores" \
    >> "${output_root}/completed_runs.tsv"
done < "${runs_tsv}"

echo "Paired VBench complete: ${output_root}"
