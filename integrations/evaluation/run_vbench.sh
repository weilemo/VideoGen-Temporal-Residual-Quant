#!/usr/bin/env bash
set -euo pipefail

baseline="${1:?usage: run_vbench.sh rollingforcing|hy_worldplay|longcat|causal_forcing}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
prompt_file="${PROMPT_FILE:-${repo_root}/integrations/evaluation/moviegen10.txt}"
eval_script="${repo_root}/forcing/vbench/vbench2_beta_long/eval_long.py"
output_root="${OUTPUT_ROOT:-${repo_root}/results/world_model_quant/vbench/${baseline}}"
expected_videos="${EXPECTED_VIDEOS:-10}"

case "${baseline}" in
  rollingforcing)
    video_root="${repo_root}/results/world_model_quant/rollingforcing/moviegen10"
    ;;
  hy_worldplay)
    video_root="${repo_root}/results/world_model_quant/hy_worldplay/moviegen10/canonical"
    ;;
  longcat)
    video_root="${repo_root}/results/world_model_quant/longcat/moviegen10"
    ;;
  causal_forcing)
    video_root="${repo_root}/results/world_model_quant/causal_forcing/moviegen10"
    ;;
  *)
    echo "unknown baseline: ${baseline}" >&2
    exit 2
    ;;
esac
video_root="${VIDEO_ROOT:-${video_root}}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate vbench
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${HOME}/storage/cache/vbench}"
dimensions=(
  subject_consistency
  background_consistency
  motion_smoothness
  dynamic_degree
  aesthetic_quality
  imaging_quality
  overall_consistency
  clip_score
)
variants=(bf16 trq_int4 trq_int2 naive_int4 naive_int2)
mkdir -p "${output_root}/inputs" "${output_root}/scores" "${output_root}/completed"

for variant in "${variants[@]}"; do
  marker="${output_root}/completed/${variant}.done"
  if [ -f "${marker}" ]; then
    continue
  fi
  named_dir="${output_root}/inputs/${variant}"
  python "${script_dir}/prepare_vbench.py" \
    --src "${video_root}/${variant}" \
    --dst "${named_dir}" \
    --expected-videos "${expected_videos}"
  for dimension in "${dimensions[@]}"; do
    python "${eval_script}" \
      --videos_path "${named_dir}" \
      --name "${baseline}_${variant}" \
      --dimension "${dimension}" \
      --mode long_custom_input \
      --prompt_file "${prompt_file}" \
      --output_path "${output_root}/scores"
  done
  touch "${marker}"
done
