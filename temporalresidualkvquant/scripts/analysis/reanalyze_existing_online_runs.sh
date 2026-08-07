#!/bin/bash
# E0: recompute absolute/growth/boundary metrics for the 2026-07-17/18 runs.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trq_root="$(cd "${script_dir}/../.." && pwd)"
runs_root="${RUNS_ROOT:-${HOME}/storage/runs}"
output_root="${OUTPUT_ROOT:-${runs_root}/trq_causal_20260720/e0}"
bf16_a="${runs_root}/trq_online_paired/smoke4_20260717/bf16_a_s*/rollout_metrics/latents/*.pt"
bf16_b="${runs_root}/trq_online_paired/smoke4_20260717/bf16_b_s*/rollout_metrics/latents/*.pt"

configs=(k2v2 k2v4 k4v2 k4v4)
latents=(
  "${runs_root}/trq_online_paired/smoke4_20260717/trq_k2v2_s*/rollout_metrics/latents/*.pt"
  "${runs_root}/trq_online_asym/smoke4_20260717/trq_k2v4_s*/rollout_metrics/latents/*.pt"
  "${runs_root}/trq_online_asym/smoke4_20260717/trq_k4v2_s*/rollout_metrics/latents/*.pt"
  "${runs_root}/trq_remaining_20260718/e3_k4v4_smoke/trq_k4v4_s*/rollout_metrics/latents/*.pt"
)

export PYTHONPATH="${trq_root}/src:${PYTHONPATH:-}"
for index in "${!configs[@]}"; do
  config="${configs[$index]}"
  python "${script_dir}/analyze_online_paired_rollout.py" \
    --bf16-a "${bf16_a}" \
    --bf16-b "${bf16_b}" \
    --trq "${latents[$index]}" \
    --config-label "${config}" \
    --first-quant-frame 24 \
    --output-dir "${output_root}/${config}"
done

python "${script_dir}/aggregate_online_metrics.py" \
  --analysis "k2v2=${output_root}/k2v2" \
  --analysis "k2v4=${output_root}/k2v4" \
  --analysis "k4v2=${output_root}/k4v2" \
  --analysis "k4v4=${output_root}/k4v4" \
  --output "${output_root}/online_config_summary.csv"

echo "E0 complete: ${output_root}"
