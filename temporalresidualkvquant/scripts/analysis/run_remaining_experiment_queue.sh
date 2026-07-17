#!/bin/bash
# Unattended serial queue for the remaining 2026-07-17 TRQ experiments.

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trq_root="$(cd "${script_dir}/../.." && pwd)"
run_root="${RUN_ROOT:-${HOME}/storage/runs/trq_remaining_20260718}"
status_dir="${run_root}/queue_status"
gpu_id="${GPU_ID:-3}"
ckpt_root="${SELF_FORCING_CKPT_ROOT:-${HOME}/storage/models/Self-Forcing}"
ckpt_path="${CKPT_PATH:-${ckpt_root}/checkpoints/self_forcing_dmd.pt}"
smoke_prompts="${PROMPTS_PATH:-${trq_root}/assets/mb32_paired_smoke4.txt}"
mb32_prompts="${MB32_PROMPTS_PATH:-${trq_root}/assets/moviegenbench_32.txt}"

mkdir -p "${run_root}" "${status_dir}"
exec > >(tee -a "${run_root}/queue.log") 2>&1

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export SELF_FORCING_CKPT_ROOT="${ckpt_root}"
export CKPT_PATH="${ckpt_path}"
export PYTHONPATH="${trq_root}/src:${trq_root}/backends/self_forcing:${PYTHONPATH:-}"

status_file="${status_dir}/stages.tsv"
touch "${status_file}"

timestamp() { date -Iseconds; }

record() {
  printf '%s\t%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" "${3:-}" >> "${status_file}"
}

run_stage() {
  local name="$1"
  shift
  local done_marker="${status_dir}/${name}.done"
  local failed_marker="${status_dir}/${name}.failed"
  if [ -f "${done_marker}" ]; then
    record "${name}" "SKIPPED_ALREADY_DONE"
    return 0
  fi
  rm -f "${failed_marker}"
  record "${name}" "RUNNING"
  echo "[$(timestamp)] START ${name}"
  if "$@" > >(tee -a "${status_dir}/${name}.log") 2>&1; then
    touch "${done_marker}"
    record "${name}" "PASSED"
    echo "[$(timestamp)] PASS ${name}"
    return 0
  else
    local rc=$?
    printf '%s\n' "${rc}" > "${failed_marker}"
    record "${name}" "FAILED" "exit=${rc}"
    echo "[$(timestamp)] FAIL ${name} exit=${rc}"
    return "${rc}"
  fi
}

skip_stage() {
  local name="$1"
  local reason="$2"
  record "${name}" "SKIPPED_MISSING_PREREQUISITE" "${reason}"
  printf '%s\n' "${reason}" > "${status_dir}/${name}.skipped"
}

run_trq() {
  local frames="$1" prompts="$2" seed="$3" k_bits="$4" v_bits="$5" output="$6"
  PROMPTS_PATH="${prompts}" NUM_OUTPUT_FRAMES="${frames}" LOCAL_ATTN_SIZE=180 SEED="${seed}" \
  HEADWISE_MODE=none TRQ_BITS=2 TRQ_K_BITS="${k_bits}" TRQ_V_BITS="${v_bits}" \
  PROFILE_RUNTIME=1 SAVE_ROLLOUT_LATENTS=1 OUTPUT_FOLDER="${output}" \
  ROLLOUT_METRICS_DIR="${output}/rollout_metrics" \
    bash "${trq_root}/scripts/self_forcing/run_hrq_predictor_ablation.sh" identity
}

run_bf16() {
  local frames="$1" prompts="$2" seed="$3" output="$4"
  PROMPTS_PATH="${prompts}" NUM_OUTPUT_FRAMES="${frames}" LOCAL_ATTN_SIZE=180 SEED="${seed}" \
  PROFILE_RUNTIME=1 SAVE_ROLLOUT_LATENTS=1 OUTPUT_FOLDER="${output}" \
  ROLLOUT_METRICS_DIR="${output}/rollout_metrics" \
    bash "${trq_root}/scripts/self_forcing/run_bf16.sh"
}

analyze_pair() {
  local bf16_a="$1" bf16_b="$2" trq="$3" output="$4"
  python "${trq_root}/scripts/analysis/analyze_online_paired_rollout.py" \
    --bf16-a "${bf16_a}" --bf16-b "${bf16_b}" --trq "${trq}" --output-dir "${output}"
}

check_prerequisites() {
  test -f "${ckpt_path}"
  test -f "${smoke_prompts}"
  test -f "${mb32_prompts}"
  test "$(wc -l < "${smoke_prompts}")" -eq 4
  test "$(wc -l < "${mb32_prompts}")" -ge 32
}

stage_k4v4_smoke() {
  local root="${run_root}/e3_k4v4_smoke"
  run_trq 180 "${smoke_prompts}" 0 4 4 "${root}/trq_k4v4_s0" &&
  run_trq 180 "${smoke_prompts}" 1 4 4 "${root}/trq_k4v4_s1"
}

stage_k4v4_analysis() {
  analyze_pair \
    "${HOME}/storage/runs/trq_online_paired/smoke4_20260717/bf16_a_s*/rollout_metrics/latents/*.pt" \
    "${HOME}/storage/runs/trq_online_paired/smoke4_20260717/bf16_b_s*/rollout_metrics/latents/*.pt" \
    "${run_root}/e3_k4v4_smoke/trq_k4v4_s*/rollout_metrics/latents/*.pt" \
    "${run_root}/e3_k4v4_smoke/analysis"
}

stage_e2_collect() {
  TRAIN_N=0 MAX_PROMPTS=0 PROMPTS_PATH="${mb32_prompts}" \
  KV_DUMP_ROOT="${run_root}/e2_mb32_layers8_19/kv_dumps" \
  KV_DUMP_FORMAT=layer_shards KV_DUMP_LAYERS=8-19 \
  NUM_OUTPUT_FRAMES=180 LOCAL_ATTN_SIZE=180 \
    bash "${trq_root}/scripts/self_forcing/collect_kv_dumps.sh"
}

stage_e2_analyze() {
  DUMPS_GLOB="${run_root}/e2_mb32_layers8_19/kv_dumps/heldout/*_layer*.pt" \
  LAYERS=8-19 KV=both NUM_BITS=2 ANCHOR_BITS=4 BLOCK_SIZE=64 \
  PREDICTOR_STRIDE=1560 BOOTSTRAP_RESAMPLES=2000 DEVICE=cpu \
  OUTPUT_DIR="${run_root}/e2_mb32_layers8_19/analysis" \
    bash "${trq_root}/scripts/analysis/run_trq_diagnostics.sh"
}

stage_e5_183() {
  local root="${run_root}/e5_eviction_183"
  local prompt="${root}/prompt1.txt"
  mkdir -p "${root}"
  sed -n '1p' "${smoke_prompts}" > "${prompt}"
  run_bf16 183 "${prompt}" 0 "${root}/bf16_a_s0" &&
  run_bf16 183 "${prompt}" 0 "${root}/bf16_b_s0" &&
  run_trq 183 "${prompt}" 0 4 4 "${root}/trq_k4v4_s0"
}

stage_e5_183_analysis() {
  local root="${run_root}/e5_eviction_183"
  analyze_pair \
    "${root}/bf16_a_s*/rollout_metrics/latents/*.pt" \
    "${root}/bf16_b_s*/rollout_metrics/latents/*.pt" \
    "${root}/trq_k4v4_s*/rollout_metrics/latents/*.pt" \
    "${root}/analysis"
}

latent_gate_passes() {
  local summary="$1"
  python - "${summary}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
raise SystemExit(0 if payload.get("passes_latent_drift_gate") is True else 1)
PY
}

stage_e5_501() {
  local root="${run_root}/e5_long_501"
  run_bf16 501 "${smoke_prompts}" 0 "${root}/bf16_s0" &&
  run_trq 501 "${smoke_prompts}" 0 2 2 "${root}/trq_k2v2_s0" &&
  run_trq 501 "${smoke_prompts}" 0 4 4 "${root}/trq_k4v4_s0"
}

stage_e5_699() {
  local root="${run_root}/e5_long_699"
  local repeat_prompts="${root}/repeat_prompts_1_23.txt"
  mkdir -p "${root}"
  sed -n '1p;4p' "${smoke_prompts}" > "${repeat_prompts}"
  run_bf16 699 "${smoke_prompts}" 0 "${root}/bf16_s0" &&
  run_trq 699 "${smoke_prompts}" 0 4 4 "${root}/trq_k4v4_s0" &&
  run_bf16 699 "${repeat_prompts}" 0 "${root}/bf16_repeat_s0"
}

stage_efficiency() {
  python "${trq_root}/scripts/analysis/summarize_runtime_metrics.py" \
    --roots \
      "${HOME}/storage/runs/trq_online_paired/smoke4_20260717" \
      "${HOME}/storage/runs/trq_online_asym/smoke4_20260717" \
      "${run_root}" \
    --output-dir "${run_root}/e6_efficiency"
}

echo "TRQ unattended queue"
echo "root=${trq_root}"
echo "run_root=${run_root}"
echo "physical_gpu=${gpu_id}"
echo "commit=$(git -C "${trq_root}" rev-parse HEAD 2>/dev/null || true)"

if ! run_stage prerequisites check_prerequisites; then
  echo "Required files are missing; queue cannot start."
  exit 2
fi

run_stage e3_k4v4_smoke stage_k4v4_smoke || true
if [ -f "${status_dir}/e3_k4v4_smoke.done" ]; then
  run_stage e3_k4v4_analysis stage_k4v4_analysis || true
else
  skip_stage e3_k4v4_analysis "K4V4 generation failed"
fi

run_stage e2_collect_mb32_layers8_19 stage_e2_collect || true
if [ -f "${status_dir}/e2_collect_mb32_layers8_19.done" ]; then
  run_stage e2_analyze_mb32_layers8_19 stage_e2_analyze || true
else
  skip_stage e2_analyze_mb32_layers8_19 "MB32 layer dump collection failed"
fi

run_stage e5_eviction_183 stage_e5_183 || true
if [ -f "${status_dir}/e5_eviction_183.done" ]; then
  run_stage e5_eviction_183_analysis stage_e5_183_analysis || true
else
  skip_stage e5_eviction_183_analysis "183-frame generation failed"
fi

long_gate_summary="${run_root}/e5_eviction_183/analysis/summary.json"
if [ -f "${long_gate_summary}" ] && latent_gate_passes "${long_gate_summary}"; then
  run_stage e5_long_501 stage_e5_501 || true
  if [ -f "${status_dir}/e5_long_501.done" ]; then
    run_stage e5_long_699 stage_e5_699 || true
  else
    skip_stage e5_long_699 "501-frame stage failed"
  fi
else
  skip_stage e5_long_501 "183-frame latent drift gate did not pass"
  skip_stage e5_long_699 "501-frame prerequisite was not run"
fi

run_stage e6_efficiency stage_efficiency || true
skip_stage e4_other_backends "Rolling-Forcing, LongCat, and HY-WorldPlay adapters/repos are absent"
skip_stage cross_kv "Cross-KV is explicitly unsupported by the stable TRQ v1 codec"

record queue COMPLETE
echo "[$(timestamp)] QUEUE COMPLETE"
