#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trq_root="$(cd "${script_dir}/../.." && pwd)"
: "${E1_DIR:?Set E1_DIR to the PASS_E1 directory}"
: "${E2_CORE_DIR:?Set E2_CORE_DIR to the corrected PASS core E2 directory}"
: "${GPU_ID:?Set GPU_ID after verifying the active lease mapping}"

run_root="${RUN_ROOT:-${trq_root}/results/conditional_innovation/e2_subgates_$(date +%Y%m%d)}"
codec_gamma_dir="${run_root}/codec_gamma"
codec_e2_dir="${run_root}/codec_gamma_e2"
offline_json="${run_root}/offline_subgates.json"
attention_root="${run_root}/attention_probe"
summary_json="${run_root}/summary.json"
prompts="${PROMPTS_PATH:-${trq_root}/assets/mb32_paired_smoke4.txt}"
mkdir -p "${run_root}/logs" "${attention_root}"

python - "${E1_DIR}/summary.json" "${E2_CORE_DIR}/summary.json" <<'PY'
import json, pathlib, sys
e1 = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
e2 = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
if e1.get("status") != "PASS_E1":
    raise SystemExit(f"E1 is not PASS_E1: {e1.get('status')}")
if e2.get("gate2_core", {}).get("status") != "PASS":
    raise SystemExit(f"corrected E2 core is not PASS: {e2.get('gate2_core')}")
PY

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${trq_root}/src:${trq_root}/backends/self_forcing:${PYTHONPATH:-}"

if [[ ! -f "${codec_gamma_dir}/codec_gamma_summary.json" ]]; then
  python "${script_dir}/calibrate_codec_gamma.py" \
    --e1-dir "${E1_DIR}" --output-dir "${codec_gamma_dir}" --layers "${LAYERS:-8-19}" \
    --key-bits 4 --value-bits 4 --anchor-bits 4 --block-size 64 \
    2>&1 | tee "${run_root}/logs/codec_gamma.log"
fi

run_codec_validation() {
  if [[ -f "${codec_e2_dir}/analysis.complete" ]]; then return 0; fi
  E1_DIR="${codec_gamma_dir}" OUTPUT_DIR="${codec_e2_dir}" LAYERS="${LAYERS:-8-19}" \
    KEY_BITS=4 VALUE_BITS=4 ANCHOR_BITS=4 BLOCK_SIZE=64 REQUIRE_CORE_PASS=1 \
    bash "${script_dir}/run_conditional_innovation_e2.sh"
}

run_offline_subgates() {
  if [[ -f "${offline_json}" ]]; then return 0; fi
  python "${script_dir}/evaluate_e2_offline_subgates.py" \
    --codec-gamma-dir "${codec_gamma_dir}" --output "${offline_json}" \
    --layers "${LAYERS:-8-19}" --key-bits 4 --value-bits 4 --anchor-bits 4 --block-size 64
}

run_codec_validation >"${run_root}/logs/codec_validation.log" 2>&1 & codec_pid=$!
run_offline_subgates >"${run_root}/logs/offline_subgates.log" 2>&1 & offline_pid=$!
codec_rc=0; wait "${codec_pid}" || codec_rc=$?
offline_rc=0; wait "${offline_pid}" || offline_rc=$?
if (( codec_rc != 0 || offline_rc != 0 )); then
  echo "E2 CPU subgate failed: codec=${codec_rc} offline=${offline_rc}" >&2
  exit 2
fi

run_attention_probe() {
  local mode="$1" params="$2" output="${attention_root}/${mode}"
  if [[ -f "${output}/probe.complete" ]]; then return 0; fi
  TRQ_QUANT_TYPE=s2pp-int4 TRQ_V_PREDICTOR_PARAMS_PATH="${params}" \
  PROMPTS_PATH="${prompts}" PROMPT_INDICES=0 NUM_OUTPUT_FRAMES=48 LOCAL_ATTN_SIZE=180 \
  SEED=0 HEADWISE_MODE=none TRQ_BITS=4 TRQ_K_BITS=4 TRQ_V_BITS=4 \
  TRQ_FIRST_QUANT_FRAME=24 TRQ_CACHE_ROLES=both TRQ_QUANTIZED_LAYERS="${LAYERS:-8-19}" \
  TRQ_ATTENTION_TRACE_DIR="${output}/attention_trace" TRQ_ATTENTION_TRACE_LAYERS="${LAYERS:-8-19}" \
  OUTPUT_FOLDER="${output}" PROFILE_RUNTIME=1 \
    bash "${trq_root}/scripts/self_forcing/run_hrq_predictor_ablation.sh" "${mode}"
  touch "${output}/probe.complete"
}

run_attention_probe cross_kv "${codec_gamma_dir}/cross_codec.npz" \
  >"${run_root}/logs/attention_cross.log" 2>&1
run_attention_probe hybrid_kv_innovation "${codec_gamma_dir}/hybrid_codec_gamma.npz" \
  >"${run_root}/logs/attention_hybrid.log" 2>&1

python "${script_dir}/summarize_e2_subgates.py" \
  --codec-gamma-summary "${codec_gamma_dir}/codec_gamma_summary.json" \
  --codec-e2-summary "${codec_e2_dir}/summary.json" \
  --offline-subgates "${offline_json}" \
  --cross-trace-dir "${attention_root}/cross_kv/attention_trace" \
  --hybrid-trace-dir "${attention_root}/hybrid_kv_innovation/attention_trace" \
  --output "${summary_json}"

for seed in 0 1; do
  output="${run_root}/e3_smoke4/hybrid/s${seed}"
  if [[ -f "${output}/e3.complete" ]]; then continue; fi
  TRQ_QUANT_TYPE=s2pp-int4 TRQ_V_PREDICTOR_PARAMS_PATH="${codec_gamma_dir}/hybrid_codec_gamma.npz" \
  PROMPTS_PATH="${prompts}" NUM_OUTPUT_FRAMES=180 LOCAL_ATTN_SIZE=180 SEED="${seed}" \
  HEADWISE_MODE=none TRQ_BITS=4 TRQ_K_BITS=4 TRQ_V_BITS=4 TRQ_FIRST_QUANT_FRAME=24 \
  TRQ_CACHE_ROLES=both TRQ_QUANTIZED_LAYERS="${LAYERS:-8-19}" \
  TRQ_ATTENTION_TRACE_DIR="${output}/attention_trace" TRQ_ATTENTION_TRACE_LAYERS="${LAYERS:-8-19}" \
  PROFILE_RUNTIME=1 SAVE_ROLLOUT_LATENTS=1 OUTPUT_FOLDER="${output}" \
  ROLLOUT_METRICS_DIR="${output}/rollout_metrics" \
    bash "${trq_root}/scripts/self_forcing/run_hrq_predictor_ablation.sh" hybrid_kv_innovation \
    >"${run_root}/logs/e3_s${seed}.log" 2>&1
  touch "${output}/e3.complete"
done
touch "${run_root}/e3.complete"
echo "PASS_E2 and E3 generation complete: ${run_root}"
