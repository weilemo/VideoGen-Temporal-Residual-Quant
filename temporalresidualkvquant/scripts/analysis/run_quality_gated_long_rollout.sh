#!/bin/bash
# E5: stage 183, 501, and 699 separately; every expansion fails closed on quality.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trq_root="$(cd "${script_dir}/../.." && pwd)"
gpu_id="${GPU_ID:-0}"
stage="${STAGE:?Set STAGE to 183, 501, or 699}"
winner="${WINNER_CONFIG:?Set WINNER_CONFIG to the E4 config name}"
quality_gate="${QUALITY_GATE_JSON:?Set QUALITY_GATE_JSON to E1/E4 quality_gate.json}"
run_root="${RUN_ROOT:-${HOME}/storage/runs/trq_causal_20260720/e5}"
prompts="${PROMPTS_PATH:-${trq_root}/assets/mb32_paired_smoke4.txt}"

case "${gpu_id}" in 0|1) ;; *) echo "ERROR: GPU_ID must be 0 or 1" >&2; exit 2 ;; esac
export CUDA_VISIBLE_DEVICES="${gpu_id}"

python - "${quality_gate}" "${winner}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
status = payload.get("configs", {}).get(sys.argv[2], {}).get("status")
if status != "PASS":
    raise SystemExit(f"quality gate for {sys.argv[2]} is {status!r}, expected PASS")
PY

run_bf16() {
  local frames="$1" indices="$2" output="$3"
  PROMPTS_PATH="${prompts}" PROMPT_INDICES="${indices}" NUM_OUTPUT_FRAMES="${frames}" \
  LOCAL_ATTN_SIZE=180 ATTENTION_SINK_FRAMES="${ATTENTION_SINK_FRAMES:-0}" SEED=0 \
  PROFILE_RUNTIME=1 SAVE_ROLLOUT_LATENTS=1 OUTPUT_FOLDER="${output}" \
  ROLLOUT_METRICS_DIR="${output}/rollout_metrics" \
    bash "${trq_root}/scripts/self_forcing/run_bf16.sh"
}

run_winner() {
  local frames="$1" indices="$2" output="$3"
  PROMPTS_PATH="${prompts}" PROMPT_INDICES="${indices}" NUM_OUTPUT_FRAMES="${frames}" \
  LOCAL_ATTN_SIZE=180 SEED=0 HEADWISE_MODE=none TRQ_BITS="${TRQ_BITS:-2}" \
  TRQ_K_BITS="${TRQ_K_BITS:-2}" TRQ_V_BITS="${TRQ_V_BITS:-2}" \
  TRQ_FIRST_QUANT_FRAME="${TRQ_FIRST_QUANT_FRAME:-24}" \
  TRQ_QUANT_INTERVAL_FRAMES="${TRQ_QUANT_INTERVAL_FRAMES:-24}" \
  TRQ_QUANT_SCHEDULE="${TRQ_QUANT_SCHEDULE:-bulk}" TRQ_GRADUAL_FRAMES="${TRQ_GRADUAL_FRAMES:-3}" \
  TRQ_PROTECTED_SINK_FRAMES="${TRQ_PROTECTED_SINK_FRAMES:-0}" \
  ATTENTION_SINK_FRAMES="${ATTENTION_SINK_FRAMES:-0}" TRQ_CACHE_ROLES="${TRQ_CACHE_ROLES:-both}" \
  TRQ_QUANTIZED_LAYERS="${TRQ_QUANTIZED_LAYERS:-all}" PROFILE_RUNTIME=1 \
  SAVE_ROLLOUT_LATENTS=1 OUTPUT_FOLDER="${output}" ROLLOUT_METRICS_DIR="${output}/rollout_metrics" \
    bash "${trq_root}/scripts/self_forcing/run_hrq_predictor_ablation.sh" identity
}

case "${stage}" in
  183)
    indices=0
    root="${run_root}/183"
    run_bf16 183 "${indices}" "${root}/bf16_a"
    run_winner 183 "${indices}" "${root}/${winner}"
    ;;
  501)
    root="${run_root}/501"
    run_bf16 501 0,1,2,3 "${root}/bf16_a"
    run_winner 501 0,1,2,3 "${root}/${winner}"
    python "${trq_root}/scripts/eval/prepare_windowed_videos.py" \
      --src "${root}/bf16_a" --dst "${root}/windows/bf16_a" --latent-frames 501
    python "${trq_root}/scripts/eval/prepare_windowed_videos.py" \
      --src "${root}/${winner}" --dst "${root}/windows/${winner}" --latent-frames 501
    ;;
  699)
    gate_501="${E5_501_QUALITY_GATE_JSON:?Set E5_501_QUALITY_GATE_JSON before 699}"
    python - "${gate_501}" "${winner}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("configs", {}).get(sys.argv[2], {}).get("status") != "PASS":
    raise SystemExit("501 windowed quality gate did not pass")
PY
    root="${run_root}/699"
    run_bf16 699 0,1,2,3 "${root}/bf16_a"
    run_winner 699 0,1,2,3 "${root}/${winner}"
    run_bf16 699 0,3 "${root}/bf16_b"
    run_winner 699 0,3 "${root}/${winner}_repeat"
    python "${trq_root}/scripts/eval/prepare_windowed_videos.py" \
      --src "${root}/bf16_a" --dst "${root}/windows/bf16_a" --latent-frames 699
    python "${trq_root}/scripts/eval/prepare_windowed_videos.py" \
      --src "${root}/${winner}" --dst "${root}/windows/${winner}" --latent-frames 699
    ;;
  *) echo "ERROR: STAGE must be 183, 501, or 699" >&2; exit 2 ;;
esac

echo "E5 stage ${stage} complete: ${root}"
