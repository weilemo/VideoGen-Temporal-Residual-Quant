#!/bin/bash
# Offline raw-vs-residual and residual-chain-drift diagnostics.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hwq_root="$(cd "${script_dir}/../.." && pwd)"

experiments="${EXPERIMENTS:-all}"
predictor_mode="${PREDICTOR_MODE:-identity}"
num_bits="${NUM_BITS:-2}"
anchor_bits="${ANCHOR_BITS:-4}"
block_size="${BLOCK_SIZE:-64}"
predictor_stride="${PREDICTOR_STRIDE:-1560}"
runtime_reset_interval="${RUNTIME_RESET_INTERVAL:-24}"
reset_intervals="${RESET_INTERVALS:-1,2,4,8,24,none}"
layers="${LAYERS:-all}"
kv="${KV:-both}"
device="${DEVICE:-cpu}"
max_dumps="${MAX_DUMPS:-0}"
sample_capacity="${SAMPLE_CAPACITY:-250000}"
bootstrap_resamples="${BOOTSTRAP_RESAMPLES:-2000}"
seed="${SEED:-0}"
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${predictor_mode}_int${num_bits}}"

kv_dump_root="${KV_DUMP_ROOT:-${hwq_root}/kv_dumps/trq_diagnostics/${run_id}}"
output_dir="${OUTPUT_DIR:-${hwq_root}/results/trq_diagnostics/${run_id}}"
dumps_glob="${DUMPS_GLOB:-${kv_dump_root}/heldout/*.pt}"
predictor_path="${PREDICTOR_PARAMS_PATH:-${hwq_root}/assets/trq_predictors/affine_channel_self_forcing_dmd.pt}"

export PYTHONPATH="${hwq_root}/src:${hwq_root}/backends/self_forcing:${PYTHONPATH:-}"

if [ "${COLLECT_DUMPS:-0}" = "1" ]; then
  echo "Collecting raw BF16 dumps first. Default 4-prompt collection needs roughly 200 GiB."
  PROMPTS_PATH="${PROMPTS_PATH:-${hwq_root}/assets/moviegenbench_32.txt}" \
  MAX_PROMPTS="${MAX_PROMPTS:-4}" \
  TRAIN_N="${TRAIN_N:-2}" \
  KV_DUMP_FORMAT="${KV_DUMP_FORMAT:-layer_shards}" \
  KV_DUMP_LAYERS="${DUMP_LAYERS:-all}" \
  KV_DUMP_ROOT="${kv_dump_root}" \
    bash "${hwq_root}/scripts/self_forcing/collect_kv_dumps.sh"
fi

if ! compgen -G "${dumps_glob}" >/dev/null && [[ "${dumps_glob}" != *,* ]]; then
  echo "ERROR: No held-out KV dumps matched: ${dumps_glob}" >&2
  echo "Run with COLLECT_DUMPS=1 after the GPU machine and checkpoint are ready." >&2
  exit 1
fi

mkdir -p "${output_dir}"
args=(
  --dumps "${dumps_glob}"
  --output-dir "${output_dir}"
  --experiments "${experiments}"
  --layers "${layers}"
  --kv "${kv}"
  --max-dumps "${max_dumps}"
  --num-bits "${num_bits}"
  --anchor-bits "${anchor_bits}"
  --block-size "${block_size}"
  --predictor-stride "${predictor_stride}"
  --predictor-mode "${predictor_mode}"
  --scale-precision "${SCALE_PRECISION:-bf16}"
  --codec-dtype "${CODEC_DTYPE:-bf16}"
  --device "${device}"
  --runtime-reset-interval "${runtime_reset_interval}"
  --reset-intervals "${reset_intervals}"
  --sample-capacity "${sample_capacity}"
  --bootstrap-resamples "${bootstrap_resamples}"
  --seed "${seed}"
)

if [ "${predictor_mode}" = "affine_channel" ]; then
  args+=(--predictor-params-path "${predictor_path}")
fi
if [ "${SKIP_DECODER_PARITY:-0}" = "1" ]; then
  args+=(--skip-decoder-parity)
fi
if [ "${NO_PLOTS:-0}" = "1" ]; then
  args+=(--no-plots)
fi

echo "Dumps:       ${dumps_glob}"
echo "Output:      ${output_dir}"
echo "Experiments: ${experiments}"
echo "Predictor:   ${predictor_mode}"
echo "Layers/KV:   ${layers} / ${kv}"

python "${hwq_root}/scripts/analysis/analyze_trq_diagnostics.py" "${args[@]}" "$@" \
  2>&1 | tee "${output_dir}/run.log"
