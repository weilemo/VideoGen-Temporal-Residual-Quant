#!/bin/bash
# Collect BF16 KV cache dumps for TRQ diagnostics and predictor fitting.
#
# Runs inference with quant_type=none and DUMP_KV_LEVEL=1, which saves
# raw BF16 KV tensors (as ChunkedKVCache objects) for offline predictor fitting.
#
# Split:  first  TRAIN_N prompts → kv_dumps/train/
#         remaining prompts      → kv_dumps/heldout/
#
# Usage:
#   bash collect_kv_dumps.sh
#
# Env overrides:
#   TRAIN_N              (default 2; zero sends all prompts to heldout)
#   PROMPTS_PATH         (default assets/moviegenbench_32.txt)
#   MAX_PROMPTS          (default 4; zero means all prompts)
#   KV_DUMP_ROOT         (default kv_dumps)
#   NUM_OUTPUT_FRAMES    (default 180)
#   LOCAL_ATTN_SIZE      (default 180)
#   KV_DUMP_FORMAT       (default chunked; diagnostics wrapper uses layer_shards)
#   KV_DUMP_LAYERS       (default all)

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hwq_root="$(cd "${script_dir}/../.." && pwd)"
self_forcing_root="${SELF_FORCING_ROOT:-${hwq_root}/backends/self_forcing}"

if [ -n "${SELF_FORCING_CKPT_ROOT:-}" ]; then
  ckpt_root="${SELF_FORCING_CKPT_ROOT}"
elif [ -d "${hwq_root}/ckpts/Self-Forcing" ]; then
  ckpt_root="${hwq_root}/ckpts/Self-Forcing"
else
  ckpt_root="${hwq_root}/ckpts/Self-Forcing"
fi
ckpt_path="${CKPT_PATH:-${ckpt_root}/self_forcing_dmd.pt}"

prompts_path="${PROMPTS_PATH:-${hwq_root}/assets/moviegenbench_32.txt}"
num_output_frames="${NUM_OUTPUT_FRAMES:-180}"
local_attn_size="${LOCAL_ATTN_SIZE:-180}"
train_n="${TRAIN_N:-2}"
max_prompts="${MAX_PROMPTS:-4}"
kv_dump_root="${KV_DUMP_ROOT:-${hwq_root}/kv_dumps}"
kv_dump_format="${KV_DUMP_FORMAT:-chunked}"
kv_dump_layers="${KV_DUMP_LAYERS:-all}"

if [ ! -f "${prompts_path}" ]; then
  echo "ERROR: Prompt file does not exist: ${prompts_path}" >&2
  exit 2
fi
if [ ! -f "${ckpt_path}" ]; then
  echo "ERROR: Checkpoint does not exist: ${ckpt_path}" >&2
  exit 2
fi
if [ "${kv_dump_format}" != "chunked" ] && [ "${kv_dump_format}" != "layer_shards" ]; then
  echo "ERROR: KV_DUMP_FORMAT must be chunked or layer_shards, got ${kv_dump_format}" >&2
  exit 2
fi

export PYTHONPATH="${hwq_root}/src:${self_forcing_root}:${PYTHONPATH:-}"
export SELF_FORCING_CKPT_ROOT="${ckpt_root}"

echo "HWQ root:      ${hwq_root}"
echo "Self-Forcing:  ${self_forcing_root}"
echo "Checkpoint:    ${ckpt_path}"
echo "Prompts:       ${prompts_path}"
echo "Train N:       ${train_n}"
echo "KV dump root:  ${kv_dump_root}"
echo "Dump format:   ${kv_dump_format}"
echo "Dump layers:   ${kv_dump_layers}"
echo ""

# Limit collection before splitting. Four 180-frame dumps need roughly 200 GiB.
mkdir -p "${kv_dump_root}"
selected_prompts="${kv_dump_root}/selected_prompts.txt"
if [ "${max_prompts}" -gt 0 ]; then
  head -n "${max_prompts}" "${prompts_path}" > "${selected_prompts}"
else
  cp "${prompts_path}" "${selected_prompts}"
fi
total_prompts=$(wc -l < "${selected_prompts}")
echo "Total prompts: ${total_prompts}"
if [ "${train_n}" -lt 0 ] || [ "${train_n}" -ge "${total_prompts}" ]; then
  echo "ERROR: TRAIN_N must be in [0, total_prompts-1], got ${train_n}/${total_prompts}" >&2
  exit 2
fi

# Split prompts into train and heldout temp files
train_prompts="${kv_dump_root}/tmp_train_prompts.txt"
heldout_prompts="${kv_dump_root}/tmp_heldout_prompts.txt"
if [ "${train_n}" -gt 0 ]; then
  head -n "${train_n}" "${selected_prompts}" > "${train_prompts}"
  tail -n +"$((train_n + 1))" "${selected_prompts}" > "${heldout_prompts}"
else
  : > "${train_prompts}"
  cp "${selected_prompts}" "${heldout_prompts}"
fi

heldout_n=$(wc -l < "${heldout_prompts}")
echo "Train prompts: ${train_n}, Heldout prompts: ${heldout_n}"
echo ""

# ── Run train dump ──────────────────────────────────────────────────
train_dump_dir="${kv_dump_root}/train"
if [ "${train_n}" -gt 0 ]; then
  echo "=== Collecting TRAIN KV dumps → ${train_dump_dir} ==="
  mkdir -p "${train_dump_dir}"

  DUMP_KV_LEVEL=1 KV_DUMP_DIR="${train_dump_dir}" \
  KV_DUMP_FORMAT="${kv_dump_format}" KV_DUMP_LAYERS="${kv_dump_layers}" \
    torchrun --nproc_per_node=1 --standalone "${self_forcing_root}/inference.py" \
      --config_path "${self_forcing_root}/configs/self_forcing_dmd.yaml" \
      --checkpoint_path "${ckpt_path}" \
      --data_path "${train_prompts}" \
      --output_folder "${kv_dump_root}/train_videos" \
      --num_samples 1 \
      --num_output_frames "${num_output_frames}" \
      --local_attn_size "${local_attn_size}" \
      --use_ema \
      --save_with_index \
      --quant_type "none"

  echo "Train KV dumps saved to: ${train_dump_dir}"
else
  echo "Skipping train dumps (TRAIN_N=0)."
fi

# ── Run heldout dump ────────────────────────────────────────────────
if [ "${heldout_n}" -gt 0 ]; then
  heldout_dump_dir="${kv_dump_root}/heldout"
  echo ""
  echo "=== Collecting HELDOUT KV dumps → ${heldout_dump_dir} ==="
  mkdir -p "${heldout_dump_dir}"

  DUMP_KV_LEVEL=1 KV_DUMP_DIR="${heldout_dump_dir}" \
  KV_DUMP_FORMAT="${kv_dump_format}" KV_DUMP_LAYERS="${kv_dump_layers}" \
    torchrun --nproc_per_node=1 --standalone "${self_forcing_root}/inference.py" \
      --config_path "${self_forcing_root}/configs/self_forcing_dmd.yaml" \
      --checkpoint_path "${ckpt_path}" \
      --data_path "${heldout_prompts}" \
      --output_folder "${kv_dump_root}/heldout_videos" \
      --num_samples 1 \
      --num_output_frames "${num_output_frames}" \
      --local_attn_size "${local_attn_size}" \
      --use_ema \
      --save_with_index \
      --quant_type "none"

  echo "Heldout KV dumps saved to: ${heldout_dump_dir}"
else
  echo "WARNING: No heldout prompts (total=${total_prompts}, train_n=${train_n})."
  echo "         Set TRAIN_N < ${total_prompts} to create a held-out split."
fi

# ── Print usage hint ─────────────────────────────────────────────────
echo ""
echo "=== Done collecting KV dumps ==="
echo ""
if [ "${kv_dump_format}" = "layer_shards" ]; then
  echo "Run TRQ diagnostics:"
  echo "  DUMPS_GLOB='${kv_dump_root}/heldout/*_layer*.pt' \\"
  echo "    bash scripts/analysis/run_trq_diagnostics.sh"
else
  echo "Run predictor fitting:"
  echo "  cd ${hwq_root}"
  echo "  python scripts/self_forcing/analyze_hrq_predictor.py \\"
  echo "    --train_dumps '${train_dump_dir}/*.pt' \\"
  echo "    --heldout_dumps '${kv_dump_root}/heldout/*.pt' \\"
  echo "    --output_dir assets/trq_predictors \\"
  echo "    --report_path results/selfforcing/vbench_eval_hrq_predictor/predictor_diagnostics.txt"
fi
