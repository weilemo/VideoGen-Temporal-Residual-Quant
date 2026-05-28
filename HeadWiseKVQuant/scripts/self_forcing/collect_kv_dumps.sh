#!/bin/bash
# Collect BF16 KV cache dumps for HRQ predictor analysis.
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
#   TRAIN_N              (default 10) number of training prompts
#   PROMPTS_PATH         (default tmp/moviegenbench_15.txt)
#   KV_DUMP_ROOT         (default kv_dumps)
#   NUM_OUTPUT_FRAMES    (default 180)
#   LOCAL_ATTN_SIZE      (default 180)

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

prompts_path="${PROMPTS_PATH:-${hwq_root}/tmp/moviegenbench_15.txt}"
num_output_frames="${NUM_OUTPUT_FRAMES:-180}"
local_attn_size="${LOCAL_ATTN_SIZE:-180}"
train_n="${TRAIN_N:-10}"
kv_dump_root="${KV_DUMP_ROOT:-${hwq_root}/kv_dumps}"

export PYTHONPATH="${hwq_root}/src:${self_forcing_root}:${PYTHONPATH:-}"
export SELF_FORCING_CKPT_ROOT="${ckpt_root}"

echo "HWQ root:      ${hwq_root}"
echo "Self-Forcing:  ${self_forcing_root}"
echo "Checkpoint:    ${ckpt_path}"
echo "Prompts:       ${prompts_path}"
echo "Train N:       ${train_n}"
echo "KV dump root:  ${kv_dump_root}"
echo ""

# ── Count total prompts ─────────────────────────────────────────────
total_prompts=$(wc -l < "${prompts_path}")
echo "Total prompts: ${total_prompts}"

# Split prompts into train and heldout temp files
train_prompts="${kv_dump_root}/tmp_train_prompts.txt"
heldout_prompts="${kv_dump_root}/tmp_heldout_prompts.txt"
mkdir -p "${kv_dump_root}"
head -n "${train_n}" "${prompts_path}" > "${train_prompts}"
tail -n +"$((train_n + 1))" "${prompts_path}" > "${heldout_prompts}"

heldout_n=$(wc -l < "${heldout_prompts}")
echo "Train prompts: ${train_n}, Heldout prompts: ${heldout_n}"
echo ""

# ── Run train dump ──────────────────────────────────────────────────
train_dump_dir="${kv_dump_root}/train"
echo "=== Collecting TRAIN KV dumps → ${train_dump_dir} ==="
mkdir -p "${train_dump_dir}"

DUMP_KV_LEVEL=1 KV_DUMP_DIR="${train_dump_dir}" \
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

# Rename dumps to include prompt index for easy identification
echo "Train KV dumps saved to: ${train_dump_dir}"

# ── Run heldout dump ────────────────────────────────────────────────
if [ "${heldout_n}" -gt 0 ]; then
  heldout_dump_dir="${kv_dump_root}/heldout"
  echo ""
  echo "=== Collecting HELDOUT KV dumps → ${heldout_dump_dir} ==="
  mkdir -p "${heldout_dump_dir}"

  DUMP_KV_LEVEL=1 KV_DUMP_DIR="${heldout_dump_dir}" \
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
echo "Run predictor analysis:"
echo "  cd ${hwq_root}"
echo "  python scripts/self_forcing/analyze_hrq_predictor.py \\"
echo "    --train_dumps '${train_dump_dir}/*.pt' \\"
echo "    --heldout_dumps '${kv_dump_root}/heldout/*.pt' \\"
echo "    --output_dir assets/hrq_predictors \\"
echo "    --report_path results/selfforcing/vbench_eval_hrq_predictor/predictor_diagnostics.txt"
