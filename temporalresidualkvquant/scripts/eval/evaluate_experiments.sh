#!/bin/bash
# VBench evaluation of 4 Self-Forcing experiment lines
# Usage: bash scripts/eval/evaluate_experiments.sh

set -euo pipefail

# Activate vbench conda environment
eval "$(conda shell.bash hook)"
conda activate vbench

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
RESULTS_DIR="$PROJECT_DIR/results/selfforcing"
EVAL_OUTPUT_DIR="$RESULTS_DIR/vbench_eval"
PROMPT_FILE="$RESULTS_DIR/vbench_prompts.txt"
VBENCH_DIR="/mnt/workspace/caipeiliang/code/moweile/videoquant/forcing/vbench"

# 4 experiment lines
EXPERIMENTS=(
    "bf16"
    "triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1"
    "rhwq_seed_0_hi_4_triton-nstages-kmeans-int4_lo_triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1"
    "rhwq_seed_0_hi_4_naive-int4_lo_naive-int2_64/kc_256_vc_256_nstages_1"
)

LABELS=(
    "bf16_baseline"
    "qvg_int2_baseline"
    "rhwq_4h_prq"
    "rhwq_4h_naive"
)

# 8 evaluation dimensions
DIMENSIONS=(
    "subject_consistency"
    "background_consistency"
    "motion_smoothness"
    "dynamic_degree"
    "aesthetic_quality"
    "imaging_quality"
    "overall_consistency"
    "clip_score"
)

mkdir -p "$EVAL_OUTPUT_DIR"

echo "==== VBench Evaluation: 4 Experiment Lines ===="
echo "Results will be saved to: $EVAL_OUTPUT_DIR"
echo "Total: ${#EXPERIMENTS[@]} experiments x ${#DIMENSIONS[@]} dimensions = $((${#EXPERIMENTS[@]} * ${#DIMENSIONS[@]})) eval runs"
echo ""

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    label="${LABELS[$i]}"
    exp_path="$RESULTS_DIR/$exp"

    if [ ! -d "$exp_path" ]; then
        echo "WARNING: $exp_path not found, skipping $label"
        continue
    fi

    # Count videos
    n_videos=$(find "$exp_path" -maxdepth 1 -type f -name "*.mp4" | wc -l)
    if [ "$n_videos" -eq 0 ]; then
        echo "WARNING: No mp4 videos found in $exp_path, skipping"
        continue
    fi

    echo "============================================"
    echo "Evaluating: $label ($n_videos videos)"
    echo "Path: $exp_path"
    echo "============================================"

    for dim in "${DIMENSIONS[@]}"; do
        echo "  [$label] -> $dim"

        python "$VBENCH_DIR/vbench2_beta_long/eval_long.py" \
            --videos_path "$exp_path" \
            --name "$label" \
            --dimension "$dim" \
            --mode long_custom_input \
            --prompt_file "$PROMPT_FILE" \
            --output_path "$EVAL_OUTPUT_DIR"
    done
    echo ""
done

echo "==== All evaluations complete ===="
echo "Results in: $EVAL_OUTPUT_DIR"
