#!/usr/bin/env python3
"""Aggregate VBench evaluation results across 4 experiment lines and produce a comparison table."""

import json
import os
import sys
from glob import glob
from collections import OrderedDict

TASK_INFO = [
    "subject consistency",
    "background consistency",
    "motion smoothness",
    "dynamic degree",
    "aesthetic quality",
    "imaging quality",
    "overall consistency",
    "clip score",
]

NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
    "clip score": {"Min": 0.0, "Max": 0.3557},
}

DIM_WEIGHT = {
    "subject consistency": 1,
    "background consistency": 1,
    "motion smoothness": 1,
    "dynamic degree": 0.5,
    "aesthetic quality": 1,
    "imaging quality": 1,
    "overall consistency": 1,
    "clip score": 1,
}

TEMPORAL_QUALITY_WEIGHT = 2
FRAME_WISE_QUALITY_WEIGHT = 2
TEXT_ALIGNMENT_WEIGHT = 1

LABEL_ORDER = [
    "bf16_baseline",
    "qvg_int2_baseline",
    "rhwq_4h_prq",
    "rhwq_4h_naive",
    "rhwq_4h_packed_int4_int2",
    "rhwq_4h_packed_int8_int4",
    "topk_hwq_packed_int4_int2",
    # 32-prompt MovieGenVideoBench comparison
    "bf16_mb32",
    "topk_hwq_int8_int4_mb32",
    "rhwq_random_int8_int4_mb32",
    "topk_k2_int8_int4_mb32",
    "topk_k6_int8_int4_mb32",
    "topk_k8_int8_int4_mb32",
    # int4+int2 32-prompt k-sweep
    "rhwq_random_int4_int2_mb32",
    "topk_k4_int4_int2_mb32",
    "topk_k2_int4_int2_mb32",
    "topk_k6_int4_int2_mb32",
    "topk_k8_int4_int2_mb32",
]

LABEL_NAMES = {
    "bf16_baseline": "BF16 Baseline",
    "qvg_int2_baseline": "QVG INT2 (PRQ)",
    "rhwq_4h_prq": "R-HWQ-4h (PRQ)",
    "rhwq_4h_naive": "R-HWQ-4h (Naive)",
    "rhwq_4h_packed_int4_int2": "R-HWQ-4h Packed (int4+int2)",
    "rhwq_4h_packed_int8_int4": "R-HWQ-4h Packed (int8+int4)",
    "topk_hwq_packed_int4_int2": "Top-K HWQ Packed (int4+int2)",
    # int8+int4
    "bf16_mb32": "BF16 Baseline",
    "topk_hwq_int8_int4_mb32": "TK4 int8+int4",
    "rhwq_random_int8_int4_mb32": "Rand4 int8+int4",
    "topk_k2_int8_int4_mb32": "TK2 int8+int4",
    "topk_k6_int8_int4_mb32": "TK6 int8+int4",
    "topk_k8_int8_int4_mb32": "TK8 int8+int4",
    # int4+int2
    "rhwq_random_int4_int2_mb32": "Rand4 int4+int2",
    "topk_k4_int4_int2_mb32": "TK4 int4+int2",
    "topk_k2_int4_int2_mb32": "TK2 int4+int2",
    "topk_k6_int4_int2_mb32": "TK6 int4+int2",
    "topk_k8_int4_int2_mb32": "TK8 int4+int2",
}


def snake_to_space(s):
    return s.replace("_", " ").strip().lower()


def extract_scalar(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        if len(v) == 0:
            return None
        if isinstance(v[0], (int, float)):
            return float(v[0])
    return None


def normalize(name, val):
    mn = NORMALIZE_DIC[name]["Min"]
    mx = NORMALIZE_DIC[name]["Max"]
    if mx <= mn:
        return 0.0
    return max(0.0, min(1.0, (val - mn) / (mx - mn)))


def load_scores_for_label(input_dir, label):
    """Load all dimension scores for a given experiment label."""
    files = sorted(glob(os.path.join(input_dir, f"{label}_*_eval_results.json")))
    scores = {}
    for fp in files:
        with open(fp, "r") as f:
            obj = json.load(f)
        for k, v in obj.items():
            dim_name = snake_to_space(k)
            if dim_name in TASK_INFO:
                val = extract_scalar(v)
                if val is not None:
                    scores[dim_name] = val
    # Fill missing with None
    for d in TASK_INFO:
        scores.setdefault(d, None)
    return scores


def compute_category_scores(scores_raw):
    """Compute normalized category scores and final score."""
    norm_weighted = {}
    for d in TASK_INFO:
        if scores_raw[d] is not None:
            norm_weighted[d] = normalize(d, scores_raw[d]) * DIM_WEIGHT[d]
        else:
            norm_weighted[d] = None

    temporal_dims = [
        "subject consistency",
        "background consistency",
        "motion smoothness",
        "dynamic degree",
    ]
    frame_dims = [
        "aesthetic quality",
        "imaging quality",
    ]
    text_dims = [
        "overall consistency",
        "clip score",
    ]

    def _safe_mean(dims):
        vals = [norm_weighted[d] for d in dims if norm_weighted[d] is not None]
        w = [DIM_WEIGHT[d] for d in dims if norm_weighted[d] is not None]
        if not vals:
            return None
        return sum(vals) / sum(w)

    temporal_quality = _safe_mean(temporal_dims)
    frame_wise_quality = _safe_mean(frame_dims)
    text_alignment = _safe_mean(text_dims)

    final_score = None
    if all(v is not None for v in [temporal_quality, frame_wise_quality, text_alignment]):
        final_score = (
            TEMPORAL_QUALITY_WEIGHT * temporal_quality
            + FRAME_WISE_QUALITY_WEIGHT * frame_wise_quality
            + TEXT_ALIGNMENT_WEIGHT * text_alignment
        ) / (TEMPORAL_QUALITY_WEIGHT + FRAME_WISE_QUALITY_WEIGHT + TEXT_ALIGNMENT_WEIGHT)

    # Also compute 6-dims mean (raw)
    six_dims = temporal_dims + frame_dims
    six_dims_raw_vals = [scores_raw[d] for d in six_dims if scores_raw[d] is not None]
    six_dims_mean_raw = sum(six_dims_raw_vals) / len(six_dims_raw_vals) if six_dims_raw_vals else None

    return {
        "temporal_quality": temporal_quality,
        "frame_wise_quality": frame_wise_quality,
        "text_alignment": text_alignment,
        "final_score": final_score,
        "six_dims_mean_raw": six_dims_mean_raw,
        "norm_weighted": norm_weighted,
    }


def print_comparison_table(all_results):
    """Print a formatted comparison table."""
    # Header
    header = f"{'Dimension':<28}"
    for label in LABEL_ORDER:
        header += f"{LABEL_NAMES[label]:>22}"
    print(header)
    print("-" * len(header))

    # Raw scores per dimension
    for dim in TASK_INFO:
        row = f"  {dim:<26}"
        for label in LABEL_ORDER:
            if label in all_results and dim in all_results[label].get("raw_scores", {}):
                val = all_results[label]["raw_scores"][dim]
                if val is not None:
                    row += f"{val:>22.4f}"
                else:
                    row += f"{'N/A':>22}"
            else:
                row += f"{'N/A':>22}"
        print(row)

    print("-" * len(header))

    # Normalized scores
    for dim in TASK_INFO:
        row = f"  {dim:<26} (norm)"
        for label in LABEL_ORDER:
            if label in all_results:
                val = all_results[label]["raw_scores"].get(dim)
                if val is not None:
                    n = normalize(dim, val)
                    row += f"{n:>22.4f}"
                else:
                    row += f"{'N/A':>22}"
            else:
                row += f"{'N/A':>22}"
        print(row)

    print("=" * len(header))

    # Category scores
    categories = [
        ("Temporal Quality", "temporal_quality"),
        ("Frame-wise Quality", "frame_wise_quality"),
        ("Text Alignment", "text_alignment"),
        ("6-Dim Mean (raw)", "six_dims_mean_raw"),
        ("FINAL SCORE", "final_score"),
    ]
    for cat_name, cat_key in categories:
        row = f"{cat_name:<28}"
        for label in LABEL_ORDER:
            if label in all_results:
                val = all_results[label].get(cat_key)
                if val is not None:
                    row += f"{val:>22.4f}"
                else:
                    row += f"{'N/A':>22}"
            else:
                row += f"{'N/A':>22}"
        print(row)

    print("=" * len(header))

    # Relative degradation vs BF16 (separate into 2-prompt and 32-prompt groups)
    print("\n--- Relative to BF16 Baseline ---")
    for bf16_label in ["bf16_baseline", "bf16_mb32"]:
        bf16_final = all_results.get(bf16_label, {}).get("final_score")
        if not bf16_final:
            continue
        group_name = "32-prompt" if "mb32" in bf16_label else "2-prompt"
        print(f"\n  [{group_name} group, vs {LABEL_NAMES[bf16_label]}: {bf16_final:.4f}]")
        for label in LABEL_ORDER:
            if label == bf16_label or label not in all_results:
                continue
            final = all_results[label].get("final_score")
            if final is not None:
                degradation = (bf16_final - final) / bf16_final * 100
                print(f"    {LABEL_NAMES[label]:<22}: {final:.4f} (↓{degradation:.2f}%)")


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "temporalresidualkvquant/results/selfforcing/vbench_eval"

    if not os.path.isdir(input_dir):
        print(f"Error: directory not found: {input_dir}")
        sys.exit(1)

    all_results = OrderedDict()
    for label in LABEL_ORDER:
        scores_raw = load_scores_for_label(input_dir, label)
        if all(v is None for v in scores_raw.values()):
            print(f"Warning: No scores found for {label}, skipping")
            continue

        cat_scores = compute_category_scores(scores_raw)
        all_results[label] = {
            "raw_scores": scores_raw,
            **cat_scores,
        }

    if not all_results:
        print("Error: No experiment results found.")
        sys.exit(1)

    print_comparison_table(all_results)

    # Save JSON summary
    summary = OrderedDict()
    for label in LABEL_ORDER:
        if label in all_results:
            r = all_results[label]
            summary[label] = OrderedDict([
                ("name", LABEL_NAMES[label]),
                ("raw_scores", r["raw_scores"]),
                ("norm_weighted", r["norm_weighted"]),
                ("temporal_quality", r["temporal_quality"]),
                ("frame_wise_quality", r["frame_wise_quality"]),
                ("text_alignment", r["text_alignment"]),
                ("final_score", r["final_score"]),
            ])

    summary_path = os.path.join(input_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
