#!/usr/bin/env python3
"""Run same-seed BF16-reference metrics for one forcing baseline."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import statistics
from pathlib import Path


DEFAULT_VARIANTS = (
    "trq_int4=trq_int4",
    "trq_int2=trq_int2",
    "naive_int4=naive_int4",
    "naive_int2=naive_int2",
)


def parse_variant(value: str) -> tuple[str, str]:
    name, separator, directory = value.partition("=")
    if not separator or not name or not directory:
        raise argparse.ArgumentTypeError("variant must have the form NAME=DIRECTORY")
    return name, directory


def load_evaluator(repo_root: Path):
    path = repo_root / "temporalresidualkvquant/scripts/eval/eval_ref_metrics.py"
    spec = importlib.util.spec_from_file_location("trq_eval_ref_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_summary(summary: dict, expected_videos: int | None) -> None:
    if expected_videos is not None and summary["num_videos"] != expected_videos:
        raise ValueError(
            f"expected {expected_videos} paired videos, got {summary['num_videos']}"
        )
    for key in ("mean_ssim", "mean_lpips"):
        value = summary.get(key)
        if value is None or not math.isfinite(value):
            raise ValueError(f"{key} is missing or non-finite: {value}")
    psnr = summary.get("mean_psnr")
    if psnr is None or math.isnan(psnr):
        raise ValueError(f"mean_psnr is missing or NaN: {psnr}")


def paired_bootstrap(
    trq_summary: dict,
    naive_summary: dict,
    *,
    resamples: int,
    seed: int,
) -> dict:
    trq_rows = {
        (row["idx"], row["sample_idx"]): row for row in trq_summary["per_video"]
    }
    naive_rows = {
        (row["idx"], row["sample_idx"]): row for row in naive_summary["per_video"]
    }
    if set(trq_rows) != set(naive_rows):
        raise ValueError("TRQ and naive per-video index sets differ")
    keys = sorted(trq_rows)
    if not keys:
        raise ValueError("cannot bootstrap an empty paired result set")
    rng = random.Random(seed)
    metrics = {
        "psnr": lambda trq, naive: trq - naive,
        "ssim": lambda trq, naive: trq - naive,
        "lpips": lambda trq, naive: naive - trq,
    }
    output = {}
    for metric, advantage in metrics.items():
        values = [
            advantage(trq_rows[key][metric], naive_rows[key][metric]) for key in keys
        ]
        samples = []
        for _ in range(resamples):
            draw = [values[rng.randrange(len(values))] for _ in values]
            samples.append(statistics.fmean(draw))
        samples.sort()
        low = samples[int(0.025 * (len(samples) - 1))]
        high = samples[int(0.975 * (len(samples) - 1))]
        output[metric] = {
            "trq_advantage_mean": statistics.fmean(values),
            "bootstrap_95_ci": [low, high],
            "paired_win_rate": sum(value > 0 for value in values) / len(values),
            "positive_means": "TRQ is closer to BF16 than naive",
        }
    return {
        "paired_video_count": len(keys),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "metrics": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        required=True,
        choices=(
            "selfforcing",
            "rollingforcing",
            "causal_forcing",
            "longcat",
            "hy_worldplay",
        ),
    )
    parser.add_argument("--root", required=True, help="Directory containing BF16 and quantized variant folders")
    parser.add_argument("--reference", default="bf16", help="BF16 reference directory relative to --root")
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_variant,
        help="Repeated NAME=DIRECTORY mapping; defaults to the four experiment variants",
    )
    parser.add_argument("--output-dir", default="", help="Default: ROOT/paired_metrics")
    parser.add_argument("--expected-videos", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Skip shared prefix/conditioning frames before comparison",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    args = parser.parse_args()

    if args.bootstrap_resamples <= 0:
        parser.error("--bootstrap-resamples must be positive")

    repo_root = Path(__file__).resolve().parents[2]
    evaluator = load_evaluator(repo_root)
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root / "paired_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = args.variant or [parse_variant(value) for value in DEFAULT_VARIANTS]

    rows = []
    summaries = {}
    for variant, relative_dir in variants:
        summary = evaluator.evaluate_directories(
            root / args.reference,
            root / relative_dir,
            max_frames=args.max_frames or None,
            start_frame=args.start_frame,
            device=args.device,
            match_by_index=True,
            strict_shape=True,
            require_lpips=True,
        )
        validate_summary(summary, args.expected_videos or None)
        summary.update(
            {
                "baseline": args.baseline,
                "variant": variant,
                "reference_semantics": "same-prompt same-seed BF16 output, not ground truth",
            }
        )
        (output_dir / f"{variant}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summaries[variant] = summary
        rows.append({key: summary[key] for key in (
            "baseline", "variant", "num_videos", "mean_psnr", "mean_ssim", "mean_lpips"
        )})

    comparisons = {}
    for bits in ("int4", "int2"):
        trq_name = f"trq_{bits}"
        naive_name = f"naive_{bits}"
        if trq_name in summaries and naive_name in summaries:
            comparisons[bits] = paired_bootstrap(
                summaries[trq_name],
                summaries[naive_name],
                resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed,
            )

    aggregate = {
        "baseline": args.baseline,
        "root": str(root),
        "reference": args.reference,
        "metric_scope": "quantization trajectory fidelity against same-seed BF16",
        "start_frame": args.start_frame,
        "higher_is_better": ["psnr", "ssim"],
        "lower_is_better": ["lpips"],
        "variants": rows,
        "paired_trq_vs_naive": comparisons,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
