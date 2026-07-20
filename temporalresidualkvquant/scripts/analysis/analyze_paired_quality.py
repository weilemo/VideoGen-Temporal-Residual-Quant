#!/usr/bin/env python3

import argparse
from pathlib import Path

from trq.analysis.paired_quality import (
    VBenchRun,
    analyze_paired_quality,
    write_quality_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prompt-paired VBench and latent metrics")
    parser.add_argument(
        "--vbench-run",
        action="append",
        required=True,
        metavar="CONFIG,SEED,LABEL,DIR",
    )
    parser.add_argument("--latent", action="append", default=[], metavar="CONFIG=CSV")
    parser.add_argument("--failure-tags")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    runs = []
    for value in args.vbench_run:
        parts = value.split(",", 3)
        if len(parts) != 4:
            parser.error(f"invalid --vbench-run {value!r}; expected CONFIG,SEED,LABEL,DIR")
        config, seed, label, input_dir = parts
        runs.append(VBenchRun(config, int(seed), label, Path(input_dir).expanduser()))

    latent_csvs = {}
    for value in args.latent:
        if "=" not in value:
            parser.error(f"invalid --latent {value!r}; expected CONFIG=CSV")
        config, path = value.split("=", 1)
        latent_csvs[config] = Path(path).expanduser()

    rows, correlations, summary = analyze_paired_quality(
        runs,
        latent_csvs=latent_csvs,
        failure_tags_path=args.failure_tags,
    )
    write_quality_analysis(args.output_dir, rows, correlations, summary)
    print(f"Paired quality analysis complete: {args.output_dir}")
    for config, gate in summary["configs"].items():
        print(f"{config}: {gate['status']}")


if __name__ == "__main__":
    main()
