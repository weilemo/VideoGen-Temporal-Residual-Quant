#!/usr/bin/env python3

import argparse
from pathlib import Path

from trq.analysis.protection_gate import analyze_protection_candidates, write_protection_gate


def parse_mapping(values, option):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects CONFIG=PATH, got {value!r}")
        config, path = value.split("=", 1)
        result[config] = Path(path).expanduser()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the E4 quality/jump/bytes gate")
    parser.add_argument("--quality-gate", required=True)
    parser.add_argument("--baseline-latent", required=True)
    parser.add_argument("--candidate-latent", action="append", required=True)
    parser.add_argument("--runtime-root", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows, summary = analyze_protection_candidates(
        quality_gate_path=args.quality_gate,
        baseline_latent_csv=args.baseline_latent,
        candidate_latent_csvs=parse_mapping(args.candidate_latent, "--candidate-latent"),
        runtime_roots=parse_mapping(args.runtime_root, "--runtime-root"),
    )
    write_protection_gate(args.output_dir, rows, summary)
    for row in rows:
        print(f"{row['config']}: {row['status']}")


if __name__ == "__main__":
    main()
