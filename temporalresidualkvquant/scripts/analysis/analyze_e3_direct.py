#!/usr/bin/env python3
"""Analyze direct BF16/Cross/Hybrid E3 latent trajectories."""

from __future__ import annotations

import argparse
import json

from trq.analysis.e3_quality import analyze_e3_direct, write_e3_direct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", required=True)
    parser.add_argument("--cross", required=True)
    parser.add_argument("--hybrid", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    position_rows, pair_rows, summary = analyze_e3_direct(
        args.bf16, args.cross, args.hybrid
    )
    write_e3_direct(args.output_dir, position_rows, pair_rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
