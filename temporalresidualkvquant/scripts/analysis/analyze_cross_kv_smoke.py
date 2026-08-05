#!/usr/bin/env python3
"""Analyze the frozen BF16/Temporal/Cross Smoke4 experiment."""

from __future__ import annotations

import argparse
import json

from trq.analysis.cross_kv_smoke import analyze_cross_kv_smoke, write_cross_kv_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", required=True)
    parser.add_argument("--temporal", required=True)
    parser.add_argument("--cross", required=True)
    parser.add_argument("--temporal-traces", required=True)
    parser.add_argument("--cross-traces", required=True)
    parser.add_argument("--runtime-root", action="append", default=[], metavar="METHOD=DIR")
    parser.add_argument("--quality", action="append", default=[], metavar="METHOD,SEED,JSON")
    parser.add_argument("--failure-tags")
    parser.add_argument("--cross-params")
    parser.add_argument("--expected-pairs", type=int, default=8)
    parser.add_argument("--bytes-tolerance", type=float, default=0.005)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    runtime_roots = {}
    for value in args.runtime_root:
        if "=" not in value:
            parser.error(f"invalid --runtime-root {value!r}; expected METHOD=DIR")
        method, path = value.split("=", 1)
        runtime_roots[method] = path

    quality_runs = []
    for value in args.quality:
        parts = value.split(",", 2)
        if len(parts) != 3:
            parser.error(f"invalid --quality {value!r}; expected METHOD,SEED,JSON")
        quality_runs.append((parts[0], int(parts[1]), parts[2]))

    tables, summary = analyze_cross_kv_smoke(
        bf16_root=args.bf16,
        temporal_root=args.temporal,
        cross_root=args.cross,
        temporal_trace_root=args.temporal_traces,
        cross_trace_root=args.cross_traces,
        runtime_roots=runtime_roots,
        quality_runs=quality_runs,
        failure_tags=args.failure_tags,
        cross_params=args.cross_params,
        expected_pairs=args.expected_pairs,
        bytes_tolerance=args.bytes_tolerance,
    )
    write_cross_kv_smoke(args.output_dir, tables, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
