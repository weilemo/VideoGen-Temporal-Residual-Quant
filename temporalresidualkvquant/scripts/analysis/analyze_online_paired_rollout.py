#!/usr/bin/env python3

import argparse

from trq.analysis.paired_rollout import analyze_paired_rollouts, write_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze BF16-repeat and TRQ paired rollout latents")
    parser.add_argument("--bf16-a", required=True)
    parser.add_argument("--bf16-b", required=True)
    parser.add_argument("--trq", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    position_rows, pair_rows, summary = analyze_paired_rollouts(
        args.bf16_a,
        args.bf16_b,
        args.trq,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    write_analysis(args.output_dir, position_rows, pair_rows, summary)
    print(f"Online paired rollout analysis complete: {args.output_dir}")
    print(f"Latent drift gate: {'PASS' if summary['passes_latent_drift_gate'] else 'FAIL'}")


if __name__ == "__main__":
    main()
