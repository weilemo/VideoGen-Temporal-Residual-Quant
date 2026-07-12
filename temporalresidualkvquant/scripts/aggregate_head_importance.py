#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

try:
    from trq.head_importance import build_topk_policy_from_focused_forcing, write_topk_policy
except ModuleNotFoundError:
    module_path = Path(__file__).resolve().parents[1] / "src" / "trq" / "head_importance.py"
    spec = importlib.util.spec_from_file_location("hwq_head_importance_standalone", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    build_topk_policy_from_focused_forcing = module.build_topk_policy_from_focused_forcing
    write_topk_policy = module.write_topk_policy


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate focused-forcing head-ablation DMD losses into a per-layer top-k policy."
    )
    parser.add_argument("--input", required=True, help="A JSON file or a directory containing focused-forcing JSON outputs")
    parser.add_argument("--output", required=True, help="Path to write the top-k policy JSON")
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=30)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument(
        "--score_direction",
        choices=["higher", "lower"],
        default="higher",
        help="Use higher DMD loss as more important by default",
    )
    parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="Write policies for layers with complete data even if some layers are missing",
    )
    args = parser.parse_args()

    try:
        payload = build_topk_policy_from_focused_forcing(
            args.input,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            top_k=args.top_k,
            score_direction=args.score_direction,
            allow_incomplete=args.allow_incomplete,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output_path = write_topk_policy(payload, args.output)
    print(f"Wrote top-k policy for {len(payload['top_heads_by_layer'])} layers to {output_path}")


if __name__ == "__main__":
    main()
