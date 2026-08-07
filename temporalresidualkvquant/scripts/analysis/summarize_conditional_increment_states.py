#!/usr/bin/env python3
"""Compare raw and decoder-reconstructed conditional-increment results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from trq.analysis.conditional_innovation import bootstrap_median_ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--reconstructed-k", required=True)
    parser.add_argument("--full-decoder-state", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-retention", type=float, default=0.80)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.minimum_retention <= 1:
        raise ValueError("minimum retention must be in [0, 1]")
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")

    roots = {
        "raw": Path(args.raw).expanduser().resolve(),
        "reconstructed_k": Path(args.reconstructed_k).expanduser().resolve(),
        "full_decoder_state": Path(args.full_decoder_state).expanduser().resolve(),
    }
    summaries = {name: _read_json(root / "summary.json") for name, root in roots.items()}
    prompt_rows = {name: _read_prompt_rows(root / "prompt_rows.csv") for name, root in roots.items()}
    prompt_ids = set(prompt_rows["raw"])
    for name, rows in prompt_rows.items():
        if set(rows) != prompt_ids:
            raise ValueError(f"prompt identities differ for state source {name}")

    retention = {}
    for offset, name in enumerate(("reconstructed_k", "full_decoder_state"), start=1):
        ratios = []
        for prompt_id in sorted(prompt_ids):
            raw_effect = prompt_rows["raw"][prompt_id]
            if raw_effect <= 0:
                raise ValueError(
                    f"raw partial R2 must be positive for retention at prompt {prompt_id}"
                )
            ratios.append(prompt_rows[name][prompt_id] / raw_effect)
        retention[name] = bootstrap_median_ci(
            ratios,
            resamples=args.bootstrap_resamples,
            seed=args.seed + offset,
        )

    full_pass = summaries["full_decoder_state"]["status"] == "PASS_STRUCTURE_GATE"
    reconstructed_k_pass = summaries["reconstructed_k"]["status"] == "PASS_STRUCTURE_GATE"
    retention_pass = (
        float(retention["full_decoder_state"]["median"]) >= args.minimum_retention
    )
    passed = full_pass and reconstructed_k_pass and retention_pass
    result = {
        "schema_version": 1,
        "status": (
            "PASS_RECONSTRUCTED_STATE_GATE"
            if passed
            else "FAIL_RECONSTRUCTED_STATE_GATE"
        ),
        "source_status": {
            name: summary["status"] for name, summary in summaries.items()
        },
        "effect_retention": retention,
        "minimum_full_decoder_retention": args.minimum_retention,
        "scientific_scope": (
            "offline decoder-reconstructed-state conditional predictive gain only; "
            "passing authorizes a larger held-out structure study, not an online codec claim"
        ),
        "next_stage": (
            "freeze the protocol and collect the preregistered 8-calibration/32-validation scale study"
            if passed
            else "stop scale-up and diagnose which reconstructed condition destroys the raw-K gain"
        ),
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "report.md", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if args.require_pass and not passed else 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_prompt_rows(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no prompt rows in {path}")
    return {row["prompt_id"]: float(row["partial_r2_joint"]) for row in rows}


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Reconstructed-State Conditional Increment Gate",
        "",
        f"- Status: **{result['status']}**",
        f"- Minimum full-decoder retention: "
        f"{result['minimum_full_decoder_retention']:.3f}",
        "",
        "## Effect Retention",
        "",
    ]
    for name, values in result["effect_retention"].items():
        lines.append(
            f"- {name}: median {values['median']:.6f}, CI95 "
            f"[{values['ci95_lower']:.6f}, {values['ci95_upper']:.6f}]"
        )
    lines.extend(["", f"> {result['scientific_scope']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
