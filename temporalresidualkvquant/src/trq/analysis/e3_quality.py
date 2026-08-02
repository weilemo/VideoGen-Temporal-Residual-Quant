"""Direct BF16-referenced latent analysis for Conditional Innovation E3."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from trq.analysis.paired_rollout import (
    load_rollouts,
    paired_distances,
    theil_sen_slope,
    validate_paired_metadata,
)


def analyze_e3_direct(
    bf16: str | Path,
    cross: str | Path,
    hybrid: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups = {
        "bf16": load_rollouts(bf16),
        "cross": load_rollouts(cross),
        "hybrid": load_rollouts(hybrid),
    }
    key_sets = {name: set(records) for name, records in groups.items()}
    if len({frozenset(keys) for keys in key_sets.values()}) != 1:
        detail = {name: [list(key) for key in sorted(keys)] for name, keys in key_sets.items()}
        raise ValueError(f"E3 rollout keys do not match: {detail}")

    position_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for key in sorted(key_sets["bf16"]):
        reference = groups["bf16"][key]
        for method in ("cross", "hybrid"):
            candidate = groups[method][key]
            validate_paired_metadata(key, reference, reference, candidate)
            if reference.latents.shape != candidate.latents.shape:
                raise ValueError(
                    f"latent shape mismatch for {method} {key}: "
                    f"{tuple(reference.latents.shape)} != {tuple(candidate.latents.shape)}"
                )
            rel_l2, cosine = paired_distances(reference.latents, candidate.latents)
            for position, (rel_value, cosine_value) in enumerate(zip(rel_l2, cosine)):
                position_rows.append(
                    {
                        "method": method,
                        "prompt_index": key[0],
                        "seed": key[1],
                        "sample_index": key[2],
                        "position": position,
                        "relative_l2": float(rel_value),
                        "cosine_distance": float(cosine_value),
                    }
                )
            window = max(1, int(np.ceil(rel_l2.size * 0.2)))
            pair_rows.append(
                {
                    "method": method,
                    "prompt_index": key[0],
                    "seed": key[1],
                    "sample_index": key[2],
                    "positions": int(rel_l2.size),
                    "mean_relative_l2": float(np.mean(rel_l2)),
                    "early_median": float(np.median(rel_l2[:window])),
                    "late_median": float(np.median(rel_l2[-window:])),
                    "final": float(rel_l2[-1]),
                    "p95": float(np.quantile(rel_l2, 0.95)),
                    "max": float(np.max(rel_l2)),
                    "theil_sen_slope": theil_sen_slope(rel_l2),
                }
            )

    metric_names = (
        "mean_relative_l2",
        "early_median",
        "late_median",
        "final",
        "p95",
        "max",
        "theil_sen_slope",
    )
    methods = {}
    for method in ("cross", "hybrid"):
        rows = [row for row in pair_rows if row["method"] == method]
        methods[method] = {
            "pairs": len(rows),
            **{
                f"median_{name}": float(np.median([float(row[name]) for row in rows]))
                for name in metric_names
            },
        }

    paired_ratios = []
    cross_rows = {
        (row["prompt_index"], row["seed"], row["sample_index"]): row
        for row in pair_rows
        if row["method"] == "cross"
    }
    for row in pair_rows:
        if row["method"] != "hybrid":
            continue
        key = (row["prompt_index"], row["seed"], row["sample_index"])
        paired_ratios.append(
            float(row["mean_relative_l2"])
            / max(float(cross_rows[key]["mean_relative_l2"]), 1e-12)
        )

    summary = {
        "schema_version": 1,
        "status": "MEASURED_REQUIRES_MANUAL_REVIEW",
        "scientific_scope": (
            "Direct same-seed BF16 trajectory similarity; no BF16-repeat noise subtraction "
            "and no perceptual-quality claim"
        ),
        "keys": len(key_sets["bf16"]),
        "methods": methods,
        "hybrid_over_cross_mean_relative_l2": {
            "pair_median": float(np.median(paired_ratios)),
            "pair_values": paired_ratios,
        },
    }
    return position_rows, pair_rows, summary


def write_e3_direct(
    output_dir: str | Path,
    position_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "position_rows.csv", position_rows)
    _write_csv(output / "pair_rows.csv", pair_rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty E3 table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
