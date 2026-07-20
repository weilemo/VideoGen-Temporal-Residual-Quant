"""E4 quality, boundary-jump, and physical-byte gate aggregation."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


def analyze_protection_candidates(
    *,
    quality_gate_path: str | Path,
    baseline_latent_csv: str | Path,
    candidate_latent_csvs: dict[str, Path],
    runtime_roots: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quality = json.loads(Path(quality_gate_path).read_text(encoding="utf-8"))
    baseline_jump = median_csv_field(baseline_latent_csv, "boundary_jump")
    if baseline_jump <= 0.0:
        raise ValueError(f"K2V2 baseline boundary jump must be positive, got {baseline_jump}")

    rows = []
    for config, latent_csv in sorted(candidate_latent_csvs.items()):
        candidate_jump = median_csv_field(latent_csv, "boundary_jump")
        saving_fraction = median_runtime_saving(runtime_roots[config]) if config in runtime_roots else None
        jump_reduction = 1.0 - candidate_jump / baseline_jump
        quality_status = quality.get("configs", {}).get(config, {}).get("status")
        if quality_status is None or saving_fraction is None:
            status = "INCOMPLETE"
        elif quality_status != "PASS" or jump_reduction < 0.5 or saving_fraction < 0.6:
            status = "FAIL"
        elif saving_fraction < 0.7:
            status = "QUALITY_CEILING"
        else:
            status = "PASS"
        rows.append({
            "config": config,
            "status": status,
            "quality_status": quality_status,
            "baseline_boundary_jump": baseline_jump,
            "candidate_boundary_jump": candidate_jump,
            "boundary_jump_reduction": jump_reduction,
            "median_saving_fraction": saving_fraction,
            "passes_quality": quality_status == "PASS",
            "passes_jump_reduction": jump_reduction >= 0.5,
            "passes_70_percent_saving": saving_fraction is not None and saving_fraction >= 0.7,
        })
    summary = {
        "schema_version": 1,
        "policy": {
            "minimum_boundary_jump_reduction": 0.5,
            "target_saving_fraction": 0.7,
            "quality_ceiling_saving_range": [0.6, 0.7],
        },
        "candidates": {row["config"]: row for row in rows},
        "passing_configs": [row["config"] for row in rows if row["status"] == "PASS"],
    }
    return rows, summary


def median_csv_field(path: str | Path, field: str) -> float:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        values = [
            float(row[field])
            for row in csv.DictReader(handle)
            if row.get(field) not in (None, "")
        ]
    if not values:
        raise ValueError(f"{path} contains no values for {field}")
    return float(statistics.median(values))


def median_runtime_saving(root: str | Path) -> float | None:
    values = []
    for path in sorted(Path(root).expanduser().glob("**/rollout_metrics/runtime/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = (
            payload.get("pipeline", {})
            .get("kv_cache", {})
            .get("total", {})
            .get("saving_fraction")
        )
        if value is not None:
            values.append(float(value))
    return float(statistics.median(values)) if values else None


def write_protection_gate(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "quality_bytes_pareto.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "protection_gate.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
