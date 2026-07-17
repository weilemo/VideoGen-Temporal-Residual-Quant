#!/usr/bin/env python3
"""Aggregate structured online rollout runtime metrics across experiment folders."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_rows(roots: list[str]) -> list[dict]:
    rows = []
    seen = set()
    for root_text in roots:
        root = Path(root_text).expanduser()
        if not root.exists():
            continue
        for path in sorted(root.glob("**/rollout_metrics/runtime/*.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata", {})
            pipeline = payload.get("pipeline", {})
            stages = pipeline.get("stages_ms", {})
            operations = pipeline.get("operations", {})
            cache = pipeline.get("kv_cache", {}).get("total", {})
            memory = pipeline.get("memory", {})
            run_dir = path.parents[2]
            rows.append(
                {
                    "run": str(run_dir),
                    "method": run_dir.name,
                    "prompt_index": metadata.get("prompt_index"),
                    "seed": metadata.get("seed"),
                    "frames": metadata.get("num_output_frames"),
                    "quant_type": metadata.get("quant_type"),
                    "k_bits": metadata.get("trq_k_bits"),
                    "v_bits": metadata.get("trq_v_bits"),
                    "wall_time_e2e_ms": payload.get("wall_time_e2e_ms"),
                    "diffusion_ms": stages.get("diffusion"),
                    "vae_decode_ms": stages.get("vae_decode"),
                    "quantize_ms": operations.get("quantize_ms"),
                    "dequantize_ms": operations.get("dequantize_ms"),
                    "kv_physical_bytes": cache.get("physical_bytes"),
                    "kv_bf16_equivalent_bytes": cache.get("bf16_equivalent_bytes"),
                    "effective_bits_per_value": cache.get("effective_bits_per_value"),
                    "compression_ratio": cache.get("compression_ratio"),
                    "saving_fraction": cache.get("saving_fraction"),
                    "max_allocated_bytes": memory.get("max_allocated_bytes"),
                    "max_reserved_bytes": memory.get("max_reserved_bytes"),
                    "source": str(path),
                }
            )
    return rows


def median(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.median(clean) if clean else None


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.roots)
    if not rows:
        raise SystemExit("No rollout runtime JSON files found")

    columns = list(rows[0])
    with (output_dir / "runtime_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["method"],
            row["frames"],
            row["quant_type"],
            row["k_bits"],
            row["v_bits"],
        )
        grouped[key].append(row)

    numeric = [
        "wall_time_e2e_ms",
        "diffusion_ms",
        "vae_decode_ms",
        "quantize_ms",
        "dequantize_ms",
        "kv_physical_bytes",
        "kv_bf16_equivalent_bytes",
        "effective_bits_per_value",
        "compression_ratio",
        "saving_fraction",
        "max_allocated_bytes",
        "max_reserved_bytes",
    ]
    summary = []
    for key, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        record = {
            "method": key[0],
            "frames": key[1],
            "quant_type": key[2],
            "k_bits": key[3],
            "v_bits": key[4],
            "samples": len(items),
        }
        record.update({f"median_{name}": median([item[name] for item in items]) for name in numeric})
        summary.append(record)

    (output_dir / "runtime_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "runtime_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Aggregated {len(rows)} runs into {len(summary)} groups: {output_dir}")


if __name__ == "__main__":
    main()
