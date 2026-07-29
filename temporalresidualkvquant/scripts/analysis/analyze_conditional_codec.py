#!/usr/bin/env python3
"""Run E2 fixed-bit reconstructed-state conditional-innovation analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from trq.analysis.conditional_codec import simulate_conditional_codec
from trq.analysis.conditional_innovation import CrossKVModel
from trq.analysis.kv_dump import iter_kv_dump_layers


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-bit Cross-KV and conditional-innovation codecs after PASS_E1."
    )
    parser.add_argument("--e1-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--unit-size", type=int, default=0)
    parser.add_argument("--key-bits", type=int, default=4, choices=(2, 4, 8))
    parser.add_argument("--value-bits", type=int, default=4, choices=(2, 4, 8))
    parser.add_argument("--anchor-bits", type=int, default=4, choices=(2, 4, 8))
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--reset-spans", default="2,4,8")
    parser.add_argument("--max-validation-dumps", type=int, default=0)
    parser.add_argument("--hybrid-ratio-threshold", type=float, default=1.0)
    parser.add_argument("--shuffled-ratio-threshold", type=float, default=0.98)
    parser.add_argument("--maximum-increase-fraction", type=float, default=1.0)
    parser.add_argument("--require-core-pass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    e1_dir = Path(args.e1_dir).expanduser().resolve()
    e1_summary = _read_json(e1_dir / "summary.json")
    if e1_summary.get("status") != "PASS_E1":
        raise RuntimeError(
            f"E2 requires PASS_E1, got {e1_summary.get('status')!r} from {e1_dir}"
        )
    manifest = _read_json(e1_dir / "manifest.json")
    validation_paths = [Path(path) for path in manifest["validation_dumps"]]
    if args.max_validation_dumps:
        validation_paths = validation_paths[: args.max_validation_dumps]
    parameters = torch.load(
        e1_dir / "conditional_innovation_params.pt",
        map_location="cpu",
        weights_only=False,
    )
    reset_spans = tuple(int(value.strip()) for value in args.reset_spans.split(","))

    output_dir = Path(args.output_dir).expanduser().resolve()
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "git_sha": _git_sha(),
            "e1_dir": str(e1_dir),
            "e1_git_sha": manifest.get("git_sha"),
            "validation_dumps": [str(path) for path in validation_paths],
            "config": vars(args),
            "scientific_scope": (
                "E2 fixed-bit offline reconstructed-state mechanism test; "
                "attention-output, video quality, peak VRAM, and latency are not evaluated"
            ),
        },
    )

    total_records = 0
    for path_index, path in enumerate(validation_paths, start=1):
        for layer, key, value, metadata in iter_kv_dump_layers(path, layers=args.layers):
            layer_params = parameters["layers"].get(layer, parameters["layers"].get(str(layer)))
            if layer_params is None or "innovation_gamma" not in layer_params:
                raise ValueError(f"E1 parameters lack hybrid predictor for layer {layer}")
            shard_path = shard_dir / _shard_name(path, layer)
            if shard_path.exists():
                total_records += 1
                continue
            unit_size = _unit_size(args.unit_size, metadata)
            model = CrossKVModel(
                weight=layer_params["cross_weight"],
                bias=layer_params["cross_bias"],
            )
            print(
                f"[e2] dump {path_index}/{len(validation_paths)} layer {layer}: {path.name}",
                flush=True,
            )
            result = simulate_conditional_codec(
                key,
                value,
                model,
                layer_params["innovation_gamma"],
                unit_size=unit_size,
                key_bits=args.key_bits,
                value_bits=args.value_bits,
                anchor_bits=args.anchor_bits,
                block_size=args.block_size,
                reset_spans=reset_spans,
            )
            payload = {
                "schema_version": SCHEMA_VERSION,
                "dump_path": str(path),
                "layer": layer,
                "unit_size": unit_size,
                **result,
            }
            _write_json_atomic(shard_path, payload)
            total_records += 1
            _write_json(
                output_dir / "progress.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "stage": "codec_shards",
                    "completed_layer_records": total_records,
                    "current_dump": str(path),
                    "current_layer": layer,
                },
            )

    shard_payloads = [_read_json(path) for path in sorted(shard_dir.glob("*.json"))]
    if not shard_payloads:
        raise RuntimeError("E2 produced no layer records")
    summary = _aggregate(shard_payloads, args)
    layer_rows, unit_rows = _flatten_rows(shard_payloads)
    _write_csv(output_dir / "layer_rows.csv", layer_rows)
    _write_csv(output_dir / "unit_rows.csv", unit_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "report.md", summary)
    _write_json(
        output_dir / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "analysis_complete",
            "status": summary["status"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_core_pass and summary["gate2_core"]["status"] != "PASS":
        return 2
    return 0


def _aggregate(shards: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    totals: dict[str, dict[str, float]] = {}
    maximum_increase: dict[str, float] = {}
    all_finite: dict[str, bool] = {}
    for shard in shards:
        for method, row in shard["methods"].items():
            target = totals.setdefault(
                method,
                {
                    "squared_error": 0.0,
                    "target_energy": 0.0,
                    "values": 0.0,
                    "payload_bytes": 0.0,
                    "key_payload_bytes": 0.0,
                    "predictor_bytes": 0.0,
                    "physical_bytes": 0.0,
                },
            )
            for key in target:
                target[key] += float(row[key])
            maximum_increase[method] = max(
                maximum_increase.get(method, 0.0),
                float(row["monotonic_increase_fraction"]),
            )
            all_finite[method] = all_finite.get(method, True) and bool(row["finite"])

    methods: dict[str, dict[str, float | int | bool]] = {}
    for method, row in totals.items():
        methods[method] = {
            "mse": row["squared_error"] / max(row["values"], 1.0),
            "rel_l2": math.sqrt(row["squared_error"] / max(row["target_energy"], 1e-30)),
            "payload_bytes": int(row["payload_bytes"]),
            "key_payload_bytes": int(row["key_payload_bytes"]),
            "predictor_bytes": int(row["predictor_bytes"]),
            "physical_bytes": int(row["physical_bytes"]),
            "maximum_monotonic_increase_fraction": maximum_increase[method],
            "finite": all_finite[method],
        }

    closed_ratio = float(methods["closed_hybrid"]["mse"]) / max(
        float(methods["cross"]["mse"]), 1e-30
    )
    oracle_ratio = float(methods["oracle_hybrid"]["mse"]) / max(
        float(methods["cross"]["mse"]), 1e-30
    )
    shuffled_ratio = float(methods["shuffled_hybrid"]["mse"]) / max(
        float(methods["cross"]["mse"]), 1e-30
    )
    core_pass = (
        closed_ratio <= args.hybrid_ratio_threshold
        and shuffled_ratio >= args.shuffled_ratio_threshold
        and bool(methods["closed_hybrid"]["finite"])
        and float(methods["closed_hybrid"]["maximum_monotonic_increase_fraction"])
        < args.maximum_increase_fraction
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "BLOCKED_E2_SUBGATES" if core_pass else "FAIL_E2_CORE",
        "scientific_scope": (
            "fixed-bit offline reconstructed-state mechanism evidence only; "
            "no attention-output, video-quality, or system claim"
        ),
        "layer_records": len(shards),
        "methods": methods,
        "gate2_core": {
            "status": "PASS" if core_pass else "FAIL",
            "closed_hybrid_over_cross_mse": closed_ratio,
            "oracle_hybrid_over_cross_mse": oracle_ratio,
            "shuffled_hybrid_over_cross_mse": shuffled_ratio,
            "hybrid_ratio_threshold": args.hybrid_ratio_threshold,
            "shuffled_ratio_threshold": args.shuffled_ratio_threshold,
            "maximum_increase_fraction": args.maximum_increase_fraction,
        },
        "attention_output_gate": {
            "status": "NOT_EVALUATED",
            "reason": "raw KV dumps do not contain the matched BF16 query tensors",
        },
        "event_recovery_gate": {
            "status": "NOT_EVALUATED",
            "reason": "the current dumps do not carry reviewed event-boundary labels",
        },
        "saturation_gate": {
            "status": "NOT_EVALUATED",
            "reason": "packed reconstruction is measured, but clip counters are not captured",
        },
        "missing_baselines": {
            "concat_predictor": "NOT_EVALUATED",
            "codec_gamma_calibration": "NOT_EVALUATED; E1 BF16 gamma is reused",
            "event_aware_reset": "NOT_EVALUATED; fixed reset spans are reported",
        },
        "next_stage": (
            "complete attention-output, event-recovery, saturation, and codec-gamma "
            "subgates before E3"
        ),
    }


def _flatten_rows(
    shards: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layer_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    for shard in shards:
        common = {
            "dump_path": shard["dump_path"],
            "layer": shard["layer"],
            "unit_size": shard["unit_size"],
        }
        for method, row in shard["methods"].items():
            layer_rows.append({**common, "method": method, **row})
        for row in shard["unit_rows"]:
            unit_rows.append({**common, **row})
    return layer_rows, unit_rows


def _unit_size(requested: int, metadata: dict[str, Any]) -> int:
    if requested > 0:
        return requested
    for source in (metadata, metadata.get("dump_metadata", {})):
        if isinstance(source, dict):
            for key in ("frame_seq_length", "unit_size", "predictor_stride"):
                value = source.get(key)
                if value is not None and int(value) > 0:
                    return int(value)
    raise ValueError("--unit-size is required when dump metadata lacks frame_seq_length")


def _shard_name(path: Path, layer: int) -> str:
    digest = hashlib.sha256(f"{path.resolve()}:{layer}".encode("utf-8")).hexdigest()[:16]
    return f"{digest}_layer{layer}.json"


def _validate_args(args: argparse.Namespace) -> None:
    if args.unit_size < 0 or args.block_size <= 0 or args.max_validation_dumps < 0:
        raise ValueError("unit size, block size, and dump limit must be non-negative")
    if not 0 < args.hybrid_ratio_threshold <= 1:
        raise ValueError("hybrid ratio threshold must be in (0, 1]")
    if not 0 < args.shuffled_ratio_threshold:
        raise ValueError("shuffled ratio threshold must be positive")
    if not 0 < args.maximum_increase_fraction <= 1:
        raise ValueError("maximum increase fraction must be in (0, 1]")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, payload)
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    gate = summary["gate2_core"]
    lines = [
        "# Conditional Innovation E2 Report",
        "",
        f"- Status: **{summary['status']}**",
        f"- Fixed-bit reconstructed-state core: **{gate['status']}**",
        f"- Closed Hybrid / Cross MSE: {gate['closed_hybrid_over_cross_mse']:.6f}",
        f"- Oracle Hybrid / Cross MSE: {gate['oracle_hybrid_over_cross_mse']:.6f}",
        f"- Shuffled Hybrid / Cross MSE: {gate['shuffled_hybrid_over_cross_mse']:.6f}",
        f"- Attention-output gate: **{summary['attention_output_gate']['status']}**",
        "",
        "> This offline KV-only experiment cannot authorize E3 until matched "
        "BF16 Q tensors are captured and attention-output error is evaluated.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"conditional codec analysis failed: {exc}", file=sys.stderr)
        raise
