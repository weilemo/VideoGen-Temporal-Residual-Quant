#!/usr/bin/env python3
"""Combine the four E2 subgates into the sole E3 authorization record."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec-gamma-summary", required=True)
    parser.add_argument("--codec-e2-summary", required=True)
    parser.add_argument("--offline-subgates", required=True)
    parser.add_argument("--cross-trace-dir", required=True)
    parser.add_argument("--hybrid-trace-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attention-relative-threshold", type=float, default=1.05)
    args = parser.parse_args()

    gamma = _read(Path(args.codec_gamma_summary))
    codec = _read(Path(args.codec_e2_summary))
    offline = _read(Path(args.offline_subgates))
    cross = _trace_values(Path(args.cross_trace_dir))
    hybrid = _trace_values(Path(args.hybrid_trace_dir))
    matched = sorted(set(cross) & set(hybrid))
    ratios = [hybrid[key] / max(cross[key], 1e-12) for key in matched]
    finite = bool(ratios) and all(math.isfinite(value) for value in ratios)
    attention_ratio = float(np.median(ratios)) if ratios else math.inf
    attention_gate = {
        "status": "PASS" if finite and attention_ratio <= args.attention_relative_threshold else "FAIL",
        "matched_records": len(matched),
        "cross_records": len(cross),
        "hybrid_records": len(hybrid),
        "hybrid_over_cross_attention_output_rel_l2_median": attention_ratio,
        "threshold": args.attention_relative_threshold,
    }
    codec_gate = {
        "status": "PASS"
        if gamma.get("status") == "PASS" and codec.get("gate2_core", {}).get("status") == "PASS"
        else "FAIL",
        "calibration_status": gamma.get("status"),
        "validation_core_status": codec.get("gate2_core", {}).get("status"),
        "closed_hybrid_over_cross_mse": codec.get("gate2_core", {}).get(
            "closed_hybrid_over_cross_mse"
        ),
    }
    gates = {
        "attention_output": attention_gate,
        "event_recovery": offline["event_recovery_gate"],
        "saturation": offline["saturation_gate"],
        "codec_gamma": codec_gate,
    }
    status = "PASS_E2" if all(gate["status"] == "PASS" for gate in gates.values()) else "FAIL_E2_SUBGATES"
    result = {
        "schema_version": 1,
        "status": status,
        "gates": gates,
        "e3_authorized": status == "PASS_E2",
        "scientific_scope": (
            "E2 mechanism authorization; automatic KV-shock events are not semantic scene labels"
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["e3_authorized"] else 2


def _trace_values(root: Path) -> dict[tuple[int, int, str], float]:
    values = {}
    for path in sorted(root.rglob("boundary*_layer*.json")):
        payload = _read(path)
        key = (
            int(payload["boundary_frame"]),
            int(payload["layer_idx"]),
            str(path.parent.relative_to(root)),
        )
        value = float(payload["metrics"]["attention_output_rel_l2"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite attention trace: {path}")
        values[key] = value
    return values


def _read(path: Path) -> dict:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
