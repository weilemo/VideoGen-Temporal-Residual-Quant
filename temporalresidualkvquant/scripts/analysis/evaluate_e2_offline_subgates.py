#!/usr/bin/env python3
"""Evaluate saturation and automatic KV-shock recovery on validation dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from trq.analysis.conditional_codec import codec_gamma_statistics
from trq.analysis.conditional_innovation import CrossKVModel
from trq.analysis.kv_dump import iter_kv_dump_layers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec-gamma-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--unit-size", type=int, default=0)
    parser.add_argument("--key-bits", type=int, default=4)
    parser.add_argument("--value-bits", type=int, default=4)
    parser.add_argument("--anchor-bits", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--event-quantile", type=float, default=0.9)
    parser.add_argument("--recovery-units", type=int, default=2)
    parser.add_argument("--minimum-recovery-rate", type=float, default=0.8)
    args = parser.parse_args()

    root = Path(args.codec_gamma_dir).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    params = torch.load(
        root / "conditional_innovation_params.pt", map_location="cpu", weights_only=False
    )
    quant = {
        "values": 0,
        "nonfinite": 0,
        "overflow": 0,
        "endpoint": 0,
        "overflow_code_excess_sum": 0.0,
        "max_code_excess": 0.0,
        "by_role": {},
        "by_layer": {},
    }
    groups: list[list[dict]] = []
    for path_string in manifest["validation_dumps"]:
        for layer, key, value, metadata in iter_kv_dump_layers(Path(path_string), layers=args.layers):
            layer_params = params["layers"].get(layer, params["layers"].get(str(layer)))
            model = CrossKVModel(layer_params["cross_weight"], layer_params["cross_bias"])
            local = {"values": 0, "nonfinite": 0, "overflow": 0, "endpoint": 0}
            codec_gamma_statistics(
                key,
                value,
                model,
                layer_params["innovation_gamma"],
                unit_size=args.unit_size or _unit_size(metadata),
                key_bits=args.key_bits,
                value_bits=args.value_bits,
                anchor_bits=args.anchor_bits,
                block_size=args.block_size,
                diagnostics=local,
            )
            groups.append(local.get("event_rows", []))
            for key_name in ("values", "nonfinite", "overflow", "endpoint"):
                quant[key_name] += int(local[key_name])
            quant["overflow_code_excess_sum"] += float(
                local.get("overflow_code_excess_sum", 0.0)
            )
            quant["max_code_excess"] = max(
                quant["max_code_excess"], float(local.get("max_code_excess", 0.0))
            )
            _merge_breakdown(quant["by_role"], local.get("by_role", {}))
            _merge_breakdown(quant["by_layer"], {str(layer): local})

    all_rows = [row for group in groups for row in group]
    threshold = float(np.quantile([row["shock_rel_l2"] for row in all_rows], args.event_quantile))
    recovered = 0
    events = 0
    for group in groups:
        for index, row in enumerate(group):
            if index == 0 or row["shock_rel_l2"] < threshold:
                continue
            events += 1
            baseline = group[max(0, index - 1)]["closed_rel_l2"]
            horizon = group[index : index + args.recovery_units + 1]
            if any(candidate["closed_rel_l2"] <= 1.05 * baseline for candidate in horizon):
                recovered += 1
    recovery_rate = recovered / max(events, 1)
    overflow_fraction = quant["overflow"] / max(quant["values"], 1)
    result = {
        "saturation_gate": {
            "status": "PASS" if quant["nonfinite"] == 0 and overflow_fraction <= 1e-6 else "FAIL",
            "preclamp_overflow_fraction": overflow_fraction,
            "endpoint_occupancy_fraction": quant["endpoint"] / max(quant["values"], 1),
            "nonfinite_values": quant["nonfinite"],
            "max_code_excess": quant["max_code_excess"],
            "mean_code_excess_overflow": quant["overflow_code_excess_sum"]
            / max(quant["overflow"], 1),
            "by_role": _finalize_breakdown(quant["by_role"]),
            "by_layer": _finalize_breakdown(quant["by_layer"]),
            "note": "S2++ ranges include zero and use representability-safe stored scales; endpoint occupancy is diagnostic only",
        },
        "event_recovery_gate": {
            "status": "PASS" if events > 0 and recovery_rate >= args.minimum_recovery_rate else "FAIL",
            "event_definition": "top-quantile BF16 V unit-to-unit relative-L2 shock",
            "semantic_event_labels": "NOT_EVALUATED",
            "shock_threshold": threshold,
            "events": events,
            "recovered_within_units": args.recovery_units,
            "recovery_rate": recovery_rate,
            "minimum_recovery_rate": args.minimum_recovery_rate,
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(gate["status"] == "PASS" for gate in result.values()) else 2


def _unit_size(metadata: dict) -> int:
    for source in (metadata, metadata.get("dump_metadata", {})):
        if isinstance(source, dict):
            for key in ("frame_seq_length", "unit_size", "predictor_stride"):
                if source.get(key):
                    return int(source[key])
    raise ValueError("unit size missing from dump metadata")


def _merge_breakdown(target: dict, source: dict) -> None:
    for name, values in source.items():
        merged = target.setdefault(
            str(name),
            {
                "values": 0,
                "nonfinite": 0,
                "overflow": 0,
                "endpoint": 0,
                "overflow_code_excess_sum": 0.0,
                "max_code_excess": 0.0,
            },
        )
        for key in ("values", "nonfinite", "overflow", "endpoint"):
            merged[key] += int(values.get(key, 0))
        merged["overflow_code_excess_sum"] += float(
            values.get("overflow_code_excess_sum", 0.0)
        )
        merged["max_code_excess"] = max(
            merged["max_code_excess"], float(values.get("max_code_excess", 0.0))
        )


def _finalize_breakdown(values: dict) -> dict:
    result = {}
    for name, row in sorted(values.items()):
        result[name] = {
            **row,
            "preclamp_overflow_fraction": row["overflow"] / max(row["values"], 1),
            "endpoint_occupancy_fraction": row["endpoint"] / max(row["values"], 1),
            "mean_code_excess_overflow": row["overflow_code_excess_sum"]
            / max(row["overflow"], 1),
        }
    return result


if __name__ == "__main__":
    raise SystemExit(main())
