#!/usr/bin/env python3
"""Compare TRQ and an external QVG S2++ identity codec on raw BF16 KV dumps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from trq.analysis.codec_parity import compare_identity_codecs
from trq.analysis.kv_dump import iter_kv_dumps, resolve_dump_paths


DEFAULT_CONFIGS = (
    "common_s1560_int2_a4_g64:2:4:64:1560",
    "stride448_int2_a4_g64:2:4:64:448",
    "legacy_student_s448_int4_a8_g32:4:8:32:448",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TRQ and QVG S2++ identity on exactly the same raw KV tensors."
    )
    parser.add_argument("--dumps", required=True, help="Glob or comma-separated raw KV dumps")
    parser.add_argument("--qvg-root", required=True, help="Root of jiahui1021/qvg checkout")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="0", help="all, comma list, or ranges")
    parser.add_argument("--kv", choices=("both", "K", "V"), default="both")
    parser.add_argument("--max-dumps", type=int, default=1, help="Zero means all")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument("--codec-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--scale-precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="NAME:NUM_BITS:ANCHOR_BITS:BLOCK_SIZE:STRIDE; repeat for a matrix",
    )
    parser.add_argument(
        "--student-triton",
        action="store_true",
        help="Also compare the student's strict Triton decoder; requires CUDA",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qvg_root = Path(args.qvg_root).expanduser().resolve()
    student_module, codec_path = _load_student_codec(qvg_root)
    dump_paths = resolve_dump_paths(args.dumps)
    if args.max_dumps > 0:
        dump_paths = dump_paths[: args.max_dumps]
    configs = [_parse_config(value) for value in (args.configs or DEFAULT_CONFIGS)]
    device = _resolve_device(args.device)
    codec_dtype = torch.bfloat16 if args.codec_dtype == "bf16" else torch.float32
    scale_precision = torch.bfloat16 if args.scale_precision == "bf16" else torch.float32
    kv_keys = ("K", "V") if args.kv == "both" else (args.kv,)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = {
        "dumps": [str(path) for path in dump_paths],
        "qvg_root": str(qvg_root),
        "student_codec": str(codec_path),
        "layers": args.layers,
        "kv": list(kv_keys),
        "device": str(device),
        "codec_dtype": args.codec_dtype,
        "scale_precision": args.scale_precision,
        "student_triton": bool(args.student_triton),
        "configs": configs,
    }
    _write_json(output_dir / "resolved_config.json", resolved_config)

    rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    for layer_idx, k_tensor, v_tensor, metadata in iter_kv_dumps(
        dump_paths, layers=args.layers
    ):
        dump_path = str(metadata.get("dump_path", "unknown"))
        dump_id = str(
            metadata.get("dump_metadata", {}).get("sample_id")
            or Path(dump_path).stem
        )
        tensors = {"K": k_tensor, "V": v_tensor}
        for config in configs:
            for kv_key in kv_keys:
                tensor = tensors[kv_key].to(device=device)
                comparison, comparison_units = compare_identity_codecs(
                    tensor,
                    student_module=student_module,
                    num_bits=config["num_bits"],
                    anchor_bits=config["anchor_bits"],
                    block_size=config["block_size"],
                    predictor_stride=config["predictor_stride"],
                    layer_idx=layer_idx,
                    scale_precision=scale_precision,
                    codec_dtype=codec_dtype,
                    run_student_triton=args.student_triton,
                )
                base = {
                    "dump_id": dump_id,
                    "dump_path": dump_path,
                    "layer": int(layer_idx),
                    "kv": kv_key,
                    "config": config["name"],
                    "num_bits": config["num_bits"],
                    "anchor_bits": config["anchor_bits"],
                    "block_size": config["block_size"],
                    "predictor_stride": config["predictor_stride"],
                }
                row = dict(base)
                row.update(_flatten_comparison(comparison))
                rows.append(row)
                unit_rows.extend({**base, **unit_row} for unit_row in comparison_units)
                print(
                    f"{dump_id} L{layer_idx:02d} {kv_key} {config['name']}: "
                    f"TRQ-vs-S2 rel-L2={comparison['trq_vs_student_torch']['rel_l2']:.6g}, "
                    f"TRQ={comparison['trq_vs_raw']['rel_l2']:.6g}, "
                    f"S2={comparison['student_torch_vs_raw']['rel_l2']:.6g}"
                )

    summary = _summarize(rows)
    _write_csv(output_dir / "parity_rows.csv", rows)
    _write_jsonl(output_dir / "parity_rows.jsonl", rows)
    _write_csv(output_dir / "unit_rows.csv", unit_rows)
    _write_jsonl(output_dir / "unit_rows.jsonl", unit_rows)
    _write_json(output_dir / "summary.json", summary)
    manifest = {
        "schema": "trq_identity_codec_parity",
        "version": 1,
        "main_repo_commit": _git_value(Path(__file__).resolve().parents[3], "rev-parse", "HEAD"),
        "main_repo_status": _git_value(
            Path(__file__).resolve().parents[3], "status", "--short"
        ),
        "qvg_commit": _git_value(qvg_root, "rev-parse", "HEAD"),
        "qvg_status": _git_value(qvg_root, "status", "--short"),
        "student_codec_sha256": _sha256(codec_path),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_report(output_dir / "report.md", resolved_config, summary, manifest)


def _load_student_codec(qvg_root: Path):
    package_root = qvg_root / "references" / "quant-videogen"
    codec_path = package_root / "quant_videogen" / "real" / "s2pp.py"
    if not codec_path.is_file():
        raise FileNotFoundError(
            f"Student S2++ codec not found at {codec_path}; clone https://github.com/jiahui1021/qvg"
        )
    sys.path.insert(0, str(package_root))
    module = importlib.import_module("quant_videogen.real.s2pp")
    loaded_path = Path(module.__file__).resolve()
    if loaded_path != codec_path.resolve():
        raise RuntimeError(f"Loaded unexpected student codec: {loaded_path} != {codec_path}")
    return module, codec_path.resolve()


def _parse_config(value: str) -> dict[str, Any]:
    parts = value.split(":")
    if len(parts) != 5 or not parts[0]:
        raise ValueError(
            f"invalid config {value!r}; expected NAME:NUM_BITS:ANCHOR_BITS:BLOCK_SIZE:STRIDE"
        )
    name = parts[0]
    num_bits, anchor_bits, block_size, stride = (int(part) for part in parts[1:])
    if num_bits not in (2, 4, 8) or anchor_bits not in (2, 4, 8):
        raise ValueError(f"unsupported bit width in config {value!r}")
    if block_size <= 0 or stride <= 0:
        raise ValueError(f"block size and stride must be positive in config {value!r}")
    return {
        "name": name,
        "num_bits": num_bits,
        "anchor_bits": anchor_bits,
        "block_size": block_size,
        "predictor_stride": stride,
    }


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def _flatten_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
    row = {
        "shape": "x".join(str(value) for value in comparison["shape"]),
        "unit_lengths_match": comparison["state_unit_lengths_match"],
        "trq_unit_lengths": ",".join(map(str, comparison["trq_state"]["unit_lengths"])),
        "student_unit_lengths": ",".join(map(str, comparison["student_state"]["unit_lengths"])),
        "trq_state_bytes": comparison["trq_state"]["state_bytes"],
        "student_state_bytes": comparison["student_state"]["state_bytes"],
    }
    for name, metrics in comparison.items():
        if isinstance(metrics, dict) and "rel_l2" in metrics:
            for metric, value in metrics.items():
                row[f"{name}_{metric}"] = value
    return row


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["config"], row["kv"]), []).append(row)
    result = []
    metric_names = (
        "trq_vs_raw_rel_l2",
        "student_torch_vs_raw_rel_l2",
        "trq_vs_student_torch_rel_l2",
        "trq_encoder_decoder_parity_max_abs",
        "student_torch_vs_trq_compat_decoder_max_abs",
        "student_torch_vs_triton_max_abs",
    )
    for (config, kv_key), selected in sorted(groups.items()):
        summary = {"config": config, "kv": kv_key, "count": len(selected)}
        for metric in metric_names:
            values = [float(row[metric]) for row in selected if metric in row]
            if values:
                summary[f"mean_{metric}"] = sum(values) / len(values)
                summary[f"max_{metric}"] = max(values)
        summary["unit_length_mismatch_count"] = sum(
            not bool(row["unit_lengths_match"]) for row in selected
        )
        result.append(summary)
    return {"groups": result}


def _write_report(path: Path, config: dict, summary: dict, manifest: dict) -> None:
    lines = [
        "# Identity Codec Parity Report",
        "",
        f"- Main commit: `{manifest['main_repo_commit']}`",
        f"- Student QVG commit: `{manifest['qvg_commit']}`",
        f"- Student codec SHA256: `{manifest['student_codec_sha256']}`",
        f"- Device: `{config['device']}`",
        "",
        "| Config | KV | N | TRQ raw Rel-L2 | S2 raw Rel-L2 | TRQ vs S2 Rel-L2 | Decoder parity max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["groups"]:
        lines.append(
            "| {config} | {kv} | {count} | {trq:.6g} | {s2:.6g} | {diff:.6g} | {parity:.6g} |".format(
                config=row["config"],
                kv=row["kv"],
                count=row["count"],
                trq=row.get("mean_trq_vs_raw_rel_l2", float("nan")),
                s2=row.get("mean_student_torch_vs_raw_rel_l2", float("nan")),
                diff=row.get("mean_trq_vs_student_torch_rel_l2", float("nan")),
                parity=max(
                    row.get("max_trq_encoder_decoder_parity_max_abs", 0.0),
                    row.get("max_student_torch_vs_trq_compat_decoder_max_abs", 0.0),
                ),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: first require internal decoder parity. If TRQ and S2++ differ on the same raw tensor with the same config, the codec is the source. If they match offline but generated videos differ, inspect runtime scheduling and the first online quantization event.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_value(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
