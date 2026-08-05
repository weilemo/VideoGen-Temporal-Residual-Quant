#!/usr/bin/env python3
"""Test whether aligned current K adds held-out information beyond previous V."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from trq.analysis.conditional_increment import (
    JOINT_MODEL_NAMES,
    aggregate_prompt_rows,
    evaluate_conditional_record,
    fit_conditional_models,
)
from trq.analysis.conditional_innovation import bootstrap_median_ci
from trq.analysis.kv_dump import iter_kv_dump_layers, resolve_dump_paths


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit prompt-disjoint temporal and T+K ridge predictors with matched "
            "wrong-space, wrong-time, and wrong-prompt controls."
        )
    )
    parser.add_argument("--calibration-dumps", required=True)
    parser.add_argument("--validation-dumps", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--unit-size", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--wrong-space-shift", type=int, default=1)
    parser.add_argument("--sample-chunk", type=int, default=4096)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-partial-r2", type=float, default=0.01)
    parser.add_argument("--minimum-control-margin", type=float, default=0.005)
    parser.add_argument("--minimum-improved-groups", type=float, default=0.60)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    calibration = _index_split(args.calibration_dumps, args.layers)
    validation = _index_split(args.validation_dumps, args.layers)
    _validate_splits(calibration, validation)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[fit] prompts={len(calibration)} layers={args.layers} ridge={args.ridge}",
        flush=True,
    )
    models = fit_conditional_models(
        _records(calibration, args.layers, args.unit_size),
        ridge=args.ridge,
        wrong_space_shift=args.wrong_space_shift,
        sample_chunk=args.sample_chunk,
    )
    _write_json(output_dir / "progress.json", {"stage": "fit_complete"})

    layer_head_rows: list[dict[str, Any]] = []
    for prompt_index, record in enumerate(
        _records(validation, args.layers, args.unit_size, include_prompt=True), start=1
    ):
        prompt_id, layer, key, value, donor_key, unit_size = record
        print(
            f"[evaluate] prompt={prompt_id} layer={layer} ({prompt_index})",
            flush=True,
        )
        if layer not in models:
            raise ValueError(f"validation layer {layer} was not fitted")
        for row in evaluate_conditional_record(
            key,
            value,
            donor_key,
            models[layer],
            unit_size=unit_size,
            wrong_space_shift=args.wrong_space_shift,
        ):
            layer_head_rows.append(
                {"prompt_id": prompt_id, "layer": layer, "unit_size": unit_size, **row}
            )

    prompt_rows = aggregate_prompt_rows(layer_head_rows)
    summary = _summarize(prompt_rows, layer_head_rows, args)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": _git_sha(),
        "calibration_prompts": _manifest_split(calibration),
        "validation_prompts": _manifest_split(validation),
        "config": vars(args),
        "scientific_scope": (
            "held-out affine/ridge conditional predictive gain only; no conditional "
            "mutual-information, online-codec, video-quality, or system claim"
        ),
    }
    _write_csv(output_dir / "layer_head_rows.csv", layer_head_rows)
    _write_csv(output_dir / "prompt_rows.csv", prompt_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "manifest.json", manifest)
    _write_report(output_dir / "report.md", summary)
    _write_json(output_dir / "progress.json", {"stage": "analysis_complete", "status": summary["status"]})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if args.require_pass and summary["status"] != "PASS_STRUCTURE_GATE" else 0


def _index_split(path_spec: str, layers: str) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for path in resolve_dump_paths(path_spec):
        try:
            _, _, _, metadata = next(iter(iter_kv_dump_layers(path, layers=layers)))
        except StopIteration:
            continue
        prompt_id = _prompt_id(path, metadata)
        result.setdefault(prompt_id, []).append(path)
    if len(result) < 2:
        raise ValueError("each split requires at least two prompt identities")
    return {prompt: sorted(paths) for prompt, paths in sorted(result.items())}


def _records(
    split: dict[str, list[Path]],
    layers: str,
    requested_unit_size: int,
    *,
    include_prompt: bool = False,
) -> Iterator[Any]:
    prompt_ids = sorted(split)
    for index, prompt_id in enumerate(prompt_ids):
        donor_id = prompt_ids[(index + 1) % len(prompt_ids)]
        target_layers = _load_prompt_layers(split[prompt_id], layers)
        donor_layers = _load_prompt_layers(split[donor_id], layers)
        if set(target_layers) != set(donor_layers):
            raise ValueError(
                f"target/donor layer mismatch: {prompt_id} vs {donor_id}"
            )
        for layer in sorted(target_layers):
            key, value, metadata = target_layers[layer]
            donor_key, _, donor_metadata = donor_layers[layer]
            unit_size = _unit_size(requested_unit_size, metadata)
            donor_unit_size = _unit_size(requested_unit_size, donor_metadata)
            if unit_size != donor_unit_size:
                raise ValueError(
                    f"target/donor unit-size mismatch at layer {layer}: "
                    f"{unit_size} vs {donor_unit_size}"
                )
            payload = (layer, key, value, donor_key, unit_size)
            yield (prompt_id, *payload) if include_prompt else payload


def _load_prompt_layers(
    paths: list[Path], layers: str
) -> dict[int, tuple[Any, Any, dict[str, Any]]]:
    result: dict[int, tuple[Any, Any, dict[str, Any]]] = {}
    for path in paths:
        for layer, key, value, metadata in iter_kv_dump_layers(path, layers=layers):
            if layer in result:
                raise ValueError(f"duplicate layer {layer} within prompt dump group")
            result[layer] = (key, value, metadata)
    if not result:
        raise ValueError("prompt dump group contains no requested layers")
    return result


def _summarize(
    prompt_rows: list[dict[str, float | str]],
    layer_head_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    joint_values = [float(row["partial_r2_joint"]) for row in prompt_rows]
    joint_ci = bootstrap_median_ci(
        joint_values, resamples=args.bootstrap_resamples, seed=args.seed
    )
    controls: dict[str, Any] = {}
    controls_pass = True
    for offset, control in enumerate(JOINT_MODEL_NAMES[1:], start=1):
        key = f"joint_minus_{control}"
        ci = bootstrap_median_ci(
            [float(row[key]) for row in prompt_rows],
            resamples=args.bootstrap_resamples,
            seed=args.seed + offset,
        )
        passed = (
            float(ci["median"]) >= args.minimum_control_margin
            and float(ci["ci95_lower"]) > 0
        )
        controls[control] = {"status": "PASS" if passed else "FAIL", "margin": ci}
        controls_pass = controls_pass and passed

    grouped: dict[tuple[str, int, int], dict[str, float]] = {}
    for row in layer_head_rows:
        key = (str(row["prompt_id"]), int(row["layer"]), int(row["head"]))
        grouped.setdefault(key, {})[str(row["method"])] = float(row["sse"])
    improved_fraction = float(
        np.mean(
            [values["joint"] < values["temporal"] for values in grouped.values()]
        )
    )
    joint_pass = (
        float(joint_ci["median"]) >= args.minimum_partial_r2
        and float(joint_ci["ci95_lower"]) > 0
        and improved_fraction >= args.minimum_improved_groups
    )
    passed = joint_pass and controls_pass
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_STRUCTURE_GATE" if passed else "FAIL_STRUCTURE_GATE",
        "scientific_scope": (
            "held-out affine/ridge conditional predictive gain only; passing does not "
            "authorize an online codec or MB32"
        ),
        "joint_increment": {
            "status": "PASS" if joint_pass else "FAIL",
            "partial_r2": joint_ci,
            "minimum_partial_r2": args.minimum_partial_r2,
            "improved_prompt_layer_head_fraction": improved_fraction,
            "minimum_improved_groups": args.minimum_improved_groups,
        },
        "matched_controls": controls,
        "next_stage": (
            "repair first-boundary K parity and actual-byte accounting, then design a new "
            "small online test"
            if passed
            else "stop K-to-V as a main method; retain marginal correlation as an ablation"
        ),
    }


def _validate_splits(
    calibration: dict[str, list[Path]], validation: dict[str, list[Path]]
) -> None:
    overlap = set(calibration) & set(validation)
    if overlap:
        raise ValueError(f"calibration/validation prompt identities overlap: {sorted(overlap)}")
    path_overlap = {
        path for paths in calibration.values() for path in paths
    } & {path for paths in validation.values() for path in paths}
    if path_overlap:
        raise ValueError("calibration/validation dump paths overlap")


def _validate_args(args: argparse.Namespace) -> None:
    if args.unit_size < 0 or args.ridge < 0 or args.sample_chunk <= 0:
        raise ValueError("unit-size/ridge/sample-chunk arguments are invalid")
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap-resamples must be positive")
    if args.minimum_partial_r2 < 0 or args.minimum_control_margin < 0:
        raise ValueError("effect thresholds must be non-negative")
    if not 0 < args.minimum_improved_groups <= 1:
        raise ValueError("minimum-improved-groups must be in (0, 1]")


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


def _prompt_id(path: Path, metadata: dict[str, Any]) -> str:
    dump_metadata = metadata.get("dump_metadata", {})
    if isinstance(dump_metadata, dict):
        for key in ("prompt_id", "prompt_index", "sample_id", "name"):
            if key in dump_metadata and str(dump_metadata[key]).strip():
                seed = dump_metadata.get("seed")
                suffix = f"/seed={seed}" if seed is not None else ""
                return f"{key}={dump_metadata[key]}{suffix}"
    return re.sub(r"(?:_layer|\.layer)[_-]?\d+$", "", path.stem, flags=re.IGNORECASE)


def _manifest_split(split: dict[str, list[Path]]) -> dict[str, list[str]]:
    return {prompt: [str(path) for path in paths] for prompt, paths in split.items()}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    joint = summary["joint_increment"]
    lines = [
        "# K-to-V Conditional Increment Report",
        "",
        f"- Status: **{summary['status']}**",
        f"- Joint partial R2 median: {joint['partial_r2']['median']:.6f}",
        f"- Joint partial R2 CI95: [{joint['partial_r2']['ci95_lower']:.6f}, "
        f"{joint['partial_r2']['ci95_upper']:.6f}]",
        f"- Improved prompt/layer/head fraction: "
        f"{joint['improved_prompt_layer_head_fraction']:.6f}",
        "",
        "## Matched Controls",
        "",
    ]
    for name, control in summary["matched_controls"].items():
        lines.append(
            f"- {name}: **{control['status']}**, correct-minus-control median "
            f"{control['margin']['median']:.6f}"
        )
    lines.extend(
        [
            "",
            "> This is an offline held-out affine/ridge structure test. It does not "
            "establish conditional mutual information or end-to-end codec benefit.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"conditional increment analysis failed: {exc}", file=sys.stderr)
        raise
