#!/usr/bin/env python3
"""Run the E1 K-to-V conditional-innovation information test."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trq.analysis.conditional_innovation import (
    CrossKVAccumulator,
    CrossKVModel,
    GammaAccumulator,
    bootstrap_median_ci,
    evaluate_layer,
    innovation_units,
    prompt_aggregates,
)
from trq.analysis.kv_dump import iter_kv_dump_layers, resolve_dump_paths


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one frozen Cross-KV predictor on calibration dumps, test Gate 0 on "
            "prompt-disjoint validation dumps, and only then fit/test temporal innovation gamma."
        )
    )
    parser.add_argument("--calibration-dumps", required=True, help="Glob or comma-separated raw KV dumps")
    parser.add_argument("--validation-dumps", required=True, help="Prompt-disjoint raw KV dumps")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument(
        "--unit-size",
        type=int,
        default=0,
        help="Tokens per latent-frame unit; 0 uses dump frame_seq_length metadata",
    )
    parser.add_argument("--max-calibration-dumps", type=int, default=0)
    parser.add_argument("--max-validation-dumps", type=int, default=0)
    parser.add_argument("--cross-ridge", type=float, default=1e-4)
    parser.add_argument("--gamma-ridge", type=float, default=1e-6)
    parser.add_argument("--gamma-rho", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument(
        "--quantile-max-samples",
        type=int,
        default=262_144,
        help="Maximum deterministic samples per head for diagnostic p99 statistics",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cross-ratio-threshold", type=float, default=0.90)
    parser.add_argument("--correction-ratio-threshold", type=float, default=0.90)
    parser.add_argument("--shuffled-ratio-threshold", type=float, default=0.98)
    parser.add_argument("--minimum-improved-groups", type=float, default=0.60)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit 2 when Gate 0 or Gate 1 does not pass after writing the report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    calibration_paths = _limited(
        _filter_paths_for_layers(resolve_dump_paths(args.calibration_dumps), args.layers),
        args.max_calibration_dumps,
    )
    validation_paths = _limited(
        _filter_paths_for_layers(resolve_dump_paths(args.validation_dumps), args.layers),
        args.max_validation_dumps,
    )
    _validate_disjoint_splits(calibration_paths, validation_paths, args.layers)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cross-fit] fitting {len(calibration_paths)} calibration shards", flush=True)
    models = _fit_cross_models(calibration_paths, args.layers, args.cross_ridge)
    parameters: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "predictor_kind": "cross_kv",
        "layers": {
            layer: {"cross_weight": model.weight, "cross_bias": model.bias}
            for layer, model in models.items()
        },
    }
    torch.save(parameters, output_dir / "cross_kv_checkpoint.pt")
    _write_json(
        output_dir / "progress.json",
        {"schema_version": SCHEMA_VERSION, "stage": "cross_fit_complete"},
    )
    print("[cross-fit] checkpoint saved; evaluating Gate 0", flush=True)
    cross_rows = _evaluate_paths(
        validation_paths,
        args.layers,
        models,
        gammas=None,
        requested_unit_size=args.unit_size,
        seed=args.seed,
        quantile_max_samples=args.quantile_max_samples,
    )
    cross_prompt_rows = prompt_aggregates(cross_rows)
    cross_ci = bootstrap_median_ci(
        [float(row["cross_over_temporal"]) for row in cross_prompt_rows],
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    gate0_pass = (
        float(cross_ci["median"]) <= args.cross_ratio_threshold
        and float(cross_ci["ci95_upper"]) < 1.0
    )

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "STOPPED_CROSS_GATE",
        "scientific_scope": "E1 information test; no quantized closed loop or video-quality claim",
        "gate0_cross_kv": {
            "status": "PASS" if gate0_pass else "FAIL",
            "cross_over_temporal": cross_ci,
            "threshold": args.cross_ratio_threshold,
            "requires_ci_upper_below_one": True,
        },
        "gate1_correction": {"status": "NOT_RUN"},
    }
    final_rows = cross_rows
    _write_json(
        output_dir / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "gate0_complete",
            "gate0_status": "PASS" if gate0_pass else "FAIL",
        },
    )

    if gate0_pass:
        print("[gamma-fit] Gate 0 passed; fitting temporal innovation gamma", flush=True)
        gammas = _fit_gammas(
            calibration_paths,
            args.layers,
            models,
            requested_unit_size=args.unit_size,
            ridge=args.gamma_ridge,
            rho=args.gamma_rho,
        )
        parameters["predictor_kind"] = "hybrid_kv_innovation"
        for layer, gamma in gammas.items():
            parameters["layers"][layer]["innovation_gamma"] = gamma
        torch.save(parameters, output_dir / "hybrid_checkpoint.pt")
        _write_json(
            output_dir / "progress.json",
            {"schema_version": SCHEMA_VERSION, "stage": "gamma_fit_complete"},
        )
        print("[gamma-fit] checkpoint saved; evaluating Gate 1", flush=True)
        final_rows = _evaluate_paths(
            validation_paths,
            args.layers,
            models,
            gammas=gammas,
            requested_unit_size=args.unit_size,
            seed=args.seed,
            quantile_max_samples=args.quantile_max_samples,
        )
        prompt_rows = prompt_aggregates(final_rows)
        correction_ci = bootstrap_median_ci(
            [float(row["oracle_over_cross"]) for row in prompt_rows],
            resamples=args.bootstrap_resamples,
            seed=args.seed + 1,
        )
        shuffled_ci = bootstrap_median_ci(
            [float(row["shuffled_over_cross"]) for row in prompt_rows],
            resamples=args.bootstrap_resamples,
            seed=args.seed + 2,
        )
        improved_fraction = float(
            np.mean([float(row["oracle_over_cross"]) < 1.0 for row in final_rows])
        )
        gamma_values = torch.cat([gamma.reshape(-1) for gamma in gammas.values()]).abs()
        gamma_p95 = float(torch.quantile(gamma_values, 0.95).item())
        gamma_saturation_fraction = float((gamma_values >= args.gamma_rho - 1e-7).float().mean().item())
        gate1_pass = (
            float(correction_ci["median"]) <= args.correction_ratio_threshold
            and float(correction_ci["ci95_upper"]) < 1.0
            and float(shuffled_ci["median"]) >= args.shuffled_ratio_threshold
            and improved_fraction >= args.minimum_improved_groups
            and gamma_p95 < 0.95
        )
        summary["status"] = "PASS_E1" if gate1_pass else "FAIL_CORRECTION_GATE"
        summary["gate1_correction"] = {
            "status": "PASS" if gate1_pass else "FAIL",
            "oracle_over_cross": correction_ci,
            "shuffled_over_cross": shuffled_ci,
            "correction_ratio_threshold": args.correction_ratio_threshold,
            "shuffled_ratio_threshold": args.shuffled_ratio_threshold,
            "improved_layer_head_fraction": improved_fraction,
            "minimum_improved_layer_head_fraction": args.minimum_improved_groups,
            "gamma_abs_p95": gamma_p95,
            "gamma_rho": args.gamma_rho,
            "gamma_saturation_fraction": gamma_saturation_fraction,
            "attention_sensitive_layer_gate": "NOT_EVALUATED_IN_E1",
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": _git_sha(),
        "calibration_dumps": [str(path) for path in calibration_paths],
        "validation_dumps": [str(path) for path in validation_paths],
        "config": vars(args),
    }
    torch.save(parameters, output_dir / "conditional_innovation_params.pt")
    _write_csv(output_dir / "layer_head_rows.csv", final_rows)
    _write_csv(output_dir / "prompt_rows.csv", prompt_aggregates(final_rows))
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "manifest.json", manifest)
    _write_report(output_dir / "report.md", summary, len(calibration_paths), len(validation_paths))
    _write_json(
        output_dir / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "analysis_complete",
            "status": summary["status"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    if args.require_pass and summary["status"] != "PASS_E1":
        return 2
    return 0


def _fit_cross_models(paths: list[Path], layers: str, ridge: float) -> dict[int, CrossKVModel]:
    accumulators: dict[int, CrossKVAccumulator] = {}
    interval = max(1, len(paths) // 20)
    for index, path in enumerate(paths, start=1):
        if index == 1 or index % interval == 0 or index == len(paths):
            print(f"[cross-fit] shard {index}/{len(paths)}: {path.name}", flush=True)
        for layer, key, value, _ in iter_kv_dump_layers(path, layers=layers):
            accumulator = accumulators.setdefault(
                layer, CrossKVAccumulator(key.shape[1], key.shape[-1])
            )
            accumulator.update(key, value)
    if not accumulators:
        raise RuntimeError("no calibration layer records matched the requested layers")
    return {layer: accumulator.fit(ridge) for layer, accumulator in sorted(accumulators.items())}


def _fit_gammas(
    paths: list[Path],
    layers: str,
    models: dict[int, CrossKVModel],
    *,
    requested_unit_size: int,
    ridge: float,
    rho: float,
) -> dict[int, torch.Tensor]:
    accumulators: dict[int, GammaAccumulator] = {}
    interval = max(1, len(paths) // 20)
    for index, path in enumerate(paths, start=1):
        if index == 1 or index % interval == 0 or index == len(paths):
            print(f"[gamma-fit] shard {index}/{len(paths)}: {path.name}", flush=True)
        for layer, key, value, metadata in iter_kv_dump_layers(path, layers=layers):
            model = models.get(layer)
            if model is None:
                raise ValueError(f"validation/calibration layer mismatch: {layer}")
            unit_size = _unit_size(requested_unit_size, metadata)
            innovations = innovation_units(key, value, model, unit_size)
            accumulator = accumulators.setdefault(
                layer, GammaAccumulator(key.shape[1], key.shape[-1])
            )
            accumulator.update(innovations)
    return {
        layer: accumulator.fit(ridge, rho)
        for layer, accumulator in sorted(accumulators.items())
    }


def _evaluate_paths(
    paths: list[Path],
    layers: str,
    models: dict[int, CrossKVModel],
    *,
    gammas: dict[int, torch.Tensor] | None,
    requested_unit_size: int,
    seed: int,
    quantile_max_samples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prompt_groups = _group_paths_by_prompt(paths, layers)
    prompt_ids = sorted(prompt_groups)
    for prompt_index, prompt_id in enumerate(prompt_ids):
        stage = "Gate 1" if gammas is not None else "Gate 0"
        print(
            f"[evaluate] {stage} prompt {prompt_index + 1}/{len(prompt_ids)}: {prompt_id}",
            flush=True,
        )
        prompt_paths = prompt_groups[prompt_id]
        donor_prompt_id = prompt_ids[(prompt_index + 1) % len(prompt_ids)]
        donor_paths = prompt_groups[donor_prompt_id]
        donor_layers: dict[int, tuple[list[torch.Tensor], int]] = {}
        if gammas is not None:
            for donor_path in donor_paths:
                for donor_layer, donor_key, donor_value, donor_metadata in iter_kv_dump_layers(
                    donor_path, layers=layers
                ):
                    donor_unit_size = _unit_size(requested_unit_size, donor_metadata)
                    donor_layers[donor_layer] = (
                        innovation_units(
                            donor_key,
                            donor_value,
                            models[donor_layer],
                            donor_unit_size,
                        ),
                        donor_unit_size,
                    )
        for path in prompt_paths:
            for layer, key, value, metadata in iter_kv_dump_layers(path, layers=layers):
                if layer not in models:
                    raise ValueError(f"validation layer {layer} has no fitted Cross-KV model")
                unit_size = _unit_size(requested_unit_size, metadata)
                gamma = None if gammas is None else gammas[layer]
                shuffled_innovations = None
                if gammas is not None:
                    if layer not in donor_layers:
                        raise ValueError(
                            f"shuffled donor prompt {donor_prompt_id} lacks layer {layer}"
                        )
                    shuffled_innovations, donor_unit_size = donor_layers[layer]
                    if donor_unit_size != unit_size:
                        raise ValueError(
                            f"shuffled donor unit size {donor_unit_size} differs from {unit_size}"
                        )
                for row in evaluate_layer(
                    key,
                    value,
                    models[layer],
                    unit_size=unit_size,
                    gamma=gamma,
                    shuffled_innovations=shuffled_innovations,
                    shuffle_seed=seed + 1009 * prompt_index + layer,
                    quantile_max_samples=quantile_max_samples,
                ):
                    rows.append(
                        {
                            "prompt_id": prompt_id,
                            "dump_path": str(path),
                            "layer": layer,
                            "unit_size": unit_size,
                            **row,
                        }
                    )
    if not rows:
        raise RuntimeError("no validation layer records matched the requested layers")
    return rows


def _validate_disjoint_splits(calibration: list[Path], validation: list[Path], layers: str) -> None:
    calibration_ids = set(_group_paths_by_prompt(calibration, layers))
    validation_ids = set(_group_paths_by_prompt(validation, layers))
    if len(calibration_ids) < 2 or len(validation_ids) < 2:
        raise ValueError(
            "E1 requires at least two calibration and two validation prompts; "
            "the shuffled control uses a different validation prompt"
        )
    overlap = set(calibration) & set(validation)
    if overlap:
        raise ValueError(f"calibration/validation paths overlap: {sorted(str(path) for path in overlap)}")
    prompt_overlap = calibration_ids & validation_ids
    if prompt_overlap:
        raise ValueError(f"calibration/validation prompt identities overlap: {sorted(prompt_overlap)}")


def _first_prompt_id(path: Path, layers: str) -> str:
    try:
        _, _, _, metadata = next(iter(iter_kv_dump_layers(path, layers=layers)))
    except StopIteration as exc:
        raise ValueError(f"dump has no requested layers: {path}") from exc
    return _prompt_id(path, metadata)


def _prompt_id(path: Path, metadata: dict[str, Any]) -> str:
    dump_metadata = metadata.get("dump_metadata", {})
    if isinstance(dump_metadata, dict):
        for key in ("prompt_id", "prompt_index", "sample_id", "name"):
            if key in dump_metadata and str(dump_metadata[key]).strip():
                seed = dump_metadata.get("seed")
                suffix = f"/seed={seed}" if seed is not None else ""
                return f"{key}={dump_metadata[key]}{suffix}"
    return re.sub(r"(?:_layer|\.layer)[_-]?\d+$", "", path.stem, flags=re.IGNORECASE)


def _group_paths_by_prompt(paths: list[Path], layers: str) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in paths:
        prompt_id = _first_prompt_id(path, layers)
        groups.setdefault(prompt_id, []).append(path)
    return groups


def _filter_paths_for_layers(paths: list[Path], layers: str) -> list[Path]:
    result = []
    for path in paths:
        try:
            next(iter(iter_kv_dump_layers(path, layers=layers)))
        except StopIteration:
            continue
        result.append(path)
    if not result:
        raise ValueError("no dump files contain the requested layers")
    return result


def _unit_size(requested: int, metadata: dict[str, Any]) -> int:
    if requested > 0:
        return requested
    for source in (metadata, metadata.get("dump_metadata", {})):
        if isinstance(source, dict):
            for key in ("frame_seq_length", "unit_size", "predictor_stride"):
                value = source.get(key)
                if value is not None and int(value) > 0:
                    return int(value)
    raise ValueError("--unit-size is required when dump metadata has no frame_seq_length")


def _limited(paths: list[Path], maximum: int) -> list[Path]:
    if maximum < 0:
        raise ValueError("dump limits must be non-negative")
    return paths if maximum == 0 else paths[:maximum]


def _validate_args(args: argparse.Namespace) -> None:
    if args.unit_size < 0:
        raise ValueError("unit-size must be non-negative")
    if not 0 < args.minimum_improved_groups <= 1:
        raise ValueError("minimum-improved-groups must be in (0, 1]")
    if not 0 < args.gamma_rho < 1:
        raise ValueError("gamma-rho must be in (0, 1)")
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap-resamples must be positive")
    if args.quantile_max_samples <= 0:
        raise ValueError("quantile-max-samples must be positive")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(path: Path, summary: dict[str, Any], calibration_count: int, validation_count: int) -> None:
    gate0 = summary["gate0_cross_kv"]
    gate1 = summary["gate1_correction"]
    lines = [
        "# Conditional Innovation E1 Report",
        "",
        f"- Status: **{summary['status']}**",
        f"- Calibration dumps: {calibration_count}",
        f"- Validation dumps: {validation_count}",
        f"- Gate 0: **{gate0['status']}**",
        f"- Cross / temporal median: {gate0['cross_over_temporal']['median']:.6f}",
        f"- Cross / temporal CI95 upper: {gate0['cross_over_temporal']['ci95_upper']:.6f}",
        f"- Gate 1: **{gate1['status']}**",
    ]
    if gate1["status"] != "NOT_RUN":
        lines.extend(
            [
                f"- Oracle correction / Cross median: {gate1['oracle_over_cross']['median']:.6f}",
                f"- Correction CI95 upper: {gate1['oracle_over_cross']['ci95_upper']:.6f}",
                f"- Shuffled correction / Cross median: {gate1['shuffled_over_cross']['median']:.6f}",
                f"- Improved layer/head fraction: {gate1['improved_layer_head_fraction']:.3f}",
                f"- |gamma| P95: {gate1['gamma_abs_p95']:.6f}",
            ]
        )
    lines.extend(
        [
            "",
            "> This is an unquantized E1 information test. It does not validate reconstructed-state stability, attention-output error, video quality, physical bytes, peak VRAM, or latency.",
            "",
        ]
    )
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
        print(f"conditional innovation analysis failed: {exc}", file=sys.stderr)
        raise
