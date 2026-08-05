"""Paired analysis and gates for the Cross-KV versus temporal smoke experiment."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .paired_rollout import load_rollouts, paired_distances, validate_paired_metadata


LOWER_ATTENTION_METRICS = (
    "v_cache_read_rel_l2",
    "attention_output_rel_l2",
    "attention_logits_rel_l2",
    "softmax_kl",
)
HIGHER_ATTENTION_METRICS = (
    "attention_top1_agreement",
    "attention_topk_overlap",
)
MANUAL_FIELDS = (
    "identity_switch",
    "background_jump",
    "texture_repetition",
    "motion_freeze",
    "color_drift",
    "black_or_nan",
    "catastrophe",
    "bf16_present",
)
TRUE_VALUES = {"1", "true", "yes", "y", "fail", "failed", "catastrophe"}


def analyze_cross_kv_smoke(
    *,
    bf16_root: str | Path,
    temporal_root: str | Path,
    cross_root: str | Path,
    temporal_trace_root: str | Path,
    cross_trace_root: str | Path,
    runtime_roots: dict[str, str | Path] | None = None,
    quality_runs: Iterable[tuple[str, int, str | Path]] = (),
    failure_tags: str | Path | None = None,
    cross_params: str | Path | None = None,
    expected_pairs: int = 8,
    bytes_tolerance: float = 0.005,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if expected_pairs <= 0:
        raise ValueError("expected_pairs must be positive")
    if bytes_tolerance < 0:
        raise ValueError("bytes_tolerance must be non-negative")

    latent_positions, latent_pairs, latent_summary = _analyze_latents(
        bf16_root, temporal_root, cross_root
    )
    attention_rows, attention_summary = _analyze_attention(
        temporal_trace_root, cross_trace_root
    )
    parity_rows, parity_summary = _analyze_first_boundary_parity(
        temporal_root, cross_root
    )
    quality_rows, quality_summary = _analyze_quality(quality_runs)
    runtime_rows, runtime_summary = _analyze_runtime(
        runtime_roots or {}, cross_params=cross_params, bytes_tolerance=bytes_tolerance
    )
    manual_rows, manual_summary = _analyze_manual(failure_tags)

    pair_count_ok = latent_summary["pairs"] == expected_pairs
    gates = {
        "pair_completeness": {
            "status": "PASS" if pair_count_ok else "FAIL",
            "expected_pairs": expected_pairs,
            "observed_pairs": latent_summary["pairs"],
        },
        "first_boundary_k_parity": parity_summary,
        "v_reconstruction": attention_summary["gates"]["v_reconstruction"],
        "attention_output": attention_summary["gates"]["attention_output"],
        "latent_trajectory": latent_summary["gate"],
        "video_similarity": quality_summary["gate"],
        "actual_bytes": runtime_summary["gate"],
        "manual_review": manual_summary["gate"],
    }
    automatic_names = (
        "pair_completeness",
        "first_boundary_k_parity",
        "v_reconstruction",
        "attention_output",
        "latent_trajectory",
        "video_similarity",
        "actual_bytes",
    )
    automatic_statuses = [gates[name]["status"] for name in automatic_names]
    if "FAIL" in automatic_statuses:
        status = "FAIL_SMOKE_AUTOMATIC"
        mb32_authorized = False
    elif any(value != "PASS" for value in automatic_statuses):
        status = "INCOMPLETE_AUTOMATIC_EVIDENCE"
        mb32_authorized = False
    elif gates["manual_review"]["status"] == "FAIL":
        status = "FAIL_SMOKE_MANUAL"
        mb32_authorized = False
    elif gates["manual_review"]["status"] != "PASS":
        status = "PASS_AUTOMATIC_MANUAL_REVIEW_PENDING"
        mb32_authorized = False
    else:
        status = "PASS_SMOKE_MB32_AUTHORIZED"
        mb32_authorized = True

    summary = {
        "schema_version": 1,
        "status": status,
        "mb32_authorized": mb32_authorized,
        "scientific_scope": (
            "Smoke4 direction gate only. A pass authorizes a frozen MB32 experiment; "
            "it is not a publication-level significance result."
        ),
        "gates": gates,
        "latent": latent_summary,
        "attention": attention_summary,
        "quality": quality_summary,
        "runtime": runtime_summary,
        "manual_review": manual_summary,
    }
    tables = {
        "latent_position_rows": latent_positions,
        "latent_pair_rows": latent_pairs,
        "attention_rows": attention_rows,
        "parity_rows": parity_rows,
        "quality_rows": quality_rows,
        "runtime_rows": runtime_rows,
        "manual_rows": manual_rows,
    }
    return tables, summary


def write_cross_kv_smoke(
    output_dir: str | Path,
    tables: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
) -> None:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        if rows:
            _write_csv(output / f"{name}.csv", rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    decision = [
        "# Cross-KV Smoke Decision",
        "",
        f"- Status: **{summary['status']}**",
        f"- MB32 authorized: **{summary['mb32_authorized']}**",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    decision.extend(
        f"| {name} | {gate['status']} |" for name, gate in summary["gates"].items()
    )
    decision.extend(
        [
            "",
            "> A generated MP4, exit code 0, or automatic pass does not replace manual review.",
            "",
        ]
    )
    (output / "decision.md").write_text("\n".join(decision), encoding="utf-8")


def _analyze_latents(
    bf16_root: str | Path,
    temporal_root: str | Path,
    cross_root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups = {
        "bf16": load_rollouts(_latent_glob(bf16_root)),
        "temporal": load_rollouts(_latent_glob(temporal_root)),
        "cross": load_rollouts(_latent_glob(cross_root)),
    }
    key_sets = {name: set(records) for name, records in groups.items()}
    if len({frozenset(keys) for keys in key_sets.values()}) != 1:
        detail = {name: [list(key) for key in sorted(keys)] for name, keys in key_sets.items()}
        raise ValueError(f"Cross-KV rollout keys do not match: {detail}")

    position_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    by_key: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for key in sorted(key_sets["bf16"]):
        reference = groups["bf16"][key]
        for method in ("temporal", "cross"):
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
            row = {
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
            }
            pair_rows.append(row)
            by_key[key][method] = row

    ratios = [
        values["cross"]["mean_relative_l2"]
        / max(values["temporal"]["mean_relative_l2"], 1e-12)
        for values in by_key.values()
    ]
    wins = sum(value < 1.0 for value in ratios)
    seed_ratios = _seed_medians(
        [(key[1], values["cross"]["mean_relative_l2"] / max(values["temporal"]["mean_relative_l2"], 1e-12)) for key, values in by_key.items()]
    )
    required_wins = math.ceil(0.75 * len(ratios))
    passes = bool(
        ratios
        and np.median(ratios) < 1.0
        and wins >= required_wins
        and all(value < 1.0 for value in seed_ratios.values())
        and np.median([values["cross"]["max"] / max(values["temporal"]["max"], 1e-12) for values in by_key.values()]) <= 1.0
    )
    return position_rows, pair_rows, {
        "pairs": len(by_key),
        "cross_over_temporal_mean_relative_l2": {
            "pair_median": float(np.median(ratios)),
            "wins": wins,
            "required_wins": required_wins,
            "seed_medians": seed_ratios,
            "pair_values": ratios,
        },
        "gate": {"status": "PASS" if passes else "FAIL"},
    }


def _analyze_attention(
    temporal_root: str | Path,
    cross_root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    temporal = _load_attention_traces(Path(temporal_root))
    cross = _load_attention_traces(Path(cross_root))
    if set(temporal) != set(cross):
        raise ValueError(
            "attention trace keys do not match: "
            f"temporal_only={sorted(set(temporal) - set(cross))}, "
            f"cross_only={sorted(set(cross) - set(temporal))}"
        )
    rows: list[dict[str, Any]] = []
    lower_summaries: dict[str, Any] = {}
    higher_summaries: dict[str, Any] = {}
    for key in sorted(temporal):
        row: dict[str, Any] = {"trace_key": key, "seed": _seed_from_path(key)}
        for metric in LOWER_ATTENTION_METRICS:
            temporal_value = float(temporal[key][metric])
            cross_value = float(cross[key][metric])
            row[f"temporal_{metric}"] = temporal_value
            row[f"cross_{metric}"] = cross_value
            row[f"cross_over_temporal_{metric}"] = cross_value / max(temporal_value, 1e-12)
        for metric in HIGHER_ATTENTION_METRICS:
            temporal_value = float(temporal[key][metric])
            cross_value = float(cross[key][metric])
            row[f"temporal_{metric}"] = temporal_value
            row[f"cross_{metric}"] = cross_value
            row[f"cross_minus_temporal_{metric}"] = cross_value - temporal_value
        rows.append(row)

    for metric in LOWER_ATTENTION_METRICS:
        ratio_field = f"cross_over_temporal_{metric}"
        values = [float(row[ratio_field]) for row in rows]
        lower_summaries[metric] = {
            "pair_median_ratio": float(np.median(values)),
            "wins": sum(value < 1.0 for value in values),
            "pairs": len(values),
            "seed_medians": _seed_medians([(int(row["seed"]), float(row[ratio_field])) for row in rows]),
        }
    for metric in HIGHER_ATTENTION_METRICS:
        delta_field = f"cross_minus_temporal_{metric}"
        values = [float(row[delta_field]) for row in rows]
        higher_summaries[metric] = {
            "pair_median_delta": float(np.median(values)),
            "wins": sum(value > 0.0 for value in values),
            "pairs": len(values),
        }

    v_gate = _direction_gate(lower_summaries["v_cache_read_rel_l2"], max_median=0.90)
    attention_gate = _direction_gate(lower_summaries["attention_output_rel_l2"])
    return rows, {
        "records": len(rows),
        "lower_is_better": lower_summaries,
        "higher_is_better": higher_summaries,
        "gates": {
            "v_reconstruction": {"status": "PASS" if v_gate else "FAIL"},
            "attention_output": {"status": "PASS" if attention_gate else "FAIL"},
        },
    }


def _direction_gate(summary: dict[str, Any], *, max_median: float = 1.0) -> bool:
    required = math.ceil(0.75 * int(summary["pairs"]))
    return bool(
        summary["pairs"]
        and float(summary["pair_median_ratio"]) < max_median
        and int(summary["wins"]) >= required
        and all(float(value) < 1.0 for value in summary["seed_medians"].values())
    )


def _analyze_first_boundary_parity(
    temporal_root: str | Path,
    cross_root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    temporal = _snapshot_map(Path(temporal_root))
    cross = _snapshot_map(Path(cross_root))
    if not temporal or not cross:
        return [], {"status": "INCOMPLETE", "reason": "missing parity snapshots"}
    if set(temporal) != set(cross):
        return [], {"status": "FAIL", "reason": "parity snapshot keys differ"}
    rows = []
    passed = True
    for key in sorted(temporal):
        temporal_payload = torch.load(temporal[key], map_location="cpu", weights_only=False)
        cross_payload = torch.load(cross[key], map_location="cpu", weights_only=False)
        layer = int(temporal_payload["metadata"]["layer_idx"])
        temporal_layer = temporal_payload["layers"][layer]
        cross_layer = cross_payload["layers"][layer]
        raw_equal = torch.equal(temporal_layer["k"], cross_layer["k"])
        decoded_equal = torch.equal(
            temporal_layer["trq_decoded_k"], cross_layer["trq_decoded_k"]
        )
        passed = passed and raw_equal and decoded_equal
        rows.append(
            {
                "snapshot_key": key,
                "raw_k_equal": raw_equal,
                "decoded_k_equal": decoded_equal,
            }
        )
    return rows, {
        "status": "PASS" if passed else "FAIL",
        "records": len(rows),
        "scope": "First quantization boundary only; later K may diverge through the closed loop.",
    }


def _analyze_quality(
    runs: Iterable[tuple[str, int, str | Path]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, seed, path_text in runs:
        payload = _read_json(Path(path_text))
        for item in payload.get("per_video", []):
            rows.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "prompt_index": int(item["idx"]),
                    "sample_index": int(item.get("sample_idx", 0)),
                    "psnr": item.get("psnr"),
                    "ssim": item.get("ssim"),
                    "lpips": item.get("lpips"),
                }
            )
    if not rows:
        return [], {"gate": {"status": "INCOMPLETE"}, "reason": "missing quality metrics"}
    indexed = {
        (row["method"], row["seed"], row["prompt_index"], row["sample_index"]): row
        for row in rows
    }
    keys = sorted({key[1:] for key in indexed if key[0] == "temporal"})
    if not keys or any(("cross", *key) not in indexed for key in keys):
        return rows, {"gate": {"status": "FAIL"}, "reason": "quality pairs do not match"}
    metric_directions = {"psnr": 1.0, "ssim": 1.0, "lpips": -1.0}
    metrics: dict[str, Any] = {}
    complete = True
    no_consistent_worsening = True
    for metric, direction in metric_directions.items():
        values = []
        per_seed: dict[int, list[float]] = defaultdict(list)
        for key in keys:
            temporal_value = indexed[("temporal", *key)][metric]
            cross_value = indexed[("cross", *key)][metric]
            if temporal_value is None or cross_value is None:
                complete = False
                continue
            improvement = direction * (float(cross_value) - float(temporal_value))
            values.append(improvement)
            per_seed[int(key[0])].append(improvement)
        seed_medians = {
            str(seed): float(np.median(seed_values))
            for seed, seed_values in sorted(per_seed.items())
        }
        if len(seed_medians) < 2 or all(value < 0.0 for value in seed_medians.values()):
            no_consistent_worsening = False
        metrics[metric] = {
            "cross_minus_temporal_improvement_median": (
                float(np.median(values)) if values else None
            ),
            "wins": sum(value > 0.0 for value in values),
            "pairs": len(values),
            "seed_medians": seed_medians,
        }
    if not complete:
        status = "INCOMPLETE"
    else:
        status = "PASS" if no_consistent_worsening else "FAIL"
    return rows, {"metrics": metrics, "gate": {"status": status}}


def _analyze_runtime(
    roots: dict[str, str | Path],
    *,
    cross_params: str | Path | None,
    bytes_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not roots:
        return [], {"gate": {"status": "INCOMPLETE"}, "reason": "missing runtime roots"}
    predictor_bytes = _predictor_nbytes(cross_params)
    rows = []
    for method, root_text in roots.items():
        root = Path(root_text).expanduser()
        for path in sorted(root.rglob("rollout_metrics/runtime/*.json")):
            payload = _read_json(path)
            pipeline = payload.get("pipeline", {})
            cache = pipeline.get("kv_cache", {}).get("total", {})
            memory = pipeline.get("memory", {})
            cache_bytes = cache.get("physical_bytes")
            extra = predictor_bytes if method == "cross" else 0
            rows.append(
                {
                    "method": method,
                    "seed": payload.get("metadata", {}).get("seed"),
                    "prompt_index": payload.get("metadata", {}).get("prompt_index"),
                    "wall_time_e2e_ms": payload.get("wall_time_e2e_ms"),
                    "diffusion_ms": pipeline.get("stages_ms", {}).get("diffusion"),
                    "kv_physical_bytes": cache_bytes,
                    "predictor_parameter_bytes": extra,
                    "actual_bytes_with_predictor": (
                        None if cache_bytes is None else int(cache_bytes) + extra
                    ),
                    "max_allocated_bytes": memory.get("max_allocated_bytes"),
                    "max_reserved_bytes": memory.get("max_reserved_bytes"),
                    "source": str(path),
                }
            )
    by_method = {method: [row for row in rows if row["method"] == method] for method in roots}
    if not by_method.get("temporal") or not by_method.get("cross"):
        return rows, {"gate": {"status": "INCOMPLETE"}, "reason": "missing method runtime"}
    temporal_bytes = _median_field(by_method["temporal"], "actual_bytes_with_predictor")
    cross_bytes = _median_field(by_method["cross"], "actual_bytes_with_predictor")
    if temporal_bytes is None or cross_bytes is None:
        status = "INCOMPLETE"
        relative_difference = None
    else:
        relative_difference = abs(cross_bytes - temporal_bytes) / max(temporal_bytes, 1.0)
        status = "PASS" if relative_difference <= bytes_tolerance else "FAIL"
    return rows, {
        "predictor_parameter_bytes": predictor_bytes,
        "methods": {
            method: {
                "runs": len(method_rows),
                "median_actual_bytes_with_predictor": _median_field(
                    method_rows, "actual_bytes_with_predictor"
                ),
                "median_wall_time_e2e_ms": _median_field(method_rows, "wall_time_e2e_ms"),
                "median_max_allocated_bytes": _median_field(method_rows, "max_allocated_bytes"),
                "median_max_reserved_bytes": _median_field(method_rows, "max_reserved_bytes"),
            }
            for method, method_rows in by_method.items()
        },
        "gate": {
            "status": status,
            "relative_byte_difference": relative_difference,
            "tolerance": bytes_tolerance,
        },
    }


def _analyze_manual(
    failure_tags: str | Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if failure_tags is None or not Path(failure_tags).expanduser().is_file():
        return [], {"gate": {"status": "INCOMPLETE"}, "reason": "failure tags missing"}
    with Path(failure_tags).expanduser().open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return [], {"gate": {"status": "INCOMPLETE"}, "reason": "failure tags empty"}
    missing_fields = [field for field in MANUAL_FIELDS if field not in rows[0]]
    if missing_fields:
        return rows, {
            "gate": {"status": "INCOMPLETE"},
            "reason": f"missing manual columns: {missing_fields}",
        }
    complete = all(str(row.get(field, "")).strip() for row in rows for field in MANUAL_FIELDS)
    if not complete:
        return rows, {"gate": {"status": "INCOMPLETE"}, "reason": "manual cells remain blank"}
    indexed = {
        (
            row["config"],
            int(row["prompt_index"]),
            int(row["seed"]),
            int(row.get("sample_index", 0)),
        ): row
        for row in rows
    }
    new_cross_catastrophes = []
    for key, row in indexed.items():
        if key[0] != "cross" or not _truthy(row.get("catastrophe")):
            continue
        temporal = indexed.get(("temporal", *key[1:]))
        if temporal is not None and not _truthy(temporal.get("catastrophe")):
            new_cross_catastrophes.append(list(key[1:]))
    return rows, {
        "rows": len(rows),
        "new_cross_catastrophes": new_cross_catastrophes,
        "gate": {"status": "FAIL" if new_cross_catastrophes else "PASS"},
    }


def _load_attention_traces(root: Path) -> dict[str, dict[str, float]]:
    root = root.expanduser().resolve()
    result = {}
    for path in sorted(root.rglob("boundary*_layer*.json")):
        payload = _read_json(path)
        metrics = payload.get("metrics", {})
        required = set(LOWER_ATTENTION_METRICS) | set(HIGHER_ATTENTION_METRICS)
        missing = sorted(required - set(metrics))
        if missing:
            raise ValueError(f"attention trace lacks {missing}: {path}")
        values = {name: float(metrics[name]) for name in required}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"non-finite attention trace: {path}")
        result[str(path.relative_to(root))] = values
    if not result:
        raise FileNotFoundError(f"no attention traces found under {root}")
    return result


def _latent_glob(root: str | Path) -> str:
    path = Path(root).expanduser()
    return str(path / "**" / "rollout_metrics" / "latents" / "*.pt")


def _snapshot_map(root: Path) -> dict[str, Path]:
    root = root.expanduser().resolve()
    return {
        str(path.relative_to(root)): path
        for path in sorted(root.rglob("parity_snapshots/first_event_rank*_layer*.pt"))
    }


def _seed_from_path(value: str) -> int:
    for part in Path(value).parts:
        match = re.fullmatch(r"s(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"cannot infer seed from trace path: {value}")


def _seed_medians(values: Iterable[tuple[int, float]]) -> dict[str, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for seed, value in values:
        grouped[int(seed)].append(float(value))
    return {
        str(seed): float(np.median(seed_values))
        for seed, seed_values in sorted(grouped.items())
    }


def _predictor_nbytes(path_text: str | Path | None) -> int:
    if path_text is None:
        return 0
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Cross-KV parameters do not exist: {path}")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            return int(
                sum(
                    payload[name].nbytes
                    for name in ("cross_weight", "cross_bias")
                    if name in payload
                )
            )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(
        sum(
            value.numel() * value.element_size()
            for name, value in payload.items()
            if name in {"cross_weight", "cross_bias"} and isinstance(value, torch.Tensor)
        )
    )


def _median_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.median(values) if values else None


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
