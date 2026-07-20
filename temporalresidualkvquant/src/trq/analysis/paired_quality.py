"""Prompt-paired VBench analysis and quality gating for online TRQ runs."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "overall_consistency",
    "clip_score",
)

NORMALIZE = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "motion_smoothness": (0.706, 0.9975),
    "dynamic_degree": (0.0, 1.0),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
    "overall_consistency": (0.0, 0.364),
    "clip_score": (0.0, 0.3557),
}

DIMENSION_WEIGHTS = {
    "subject_consistency": 1.0,
    "background_consistency": 1.0,
    "motion_smoothness": 1.0,
    "dynamic_degree": 0.5,
    "aesthetic_quality": 1.0,
    "imaging_quality": 1.0,
    "overall_consistency": 1.0,
    "clip_score": 1.0,
}

LATENT_FIELDS = (
    "early_median",
    "late_median",
    "growth",
    "final",
    "p95",
    "max",
    "theil_sen_slope",
    "boundary_jump",
)

FAILURE_TAG_FIELDS = (
    "identity_switch",
    "background_jump",
    "texture_repetition",
    "motion_freeze",
    "color_drift",
    "black_or_nan",
)

_VIDEO_KEY_PATTERNS = (
    re.compile(r"(?:^|/)(\d+)-(\d+)_(?:ema|regular)(?:\.mp4)?(?:/|$)"),
    re.compile(r"(?:^|/)(\d+)_(?:ema|regular)-(\d+)(?:\.mp4)?(?:/|$)"),
)


@dataclass(frozen=True)
class VBenchRun:
    config: str
    seed: int
    label: str
    input_dir: Path


def load_vbench_run(run: VBenchRun) -> dict[tuple[int, int], dict[str, float]]:
    grouped: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for dimension in DIMENSIONS:
        path = run.input_dir / f"{run.label}_{dimension}_eval_results.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get(dimension)
        if not isinstance(value, list) or len(value) < 2 or not isinstance(value[1], list):
            raise ValueError(f"{path} does not contain per-video VBench results")
        per_video: dict[tuple[int, int], list[float]] = defaultdict(list)
        for detail in value[1]:
            if not isinstance(detail, dict) or "video_path" not in detail or "video_results" not in detail:
                continue
            normalized_path = str(detail["video_path"]).replace("\\", "/")
            match = None
            for pattern in _VIDEO_KEY_PATTERNS:
                match = pattern.search(normalized_path)
                if match is not None:
                    break
            if match is None:
                raise ValueError(f"cannot recover prompt/sample identity from {detail['video_path']!r}")
            prompt_index, sample_index = (int(item) for item in match.groups())
            score = detail["video_results"]
            if not isinstance(score, (bool, int, float)):
                raise ValueError(f"non-scalar VBench result in {path}: {score!r}")
            per_video[(prompt_index, sample_index)].append(float(score))
        if not per_video:
            raise ValueError(f"{path} contains no usable per-video scores")
        for key, scores in per_video.items():
            grouped[key][dimension] = float(np.mean(scores))

    incomplete = {key: sorted(set(DIMENSIONS) - set(scores)) for key, scores in grouped.items()}
    incomplete = {key: missing for key, missing in incomplete.items() if missing}
    if incomplete:
        raise ValueError(f"incomplete VBench dimensions for {run.config}/seed{run.seed}: {incomplete}")
    return dict(grouped)


def analyze_paired_quality(
    runs: Iterable[VBenchRun],
    *,
    latent_csvs: dict[str, Path] | None = None,
    failure_tags_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    loaded: dict[tuple[str, int, int, int], dict[str, float]] = {}
    configs: set[str] = set()
    for run in runs:
        configs.add(run.config)
        scores = load_vbench_run(run)
        for (prompt_index, sample_index), values in scores.items():
            key = (run.config, int(run.seed), prompt_index, sample_index)
            if key in loaded:
                raise ValueError(f"duplicate VBench run key: {key}")
            loaded[key] = values
    if not {"bf16_a", "bf16_b"}.issubset(configs):
        raise ValueError("paired quality analysis requires bf16_a and bf16_b runs")

    candidates = sorted(configs - {"bf16_a", "bf16_b"})
    if not candidates:
        raise ValueError("paired quality analysis requires at least one candidate config")
    latent = load_latent_metrics(latent_csvs or {})
    failure_tags = load_failure_tags(failure_tags_path)
    rows: list[dict[str, Any]] = []

    reference_keys = {
        (seed, prompt, sample)
        for config, seed, prompt, sample in loaded
        if config == "bf16_a"
    }
    repeat_keys = {
        (seed, prompt, sample)
        for config, seed, prompt, sample in loaded
        if config == "bf16_b"
    }
    if reference_keys != repeat_keys:
        raise ValueError("BF16-A/B VBench keys do not match")

    for config in candidates:
        candidate_keys = {
            (seed, prompt, sample)
            for item_config, seed, prompt, sample in loaded
            if item_config == config
        }
        if candidate_keys != reference_keys:
            raise ValueError(
                f"candidate VBench keys do not match BF16 for {config}: "
                f"candidate={sorted(candidate_keys)}, reference={sorted(reference_keys)}"
            )
        for seed, prompt_index, sample_index in sorted(reference_keys):
            a = loaded[("bf16_a", seed, prompt_index, sample_index)]
            b = loaded[("bf16_b", seed, prompt_index, sample_index)]
            candidate = loaded[(config, seed, prompt_index, sample_index)]
            reference = {dimension: float(np.median([a[dimension], b[dimension]])) for dimension in DIMENSIONS}
            row: dict[str, Any] = {
                "config": config,
                "prompt_index": prompt_index,
                "seed": seed,
                "sample_index": sample_index,
                "bf16_final": aggregate_final_score(reference),
                "candidate_final": aggregate_final_score(candidate),
            }
            row["final_delta"] = row["candidate_final"] - row["bf16_final"]
            for dimension in DIMENSIONS:
                row[f"bf16_{dimension}"] = reference[dimension]
                row[f"candidate_{dimension}"] = candidate[dimension]
                row[f"delta_{dimension}"] = candidate[dimension] - reference[dimension]
            latent_row = latent.get((config, prompt_index, seed, sample_index), {})
            for field in LATENT_FIELDS:
                row[field] = latent_row.get(field)
            rows.append(row)

    correlations = correlation_rows(rows)
    gates: dict[str, Any] = {}
    for config in candidates:
        config_rows = [row for row in rows if row["config"] == config]
        final_delta = float(np.mean([row["final_delta"] for row in config_rows]))
        critical = {
            dimension: float(np.mean([row[f"delta_{dimension}"] for row in config_rows]))
            for dimension in ("subject_consistency", "background_consistency", "motion_smoothness")
        }
        expected_tag_keys = {
            (config, int(row["prompt_index"]), int(row["seed"]), int(row["sample_index"]))
            for row in config_rows
        }
        present_tag_keys = {
            key for key in expected_tag_keys
            if key in failure_tags and failure_tags[key]["complete"]
        }
        catastrophe_count = sum(
            1
            for key in expected_tag_keys
            if key in failure_tags
            and failure_tags[key]["complete"]
            and failure_tags[key]["catastrophe"]
            and not failure_tags[key]["bf16_present"]
        )
        quantitative_pass = final_delta >= -0.005 and all(value >= -0.01 for value in critical.values())
        qualitative_complete = present_tag_keys == expected_tag_keys
        catastrophe_pass = catastrophe_count < 2
        if not quantitative_pass or not catastrophe_pass:
            status = "FAIL"
        elif not qualitative_complete:
            status = "INCOMPLETE"
        else:
            status = "PASS"
        gates[config] = {
            "status": status,
            "pairs": len(config_rows),
            "paired_final_delta_mean": final_delta,
            "critical_dimension_delta_mean": critical,
            "passes_final_delta": final_delta >= -0.005,
            "passes_critical_dimensions": all(value >= -0.01 for value in critical.values()),
            "qualitative_complete": qualitative_complete,
            "qualitative_rows": len(present_tag_keys),
            "trq_only_catastrophes": catastrophe_count,
            "passes_catastrophe_gate": catastrophe_pass,
        }

    summary = {
        "schema_version": 1,
        "quality_gate_policy": {
            "minimum_final_delta": -0.005,
            "minimum_critical_dimension_delta": -0.01,
            "maximum_repeated_trq_only_catastrophes": 1,
            "qualitative_tags_required": True,
        },
        "configs": gates,
        "passing_configs": sorted(config for config, gate in gates.items() if gate["status"] == "PASS"),
    }
    return rows, correlations, summary


def aggregate_final_score(raw: dict[str, float]) -> float:
    normalized = {
        dimension: max(
            0.0,
            min(1.0, (float(raw[dimension]) - NORMALIZE[dimension][0]) / (NORMALIZE[dimension][1] - NORMALIZE[dimension][0])),
        )
        for dimension in DIMENSIONS
    }

    def weighted(dimensions: tuple[str, ...]) -> float:
        numerator = sum(normalized[item] * DIMENSION_WEIGHTS[item] for item in dimensions)
        denominator = sum(DIMENSION_WEIGHTS[item] for item in dimensions)
        return numerator / denominator

    temporal = weighted(("subject_consistency", "background_consistency", "motion_smoothness", "dynamic_degree"))
    frame = weighted(("aesthetic_quality", "imaging_quality"))
    text = weighted(("overall_consistency", "clip_score"))
    return float((2.0 * temporal + 2.0 * frame + text) / 5.0)


def load_latent_metrics(paths: dict[str, Path]) -> dict[tuple[str, int, int, int], dict[str, float]]:
    result: dict[tuple[str, int, int, int], dict[str, float]] = {}
    for config, path in paths.items():
        with Path(path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    config,
                    int(row["prompt_index"]),
                    int(row["seed"]),
                    int(row.get("sample_index", 0)),
                )
                if key in result:
                    raise ValueError(f"duplicate latent metric key: {key}")
                result[key] = {
                    field: float(row[field])
                    for field in LATENT_FIELDS
                    if row.get(field) not in (None, "")
                }
    return result


def load_failure_tags(path: str | Path | None) -> dict[tuple[str, int, int, int], dict[str, bool | None]]:
    if path is None:
        return {}
    result = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["config"],
                int(row["prompt_index"]),
                int(row["seed"]),
                int(row.get("sample_index", 0)),
            )
            catastrophe = parse_optional_bool(row.get("catastrophe", ""))
            bf16_present = parse_optional_bool(row.get("bf16_present", ""))
            failure_tags = {
                field: parse_optional_bool(row.get(field, ""))
                for field in FAILURE_TAG_FIELDS
            }
            result[key] = {
                "catastrophe": catastrophe,
                "bf16_present": bf16_present,
                "complete": (
                    catastrophe is not None
                    and bf16_present is not None
                    and all(value is not None for value in failure_tags.values())
                ),
                **failure_tags,
            }
    return result


def parse_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def parse_optional_bool(value: Any) -> bool | None:
    if str(value).strip() == "":
        return None
    return parse_bool(value)


def correlation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    quality_fields = ("final_delta",) + tuple(f"delta_{dimension}" for dimension in DIMENSIONS)
    for config in sorted({row["config"] for row in rows}):
        config_rows = [row for row in rows if row["config"] == config]
        for latent_field in LATENT_FIELDS:
            for quality_field in quality_fields:
                pairs = [
                    (float(row[latent_field]), float(row[quality_field]))
                    for row in config_rows
                    if row.get(latent_field) is not None
                ]
                rho = spearman(pairs)
                result.append({
                    "config": config,
                    "latent_metric": latent_field,
                    "quality_metric": quality_field,
                    "pairs": len(pairs),
                    "spearman_rho": rho,
                })
    return result


def spearman(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    x = np.asarray([item[0] for item in pairs], dtype=np.float64)
    y = np.asarray([item[1] for item in pairs], dtype=np.float64)
    x_rank = rankdata(x)
    y_rank = rankdata(y)
    if np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def write_quality_analysis(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "paired_vbench.csv", rows)
    _write_csv(output / "latent_quality_spearman.csv", correlations)
    (output / "quality_gate.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = ["# Paired VBench Quality Gate", ""]
    for config, gate in summary["configs"].items():
        report.extend([
            f"## {config}",
            "",
            f"- status: `{gate['status']}`",
            f"- paired final delta mean: `{gate['paired_final_delta_mean']:.6f}`",
            f"- qualitative rows: `{gate['qualitative_rows']}/{gate['pairs']}`",
            f"- TRQ-only catastrophes: `{gate['trq_only_catastrophes']}`",
            "",
        ])
    (output / "quality_gate.md").write_text("\n".join(report), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
