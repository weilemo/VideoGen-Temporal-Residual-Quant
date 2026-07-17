"""Paired BF16-repeat versus TRQ online rollout analysis."""

from __future__ import annotations

import csv
import glob
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


RolloutKey = tuple[int, int, int]


@dataclass(frozen=True)
class RolloutRecord:
    latents: torch.Tensor
    metadata: dict[str, Any]
    path: Path


def analyze_paired_rollouts(
    bf16_a: str | Path,
    bf16_b: str | Path,
    trq: str | Path,
    *,
    bootstrap_resamples: int = 2000,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups = {
        "bf16_a": load_rollouts(bf16_a),
        "bf16_b": load_rollouts(bf16_b),
        "trq": load_rollouts(trq),
    }
    key_sets = {name: set(items) for name, items in groups.items()}
    if len({frozenset(keys) for keys in key_sets.values()}) != 1:
        detail = {name: [list(key) for key in sorted(keys)] for name, keys in key_sets.items()}
        raise ValueError(f"paired rollout keys do not match: {detail}")

    position_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for key in sorted(key_sets["bf16_a"]):
        a_record = groups["bf16_a"][key]
        b_record = groups["bf16_b"][key]
        q_record = groups["trq"][key]
        validate_paired_metadata(key, a_record, b_record, q_record)
        a = a_record.latents
        b = b_record.latents
        q = q_record.latents
        if a.shape != b.shape or a.shape != q.shape:
            raise ValueError(
                f"latent shape mismatch for {key}: {tuple(a.shape)}, {tuple(b.shape)}, {tuple(q.shape)}"
            )
        if a.ndim < 2 or a.shape[0] < 2:
            raise ValueError(f"rollout {key} must have at least two positions, got {tuple(a.shape)}")

        noise_rel, noise_cos = paired_distances(a, b)
        trq_rel, trq_cos = paired_distances(a, q)
        normalized_time = np.linspace(0.0, 1.0, num=a.shape[0], dtype=np.float64)
        excess_rel = trq_rel - noise_rel
        excess_cos = trq_cos - noise_cos
        for position in range(a.shape[0]):
            position_rows.append({
                "prompt_index": key[0],
                "seed": key[1],
                "sample_index": key[2],
                "position": position,
                "normalized_time": float(normalized_time[position]),
                "bf16_repeat_rel_l2": float(noise_rel[position]),
                "trq_rel_l2": float(trq_rel[position]),
                "trq_excess_rel_l2": float(excess_rel[position]),
                "bf16_repeat_cosine_distance": float(noise_cos[position]),
                "trq_cosine_distance": float(trq_cos[position]),
                "trq_excess_cosine_distance": float(excess_cos[position]),
            })

        early_count = max(1, int(math.ceil(a.shape[0] * 0.2)))
        noise_growth = float(np.median(noise_rel[-early_count:]) - np.median(noise_rel[:early_count]))
        trq_growth = float(np.median(trq_rel[-early_count:]) - np.median(trq_rel[:early_count]))
        noise_slope = theil_sen_slope(noise_rel)
        trq_slope = theil_sen_slope(trq_rel)
        pair_rows.append({
            "prompt_index": key[0],
            "seed": key[1],
            "sample_index": key[2],
            "positions": int(a.shape[0]),
            "bf16_repeat_growth": noise_growth,
            "trq_growth": trq_growth,
            "trq_excess_growth": trq_growth - noise_growth,
            "bf16_repeat_slope": noise_slope,
            "trq_slope": trq_slope,
            "trq_excess_slope": trq_slope - noise_slope,
            "bf16_repeat_final_rel_l2": float(noise_rel[-1]),
            "trq_final_rel_l2": float(trq_rel[-1]),
            "trq_excess_final_rel_l2": float(excess_rel[-1]),
        })

    growth_values = prompt_level_values(pair_rows, "trq_excess_growth")
    noise_growth_values = prompt_level_values(pair_rows, "bf16_repeat_growth")
    slope_values = prompt_level_values(pair_rows, "trq_excess_slope")
    noise_slope_values = prompt_level_values(pair_rows, "bf16_repeat_slope")
    growth_ci = bootstrap_median_ci(growth_values, bootstrap_resamples, seed)
    slope_ci = bootstrap_median_ci(slope_values, bootstrap_resamples, seed + 1)
    growth_margin = max(0.02, float(np.quantile(np.abs(noise_growth_values), 0.95)))
    slope_margin = max(0.02, float(np.quantile(np.abs(noise_slope_values), 0.95)))
    summary = {
        "schema_version": 1,
        "pairs": len(pair_rows),
        "prompts": len({row["prompt_index"] for row in pair_rows}),
        "bootstrap_resamples": int(bootstrap_resamples),
        "trq_excess_growth": {
            "prompt_median": float(np.median(growth_values)),
            "bootstrap_95_ci": growth_ci,
            "noise_aware_margin": growth_margin,
            "passes_no_progressive_drift_gate": bool(growth_ci[1] <= growth_margin),
        },
        "trq_excess_slope": {
            "prompt_median": float(np.median(slope_values)),
            "bootstrap_95_ci": slope_ci,
            "noise_aware_margin": slope_margin,
            "passes_no_progressive_drift_gate": bool(slope_ci[1] <= slope_margin),
        },
    }
    summary["passes_latent_drift_gate"] = bool(
        summary["trq_excess_growth"]["passes_no_progressive_drift_gate"]
        and summary["trq_excess_slope"]["passes_no_progressive_drift_gate"]
    )
    return position_rows, pair_rows, summary


def load_rollouts(source: str | Path) -> dict[RolloutKey, RolloutRecord]:
    paths = resolve_paths(source)
    if not paths:
        raise FileNotFoundError(f"no rollout latent files matched {source}")
    result: dict[RolloutKey, RolloutRecord] = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "latents" not in payload:
            raise ValueError(f"invalid rollout payload: {path}")
        metadata = payload.get("metadata", {})
        key = (
            int(metadata["prompt_index"]),
            int(metadata["seed"]),
            int(metadata.get("sample_index", 0)),
        )
        if key in result:
            raise ValueError(f"duplicate rollout key {key}: {path}")
        result[key] = RolloutRecord(
            latents=torch.as_tensor(payload["latents"]).float().contiguous(),
            metadata=dict(metadata),
            path=path,
        )
    return result


def validate_paired_metadata(
    key: RolloutKey,
    bf16_a: RolloutRecord,
    bf16_b: RolloutRecord,
    trq: RolloutRecord,
) -> None:
    required_equal = (
        "prompt",
        "num_output_frames",
        "local_attn_size",
        "effective_seed",
        "sample_index",
    )
    for field in required_equal:
        values = [record.metadata.get(field) for record in (bf16_a, bf16_b, trq)]
        if any(value is None for value in values):
            # Older synthetic/unit-test payloads may not contain the extended
            # metadata. They remain readable, but any partially present field
            # is rejected because that indicates a malformed real run.
            if not all(value is None for value in values):
                raise ValueError(f"paired rollout {key} has incomplete metadata field {field}: {values}")
            continue
        if len({json.dumps(value, sort_keys=True) for value in values}) != 1:
            raise ValueError(f"paired rollout {key} metadata mismatch for {field}: {values}")


def resolve_paths(source: str | Path) -> list[Path]:
    result: list[Path] = []
    for token in str(source).split(","):
        token = token.strip()
        if not token:
            continue
        path = Path(token).expanduser()
        if path.is_dir():
            matches = sorted(path.rglob("*.pt"))
        elif any(character in token for character in "*?["):
            matches = [Path(item) for item in sorted(glob.glob(str(path), recursive=True))]
        elif path.is_file():
            matches = [path]
        else:
            matches = []
        result.extend(item.resolve() for item in matches)
    return list(dict.fromkeys(result))


def paired_distances(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    ref = reference.reshape(reference.shape[0], -1).double()
    other = candidate.reshape(candidate.shape[0], -1).double()
    ref_norm = torch.linalg.vector_norm(ref, dim=1).clamp_min(1e-12)
    other_norm = torch.linalg.vector_norm(other, dim=1).clamp_min(1e-12)
    rel_l2 = torch.linalg.vector_norm(ref - other, dim=1) / ref_norm
    cosine = 1.0 - torch.sum(ref * other, dim=1) / (ref_norm * other_norm)
    return rel_l2.numpy(), cosine.numpy()


def theil_sen_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    x = np.linspace(0.0, 1.0, num=values.size, dtype=np.float64)
    slopes = []
    for index in range(values.size - 1):
        slopes.extend(((values[index + 1:] - values[index]) / (x[index + 1:] - x[index])).tolist())
    return float(np.median(np.asarray(slopes, dtype=np.float64)))


def prompt_level_values(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["prompt_index"])].append(float(row[field]))
    return np.asarray([np.median(values) for _, values in sorted(grouped.items())], dtype=np.float64)


def bootstrap_median_ci(values: np.ndarray, resamples: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return [float(values[0]), float(values[0])]
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(resamples, values.size), replace=True)
    medians = np.median(samples, axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def write_analysis(
    output_dir: str | Path,
    position_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "position_rows.csv", position_rows)
    write_csv(output / "pair_summary.csv", pair_rows)
    with open(output / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    growth = summary["trq_excess_growth"]
    slope = summary["trq_excess_slope"]
    report = [
        "# Online Paired Rollout Analysis",
        "",
        f"- pairs: `{summary['pairs']}`",
        f"- prompts: `{summary['prompts']}`",
        f"- latent drift gate: `{'PASS' if summary['passes_latent_drift_gate'] else 'FAIL'}`",
        f"- excess growth median / 95% CI: `{growth['prompt_median']:.6f}` / `{growth['bootstrap_95_ci']}`",
        f"- excess growth margin: `{growth['noise_aware_margin']:.6f}`",
        f"- excess slope median / 95% CI: `{slope['prompt_median']:.6f}` / `{slope['bootstrap_95_ci']}`",
        f"- excess slope margin: `{slope['noise_aware_margin']:.6f}`",
        "",
        "This gate covers latent trajectory drift only. VBench and qualitative late-frame checks remain required.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
