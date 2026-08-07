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
    first_quant_frame: int | None = None,
    boundary_before: int = 6,
    boundary_after: int = 6,
    config_label: str = "trq",
    quant_interval_frames: int = 24,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if boundary_before <= 0 or boundary_after <= 0:
        raise ValueError("boundary windows must be positive")
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
        noise_early = float(np.median(noise_rel[:early_count]))
        noise_late = float(np.median(noise_rel[-early_count:]))
        trq_early = float(np.median(trq_rel[:early_count]))
        trq_late = float(np.median(trq_rel[-early_count:]))
        excess_early = float(np.median(excess_rel[:early_count]))
        excess_late = float(np.median(excess_rel[-early_count:]))
        noise_growth = noise_late - noise_early
        trq_growth = trq_late - trq_early
        noise_slope = theil_sen_slope(noise_rel)
        trq_slope = theil_sen_slope(trq_rel)
        excess_slope = theil_sen_slope(excess_rel)
        event_frames = quantization_frames(
            q_record.metadata,
            fallback=first_quant_frame,
            interval=quant_interval_frames,
            positions=int(a.shape[0]),
        )
        boundary_frame = min(event_frames) if event_frames else None
        boundary_jump = None
        boundary_pre_median = None
        boundary_post_median = None
        if boundary_frame is not None:
            if boundary_frame - boundary_before < 0 or boundary_frame + boundary_after > a.shape[0]:
                raise ValueError(
                    f"boundary window [{boundary_frame - boundary_before}, "
                    f"{boundary_frame + boundary_after}) is outside rollout {key} with {a.shape[0]} positions"
                )
            boundary_pre_median = float(
                np.median(excess_rel[boundary_frame - boundary_before:boundary_frame])
            )
            boundary_post_median = float(
                np.median(excess_rel[boundary_frame:boundary_frame + boundary_after])
            )
            boundary_jump = boundary_post_median - boundary_pre_median
        pair_rows.append({
            "config": config_label,
            "prompt_index": key[0],
            "seed": key[1],
            "sample_index": key[2],
            "positions": int(a.shape[0]),
            "bf16_repeat_growth": noise_growth,
            "trq_growth": trq_growth,
            "early_median": excess_early,
            "late_median": excess_late,
            "growth": excess_late - excess_early,
            "final": float(excess_rel[-1]),
            "p95": float(np.quantile(excess_rel, 0.95)),
            "max": float(np.max(excess_rel)),
            "theil_sen_slope": excess_slope,
            "trq_excess_growth": excess_late - excess_early,
            "bf16_repeat_slope": noise_slope,
            "trq_slope": trq_slope,
            "trq_excess_slope": excess_slope,
            "bf16_repeat_final_rel_l2": float(noise_rel[-1]),
            "trq_final_rel_l2": float(trq_rel[-1]),
            "trq_excess_final_rel_l2": float(excess_rel[-1]),
            "first_quant_frame": boundary_frame,
            "quantization_frames": ";".join(str(value) for value in event_frames),
            "boundary_before": int(boundary_before) if boundary_frame is not None else None,
            "boundary_after": int(boundary_after) if boundary_frame is not None else None,
            "boundary_pre_median": boundary_pre_median,
            "boundary_post_median": boundary_post_median,
            "boundary_jump": boundary_jump,
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
        "schema_version": 2,
        "config": config_label,
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
        "absolute_metrics": {
            field: summarize_prompt_metric(pair_rows, field, bootstrap_resamples, seed + 10 + index)
            for index, field in enumerate(
                ("early_median", "late_median", "growth", "final", "p95", "max", "theil_sen_slope")
            )
        },
    }
    boundary_values = [
        float(row["boundary_jump"])
        for row in pair_rows
        if row["boundary_jump"] is not None
    ]
    summary["boundary_jump"] = (
        summarize_prompt_metric(pair_rows, "boundary_jump", bootstrap_resamples, seed + 30)
        if boundary_values
        else None
    )
    summary["passes_latent_drift_gate"] = bool(
        summary["trq_excess_growth"]["passes_no_progressive_drift_gate"]
        and summary["trq_excess_slope"]["passes_no_progressive_drift_gate"]
    )
    return position_rows, pair_rows, summary


def quantization_frames(
    metadata: dict[str, Any],
    *,
    fallback: int | None,
    interval: int,
    positions: int | None,
) -> list[int]:
    events = metadata.get("quantization_events")
    if isinstance(events, list):
        boundaries = [
            int(event["boundary_frame"])
            for event in events
            if isinstance(event, dict) and event.get("boundary_frame") is not None
        ]
        if boundaries:
            return sorted(set(boundaries))
    if fallback is None:
        return []
    if interval <= 0:
        raise ValueError("quant_interval_frames must be positive")
    if positions is None:
        return [int(fallback)]
    return list(range(int(fallback), int(positions), int(interval)))


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
        if row.get(field) is not None:
            grouped[int(row["prompt_index"])].append(float(row[field]))
    return np.asarray([np.median(values) for _, values in sorted(grouped.items())], dtype=np.float64)


def summarize_prompt_metric(
    rows: list[dict[str, Any]],
    field: str,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    values = prompt_level_values(rows, field)
    if values.size == 0:
        raise ValueError(f"metric {field} has no prompt-level values")
    return {
        "prompt_median": float(np.median(values)),
        "bootstrap_95_ci": bootstrap_median_ci(values, bootstrap_resamples, seed),
    }


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
    write_csv(output / "online_absolute_metrics.csv", pair_rows)
    boundary_rows = [
        {
            key: row[key]
            for key in (
                "config",
                "prompt_index",
                "seed",
                "sample_index",
                "first_quant_frame",
                "boundary_before",
                "boundary_after",
                "boundary_pre_median",
                "boundary_post_median",
                "boundary_jump",
            )
        }
        for row in pair_rows
        if row["first_quant_frame"] is not None
    ]
    if boundary_rows:
        write_csv(output / "online_boundary_jump.csv", boundary_rows)
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
    write_plots(output, position_rows, pair_rows)


def write_plots(
    output: Path,
    position_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for online rollout plots; install the analysis extra"
        ) from exc

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in position_rows:
        grouped[int(row["position"])].append(row)
    positions = np.asarray(sorted(grouped), dtype=np.int64)
    trq_values = [np.asarray([item["trq_rel_l2"] for item in grouped[pos]]) for pos in positions]
    noise_values = [np.asarray([item["bf16_repeat_rel_l2"] for item in grouped[pos]]) for pos in positions]
    excess_values = [np.asarray([item["trq_excess_rel_l2"] for item in grouped[pos]]) for pos in positions]

    fig, axis = plt.subplots(figsize=(10, 5))
    for label, values, color in (
        ("TRQ vs BF16", trq_values, "#c23b22"),
        ("BF16 repeat", noise_values, "#386cb0"),
        ("TRQ excess", excess_values, "#2f7d32"),
    ):
        median = np.asarray([np.median(value) for value in values])
        low = np.asarray([np.quantile(value, 0.25) for value in values])
        high = np.asarray([np.quantile(value, 0.75) for value in values])
        axis.plot(positions, median, label=label, color=color, linewidth=1.8)
        axis.fill_between(positions, low, high, color=color, alpha=0.16)
    event_boundaries = sorted({
        int(value)
        for row in pair_rows
        for value in str(row.get("quantization_frames", "")).split(";")
        if value
    })
    for boundary in event_boundaries:
        axis.axvline(boundary, color="#222222", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Latent-frame position")
    axis.set_ylabel("Relative L2")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "online_latent_absolute_curve.png", dpi=180)
    plt.close(fig)

    keys = sorted({
        (int(row["prompt_index"]), int(row["seed"]), int(row["sample_index"]))
        for row in position_rows
    })
    by_key_position = {
        (
            int(row["prompt_index"]),
            int(row["seed"]),
            int(row["sample_index"]),
            int(row["position"]),
        ): float(row["trq_excess_rel_l2"])
        for row in position_rows
    }
    heatmap = np.asarray([
        [by_key_position[(*key, int(position))] for position in positions]
        for key in keys
    ])
    fig, axis = plt.subplots(figsize=(11, max(3, 0.35 * len(keys))))
    image = axis.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="magma")
    axis.set_xlabel("Latent-frame position")
    axis.set_ylabel("Prompt / seed / sample")
    axis.set_yticks(range(len(keys)), [f"{p}/{s}/{i}" for p, s, i in keys])
    fig.colorbar(image, ax=axis, label="TRQ excess Rel-L2")
    fig.tight_layout()
    fig.savefig(output / "online_prompt_time_heatmap.png", dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
