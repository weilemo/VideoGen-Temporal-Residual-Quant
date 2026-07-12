#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable

import numpy as np
import torch

from trq.analysis.kv_dump import iter_kv_dumps, resolve_dump_paths
from trq.analysis.trq_diagnostics import (
    DistributionAccumulator,
    aggregate_error_records,
    trace_temporal_residual_quantization,
)


SCHEMA_VERSION = 1
SOURCES = ("raw", "reference_residual", "residual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze raw KV concentration and temporal residual chain drift."
    )
    parser.add_argument("--dumps", required=True, help="Glob or comma-separated KV dump paths")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiments", choices=("all", "distribution", "drift"), default="all")
    parser.add_argument("--layers", default="all", help="all, comma list, or ranges such as 0,5,10-15")
    parser.add_argument("--kv", choices=("both", "K", "V"), default="both")
    parser.add_argument("--max-dumps", type=int, default=0, help="Zero means all matched dumps")
    parser.add_argument("--num-bits", type=int, default=2)
    parser.add_argument("--anchor-bits", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--predictor-stride", type=int, default=1560)
    parser.add_argument("--predictor-mode", choices=("identity", "affine_channel"), default="identity")
    parser.add_argument("--predictor-params-path", default="")
    parser.add_argument("--scale-precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--codec-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument(
        "--runtime-reset-interval",
        type=int,
        default=24,
        help="Predictor units per real runtime TRQ span",
    )
    parser.add_argument(
        "--reset-intervals",
        default="1,2,4,8,24,none",
        help="Drift ablation intervals in predictor units; none means no reset",
    )
    parser.add_argument("--sample-capacity", type=int, default=250_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-decoder-parity", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dump_paths = resolve_dump_paths(args.dumps)
    if args.max_dumps > 0:
        dump_paths = _limit_dump_samples(dump_paths, args.max_dumps)
    if not dump_paths:
        raise FileNotFoundError(f"No KV dumps matched {args.dumps!r}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    codec_dtype = torch.bfloat16 if args.codec_dtype == "bf16" else torch.float32
    kv_keys = ("K", "V") if args.kv == "both" else (args.kv,)
    reset_intervals = _parse_reset_intervals(args.reset_intervals)
    if args.runtime_reset_interval not in reset_intervals:
        reset_intervals.append(args.runtime_reset_interval)
    predictor_registry = _load_predictor_registry(args)

    config = {
        "schema_version": SCHEMA_VERSION,
        "experiments": args.experiments,
        "dumps": [str(path) for path in dump_paths],
        "layers": args.layers,
        "kv": list(kv_keys),
        "num_bits": args.num_bits,
        "anchor_bits": args.anchor_bits,
        "block_size": args.block_size,
        "predictor_stride": args.predictor_stride,
        "predictor_mode": args.predictor_mode,
        "predictor_params_path": args.predictor_params_path,
        "scale_precision": args.scale_precision,
        "codec_dtype": args.codec_dtype,
        "device": str(device),
        "runtime_reset_interval": args.runtime_reset_interval,
        "runtime_reset_label": "runtime",
        "reset_intervals": [_reset_label(value) for value in reset_intervals],
        "sample_capacity": args.sample_capacity,
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
    }
    _write_json(output_dir / "resolved_config.json", config)

    global_distributions = {
        (kv_key, source): DistributionAccumulator(
            sample_capacity=args.sample_capacity,
            seed=args.seed + index * 101,
        )
        for index, (kv_key, source) in enumerate(
            (pair for kv_key in kv_keys for pair in ((kv_key, source) for source in SOURCES))
        )
    }
    distribution_rows: list[dict] = []
    drift_rows: list[dict] = []
    dump_metadata: dict[str, dict] = {}

    for dump_index, (layer_idx, k_tensor, v_tensor, metadata) in enumerate(
        iter_kv_dumps(dump_paths, layers=args.layers)
    ):
        dump_path = str(metadata.get("dump_path", "unknown"))
        dump_id = str(
            metadata.get("dump_metadata", {}).get("sample_id")
            or _sample_id_from_path(Path(dump_path))
        )
        dump_metadata[dump_path] = metadata
        tensors = {"K": k_tensor, "V": v_tensor}
        for kv_key in kv_keys:
            tensor = tensors[kv_key].to(device=device)
            params = _predictor_params_for_layer(
                predictor_registry,
                kv_key=kv_key,
                layer_idx=layer_idx,
            )
            local_distributions = {
                source: DistributionAccumulator(
                    sample_capacity=min(args.sample_capacity, 50_000),
                    seed=args.seed + dump_index * 1009 + layer_idx * 17 + source_index,
                )
                for source_index, source in enumerate(SOURCES)
            }

            def collect_values(source: str, values: torch.Tensor) -> None:
                local_distributions[source].update(values)
                global_distributions[(kv_key, source)].update(values)

            runtime_span_lengths = _runtime_span_unit_lengths(
                metadata,
                predictor_stride=args.predictor_stride,
                sequence_tokens=int(tensor.shape[2]),
            )
            if runtime_span_lengths is None:
                runtime_trace_kwargs = {
                    "reset_interval_units": args.runtime_reset_interval,
                }
                runtime_schedule_source = "fallback_uniform"
                recorded_span_lengths = [args.runtime_reset_interval]
            else:
                runtime_trace_kwargs = {
                    "span_unit_lengths": runtime_span_lengths,
                }
                runtime_schedule_source = "dump_schedule"
                recorded_span_lengths = runtime_span_lengths
            runtime_records = trace_temporal_residual_quantization(
                tensor,
                num_bits=args.num_bits,
                block_size=args.block_size,
                anchor_bits=args.anchor_bits,
                predictor_stride=args.predictor_stride,
                predictor_mode=args.predictor_mode,
                predictor_params=params,
                layer_idx=layer_idx,
                scale_precision=args.scale_precision,
                codec_dtype=codec_dtype,
                value_callback=collect_values,
                verify_decoder=not args.skip_decoder_parity,
                **runtime_trace_kwargs,
            )
            non_anchor_runtime = [
                record for record in runtime_records if record["chain_position"] > 0
            ]
            distribution_rows.append(
                _distribution_row(
                    dump_id=dump_id,
                    dump_path=dump_path,
                    layer_idx=layer_idx,
                    kv_key=kv_key,
                    accumulators=local_distributions,
                    error_records=non_anchor_runtime,
                    all_runtime_records=runtime_records,
                    runtime_schedule_source=runtime_schedule_source,
                    runtime_span_unit_lengths=recorded_span_lengths,
                )
            )

            if args.experiments in {"all", "drift"}:
                for record in runtime_records:
                    drift_rows.append(
                        {
                            "dump_id": dump_id,
                            "dump_path": dump_path,
                            "layer": layer_idx,
                            "kv": kv_key,
                            "reset_interval": "runtime",
                            "runtime_schedule_source": runtime_schedule_source,
                            **record,
                        }
                    )
                for reset_interval in reset_intervals:
                    if (
                        runtime_span_lengths is None
                        and reset_interval == args.runtime_reset_interval
                    ):
                        records = runtime_records
                    else:
                        records = trace_temporal_residual_quantization(
                            tensor,
                            num_bits=args.num_bits,
                            block_size=args.block_size,
                            anchor_bits=args.anchor_bits,
                            predictor_stride=args.predictor_stride,
                            predictor_mode=args.predictor_mode,
                            predictor_params=params,
                            layer_idx=layer_idx,
                            scale_precision=args.scale_precision,
                            codec_dtype=codec_dtype,
                            reset_interval_units=reset_interval,
                            verify_decoder=not args.skip_decoder_parity,
                        )
                    reset_label = _reset_label(reset_interval)
                    for record in records:
                        drift_rows.append(
                            {
                                "dump_id": dump_id,
                                "dump_path": dump_path,
                                "layer": layer_idx,
                                "kv": kv_key,
                                "reset_interval": reset_label,
                                **record,
                            }
                        )

            del tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del k_tensor, v_tensor
        gc.collect()

    if not distribution_rows:
        raise RuntimeError("No layer/KV tensors were analyzed")

    distribution_summary = _summarize_distribution_hypotheses(
        distribution_rows,
        kv_keys=kv_keys,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    for kv_key in kv_keys:
        source_stats = {
            source: global_distributions[(kv_key, source)].summary()
            for source in SOURCES
        }
        distribution_summary[kv_key]["global_distributions"] = source_stats
        distribution_summary[kv_key]["global_residual_over_raw_rms"] = _ratio(
            source_stats["residual"]["rms"], source_stats["raw"]["rms"]
        )
    drift_summary_rows = _summarize_drift_rows(drift_rows)
    drift_verdicts = _summarize_drift_verdicts(
        drift_summary_rows,
        kv_keys=kv_keys,
        reset_label="runtime",
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "distribution": distribution_summary,
        "drift": drift_verdicts,
    }

    _write_jsonl(output_dir / "distribution_rows.jsonl", distribution_rows)
    _write_csv(output_dir / "distribution_summary.csv", distribution_rows)
    if drift_rows:
        _write_jsonl(output_dir / "chain_drift_rows.jsonl", drift_rows)
        _write_csv(output_dir / "chain_drift_summary.csv", drift_summary_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "manifest.json",
        _build_manifest(config, dump_paths, dump_metadata),
    )

    if not args.no_plots:
        _plot_distributions(global_distributions, distribution_summary, figures_dir, kv_keys)
        if drift_rows:
            _plot_chain_drift(
                drift_summary_rows,
                figures_dir,
                kv_keys=kv_keys,
                reset_label="runtime",
            )
            _plot_drift_heatmap(
                drift_rows,
                figures_dir,
                kv_keys=kv_keys,
                reset_label="runtime",
            )
            _plot_reset_ablation(drift_rows, figures_dir, kv_keys)
    analyzed_dump_count = len({row["dump_id"] for row in distribution_rows})
    _write_report(output_dir / "report.md", config, summary, analyzed_dump_count)
    print(f"TRQ diagnostics complete: {output_dir}")


def _distribution_row(
    *,
    dump_id: str,
    dump_path: str,
    layer_idx: int,
    kv_key: str,
    accumulators: dict[str, DistributionAccumulator],
    error_records: list[dict],
    all_runtime_records: list[dict],
    runtime_schedule_source: str,
    runtime_span_unit_lengths: list[int],
) -> dict:
    summaries = {source: accumulator.summary() for source, accumulator in accumulators.items()}
    errors = aggregate_error_records(error_records)
    raw = summaries["raw"]
    residual = summaries["residual"]
    reference = summaries["reference_residual"]
    row = {
        "dump_id": dump_id,
        "dump_path": dump_path,
        "layer": layer_idx,
        "kv": kv_key,
        "runtime_schedule_source": runtime_schedule_source,
        "runtime_span_unit_lengths": runtime_span_unit_lengths,
        "paired_value_count": raw["count"],
        "residual_over_raw_std": _ratio(residual["std"], raw["std"]),
        "residual_over_raw_rms": _ratio(residual["rms"], raw["rms"]),
        "residual_over_raw_p99_abs": _ratio(residual["p99_abs"], raw["p99_abs"]),
        "reference_residual_over_raw_rms": _ratio(reference["rms"], raw["rms"]),
        "trq_state_nbytes": sum(int(record["span_state_nbytes"]) for record in all_runtime_records),
        "trq_effective_bits_per_value": _ratio(
            8.0 * sum(int(record["span_state_nbytes"]) for record in all_runtime_records),
            sum(int(record["num_values"]) for record in all_runtime_records),
        ),
        **{f"raw_{key}": value for key, value in raw.items()},
        **{f"chain_residual_{key}": value for key, value in residual.items()},
        **{f"reference_residual_{key}": value for key, value in reference.items()},
        **errors,
    }
    row["raw_nmse"] = _ratio(row.get("raw_error_sse", float("nan")), row.get("target_sse", float("nan")))
    row["trq_nmse"] = _ratio(row.get("chain_error_sse", float("nan")), row.get("target_sse", float("nan")))
    row["trq_minus_raw_nmse"] = row["trq_nmse"] - row["raw_nmse"]
    return row


def _summarize_distribution_hypotheses(
    rows: list[dict],
    *,
    kv_keys: Iterable[str],
    bootstrap_resamples: int,
    seed: int,
) -> dict:
    result = {}
    for kv_index, kv_key in enumerate(kv_keys):
        selected = [row for row in rows if row["kv"] == kv_key]
        dump_ids = sorted({row["dump_id"] for row in selected})
        ratios = np.asarray(
            [
                np.median([
                    row["residual_over_raw_rms"] for row in selected if row["dump_id"] == dump_id
                ])
                for dump_id in dump_ids
            ],
            dtype=np.float64,
        )
        nmse_delta = np.asarray(
            [
                np.median([
                    row["trq_minus_raw_nmse"] for row in selected if row["dump_id"] == dump_id
                ])
                for dump_id in dump_ids
            ],
            dtype=np.float64,
        )
        dump_count = len(dump_ids)
        ratio_ci = _bootstrap_ci(
            np.log(np.clip(ratios, 1e-30, None)),
            np.median,
            bootstrap_resamples,
            seed + kv_index,
        )
        nmse_ci = _bootstrap_ci(
            nmse_delta,
            np.median,
            bootstrap_resamples,
            seed + 100 + kv_index,
        )
        win_rate = float(np.mean(ratios < 1.0)) if ratios.size else float("nan")
        result[kv_key] = {
            "n_rows": len(selected),
            "n_dumps": dump_count,
            "median_residual_over_raw_rms": float(np.median(ratios)),
            "median_log_rms_ratio_ci95": ratio_ci,
            "concentration_win_rate": win_rate,
            "concentration_hypothesis_supported": bool(
                dump_count >= 2 and ratio_ci[1] < 0.0 and win_rate >= 0.80
            ),
            "median_trq_minus_raw_nmse": float(np.median(nmse_delta)),
            "median_trq_minus_raw_nmse_ci95": nmse_ci,
            "int2_hypothesis_supported": bool(dump_count >= 2 and nmse_ci[1] < 0.0),
        }
    return result


def _summarize_drift_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        key = (str(row["reset_interval"]), row["kv"], int(row["chain_position"]))
        groups.setdefault(key, []).append(row)
    summaries = []
    for (reset_interval, kv_key, chain_position), selected in sorted(groups.items()):
        aggregate = aggregate_error_records(selected)
        dump_ids = sorted({row["dump_id"] for row in selected})
        chain_values = np.asarray([
            np.mean([row["chain_rel_l2"] for row in selected if row["dump_id"] == dump_id])
            for dump_id in dump_ids
        ], dtype=np.float64)
        teacher_values = np.asarray([
            np.mean([row["teacher_rel_l2"] for row in selected if row["dump_id"] == dump_id])
            for dump_id in dump_ids
        ], dtype=np.float64)
        delta_values = np.asarray([
            np.mean([
                math.sqrt(row["chain_teacher_delta_sse"] / max(row["target_sse"], 1e-30))
                for row in selected if row["dump_id"] == dump_id
            ])
            for dump_id in dump_ids
        ], dtype=np.float64)
        summaries.append(
            {
                "reset_interval": reset_interval,
                "kv": kv_key,
                "chain_position": chain_position,
                "n": len(selected),
                "n_dumps": len(dump_ids),
                **aggregate,
                "chain_macro_mean": float(chain_values.mean()),
                "chain_macro_ci_low": _normal_ci(chain_values)[0],
                "chain_macro_ci_high": _normal_ci(chain_values)[1],
                "teacher_macro_mean": float(teacher_values.mean()),
                "teacher_macro_ci_low": _normal_ci(teacher_values)[0],
                "teacher_macro_ci_high": _normal_ci(teacher_values)[1],
                "excess_macro_mean": float(delta_values.mean()),
                "excess_macro_ci_low": _normal_ci(delta_values)[0],
                "excess_macro_ci_high": _normal_ci(delta_values)[1],
            }
        )
    return summaries


def _summarize_drift_verdicts(
    summary_rows: list[dict],
    *,
    kv_keys: Iterable[str],
    reset_label: str,
) -> dict:
    result = {}
    for kv_key in kv_keys:
        selected = [
            row for row in summary_rows
            if row["kv"] == kv_key
            and row["reset_interval"] == reset_label
            and row["chain_position"] > 0
        ]
        selected.sort(key=lambda row: row["chain_position"])
        if not selected:
            result[kv_key] = {"n_positions": 0}
            continue
        x = np.asarray([row["chain_position"] for row in selected], dtype=np.float64)
        chain = np.asarray([row["chain_rel_l2"] for row in selected], dtype=np.float64)
        teacher = np.asarray([row["teacher_rel_l2"] for row in selected], dtype=np.float64)
        excess = np.asarray([row["chain_teacher_delta_rel_l2"] for row in selected], dtype=np.float64)
        slope = float(np.polyfit(x, chain, deg=1)[0]) if x.size > 1 else 0.0
        result[kv_key] = {
            "n_positions": int(x.size),
            "chain_rel_l2_slope_per_position": slope,
            "late_over_early_chain_error": _ratio(chain[-1], chain[0]),
            "max_chain_rel_l2": float(chain.max()),
            "max_teacher_rel_l2": float(teacher.max()),
            "max_excess_rel_l2": float(excess.max()),
            "last_chain_rel_l2": float(chain[-1]),
            "last_teacher_rel_l2": float(teacher[-1]),
        }
    return result


def _plot_distributions(accumulators, summary, figures_dir: Path, kv_keys: Iterable[str]) -> None:
    plt = _load_pyplot()
    kv_keys = list(kv_keys)
    figure, axes = plt.subplots(len(kv_keys), 2, figsize=(12, 4.2 * len(kv_keys)), squeeze=False)
    colors = {"raw": "#333333", "reference_residual": "#2c7fb8", "residual": "#d95f0e"}
    labels = {"raw": "Raw KV", "reference_residual": "Oracle residual", "residual": "Chain residual"}
    for row_index, kv_key in enumerate(kv_keys):
        samples = {
            source: accumulators[(kv_key, source)].samples
            for source in SOURCES
        }
        raw_abs = np.abs(samples["raw"])
        limit = float(np.quantile(raw_abs, 0.995)) if raw_abs.size else 1.0
        limit = max(limit, 1e-8)
        bins = np.linspace(-limit, limit, 240)
        for source in SOURCES:
            values = samples[source]
            if values.size:
                axes[row_index, 0].hist(
                    np.clip(values, -limit, limit),
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.5,
                    color=colors[source],
                    label=labels[source],
                )
                abs_values = np.sort(np.abs(values))
                cdf = np.arange(1, abs_values.size + 1) / abs_values.size
                axes[row_index, 1].plot(abs_values, cdf, color=colors[source], label=labels[source])
        axes[row_index, 0].set_yscale("log")
        axes[row_index, 0].set_title(f"{kv_key}: signed density (clipped at raw p99.5)")
        axes[row_index, 0].set_xlabel("value")
        axes[row_index, 0].set_ylabel("density")
        axes[row_index, 1].set_xscale("log")
        axes[row_index, 1].set_title(f"{kv_key}: ECDF of absolute magnitude")
        axes[row_index, 1].set_xlabel("absolute value")
        axes[row_index, 1].set_ylabel("CDF")
        ratio = summary[kv_key]["median_residual_over_raw_rms"]
        axes[row_index, 0].text(0.02, 0.96, f"median RMS ratio={ratio:.3f}", transform=axes[row_index, 0].transAxes, va="top")
        for axis in axes[row_index]:
            axis.grid(alpha=0.2)
            axis.legend()
    figure.tight_layout()
    _save_figure(figure, figures_dir / "raw_vs_residual_distribution")
    plt.close(figure)


def _plot_chain_drift(summary_rows, figures_dir: Path, *, kv_keys, reset_label: str) -> None:
    plt = _load_pyplot()
    kv_keys = list(kv_keys)
    figure, axes = plt.subplots(1, len(kv_keys), figsize=(6 * len(kv_keys), 4.5), squeeze=False)
    specs = (
        ("chain", "Chain", "#d95f0e"),
        ("teacher", "Teacher-forced", "#2c7fb8"),
        ("excess", "Chain excess", "#756bb1"),
    )
    for index, kv_key in enumerate(kv_keys):
        axis = axes[0, index]
        selected = [
            row for row in summary_rows
            if row["kv"] == kv_key
            and row["reset_interval"] == reset_label
            and row["chain_position"] > 0
        ]
        selected.sort(key=lambda row: row["chain_position"])
        x = np.asarray([row["chain_position"] for row in selected])
        for prefix, label, color in specs:
            mean = np.asarray([row[f"{prefix}_macro_mean"] for row in selected])
            low = np.asarray([row[f"{prefix}_macro_ci_low"] for row in selected])
            high = np.asarray([row[f"{prefix}_macro_ci_high"] for row in selected])
            axis.plot(x, mean, marker="o", markersize=3, color=color, label=label)
            axis.fill_between(x, low, high, color=color, alpha=0.15)
        axis.set_title(f"{kv_key}: residual-chain drift")
        axis.set_xlabel("position inside TRQ span")
        axis.set_ylabel("relative L2")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    _save_figure(figure, figures_dir / "residual_chain_drift")
    plt.close(figure)


def _plot_drift_heatmap(rows, figures_dir: Path, *, kv_keys, reset_label: str) -> None:
    plt = _load_pyplot()
    kv_keys = list(kv_keys)
    figure, axes = plt.subplots(1, len(kv_keys), figsize=(6 * len(kv_keys), 6), squeeze=False)
    for index, kv_key in enumerate(kv_keys):
        selected = [
            row for row in rows
            if row["kv"] == kv_key
            and row["reset_interval"] == reset_label
            and row["chain_position"] > 0
        ]
        layers = sorted({int(row["layer"]) for row in selected})
        positions = sorted({int(row["chain_position"]) for row in selected})
        matrix = np.full((len(layers), len(positions)), np.nan)
        for layer_index, layer in enumerate(layers):
            for position_index, position in enumerate(positions):
                values = [
                    math.sqrt(row["chain_teacher_delta_sse"] / max(row["target_sse"], 1e-30))
                    for row in selected
                    if int(row["layer"]) == layer and int(row["chain_position"]) == position
                ]
                if values:
                    matrix[layer_index, position_index] = float(np.mean(values))
        image = axes[0, index].imshow(matrix, aspect="auto", origin="lower", interpolation="nearest")
        axes[0, index].set_title(f"{kv_key}: chain excess by layer")
        axes[0, index].set_xlabel("position inside span")
        axes[0, index].set_ylabel("layer")
        axes[0, index].set_xticks(range(len(positions)), positions)
        axes[0, index].set_yticks(range(len(layers)), layers)
        figure.colorbar(image, ax=axes[0, index], label="excess relative L2")
    figure.tight_layout()
    _save_figure(figure, figures_dir / "chain_drift_layer_heatmap")
    plt.close(figure)


def _plot_reset_ablation(rows, figures_dir: Path, kv_keys: Iterable[str]) -> None:
    plt = _load_pyplot()
    kv_keys = list(kv_keys)
    labels = sorted({str(row["reset_interval"]) for row in rows}, key=_reset_sort_key)
    figure, axes = plt.subplots(1, len(kv_keys), figsize=(5.5 * len(kv_keys), 4.2), squeeze=False)
    for index, kv_key in enumerate(kv_keys):
        values = []
        for label in labels:
            errors = [
                row["chain_rel_l2"]
                for row in rows
                if row["kv"] == kv_key and str(row["reset_interval"]) == label
            ]
            values.append(float(np.quantile(errors, 0.95)) if errors else float("nan"))
        axes[0, index].plot(range(len(labels)), values, marker="o", color="#d95f0e")
        axes[0, index].set_xticks(range(len(labels)), labels)
        axes[0, index].set_title(f"{kv_key}: reset interval ablation")
        axes[0, index].set_xlabel("units per span")
        axes[0, index].set_ylabel("p95 chain relative L2")
        axes[0, index].grid(alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, figures_dir / "reset_interval_ablation")
    plt.close(figure)


def _write_report(path: Path, config: dict, summary: dict, dump_count: int) -> None:
    lines = [
        "# TRQ Diagnostic Report",
        "",
        "## Scope",
        "",
        f"- Dumps: {dump_count}",
        f"- Predictor: `{config['predictor_mode']}`",
        f"- Residual bits: {config['num_bits']}",
        f"- Anchor bits: {config['anchor_bits']}",
        f"- Block size: {config['block_size']}",
        f"- Predictor stride: {config['predictor_stride']} tokens",
        f"- Runtime schedule: dump metadata when available; otherwise "
        f"{config['runtime_reset_interval']} units per span",
        "- Raw and residual statistics use the same non-anchor target mask.",
        "- K is pre-RoPE because Self-Forcing stores pre-RoPE cache tensors.",
        "",
        "## Raw KV vs Residual",
        "",
        "| KV | Median residual/raw RMS | Win rate | INT2 NMSE delta | Concentration supported | INT2 supported |",
        "|---|---:|---:|---:|---|---|",
    ]
    for kv_key, values in summary["distribution"].items():
        lines.append(
            f"| {kv_key} | {values['median_residual_over_raw_rms']:.4f} | "
            f"{values['concentration_win_rate']:.1%} | {values['median_trq_minus_raw_nmse']:.6f} | "
            f"{values['concentration_hypothesis_supported']} | {values['int2_hypothesis_supported']} |"
        )
    lines.extend([
        "",
        "A hypothesis is only marked supported with at least two dumps. Concentration additionally requires the upper 95% bootstrap bound of median log RMS ratio below zero and at least 80% paired wins. INT2 requires the upper 95% bound of TRQ-minus-direct NMSE below zero.",
        "",
        "## Residual Chain Drift",
        "",
        "| KV | Positions | Slope / position | Late / early | Max chain Rel-L2 | Max teacher Rel-L2 | Max excess Rel-L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for kv_key, values in summary["drift"].items():
        if not values.get("n_positions"):
            lines.append(f"| {kv_key} | 0 | - | - | - | - | - |")
            continue
        lines.append(
            f"| {kv_key} | {values['n_positions']} | {values['chain_rel_l2_slope_per_position']:.6g} | "
            f"{values['late_over_early_chain_error']:.4f} | {values['max_chain_rel_l2']:.4f} | "
            f"{values['max_teacher_rel_l2']:.4f} | {values['max_excess_rel_l2']:.4f} |"
        )
    lines.extend([
        "",
        "Chain error is compared with a teacher-forced residual codec. Their difference isolates indirect propagation through the next residual range; residual errors do not algebraically add without bound.",
        "",
        "## Artifacts",
        "",
        "- `manifest.json`: provenance and dump metadata",
        "- `resolved_config.json`: exact analysis settings",
        "- `distribution_rows.jsonl`: paired per-dump/layer/KV statistics",
        "- `chain_drift_rows.jsonl`: per-unit traces",
        "- `distribution_summary.csv` and `chain_drift_summary.csv`: tabular summaries",
        "- `figures/`: PNG and PDF figures",
        "",
        "## Limitation",
        "",
        "This is an offline fixed-trajectory codec diagnostic. It measures cache reconstruction drift, not the downstream divergence of a TRQ-generated video from a paired BF16 rollout. A later online paired-seed experiment is required for that causal claim.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_manifest(config: dict, dump_paths: list[Path], metadata: dict[str, dict]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "git": _git_metadata(),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "config": config,
        "dumps": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": _sha256(path) if path.is_file() else None,
                "metadata": metadata.get(str(path), {}),
            }
            for path in dump_paths
        ],
    }


def _load_predictor_registry(args: argparse.Namespace) -> dict | None:
    if args.predictor_mode == "identity":
        return None
    path = Path(args.predictor_params_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            "affine_channel requires --predictor-params-path pointing to a fitted train-split file"
        )
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {
                "alpha": torch.from_numpy(data["alpha"].astype(np.float32)),
                "beta": torch.from_numpy(data["beta"].astype(np.float32)),
            }
    registry = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(registry, dict):
        raise ValueError(f"predictor file must contain a dictionary: {path}")
    return registry


def _predictor_params_for_layer(registry, *, kv_key: str, layer_idx: int) -> dict | None:
    if registry is None:
        return None
    if kv_key not in registry:
        return registry
    table = registry[kv_key]
    params = table.get(layer_idx, table.get(str(layer_idx)))
    if params is None:
        raise KeyError(f"predictor registry has no {kv_key} layer {layer_idx}")
    return params


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {value}")
    return device


def _parse_reset_intervals(value: str) -> list[int | None]:
    result: list[int | None] = []
    for item in value.split(","):
        item = item.strip().lower()
        parsed = None if item in {"none", "full", "no-reset"} else int(item)
        if parsed is not None and parsed <= 0:
            raise ValueError(f"reset interval must be positive: {item}")
        if parsed not in result:
            result.append(parsed)
    return result


def _reset_label(value: int | None) -> str:
    return "none" if value is None else str(value)


def _reset_sort_key(value: str) -> tuple[int, int]:
    if value == "runtime":
        return (-1, 0)
    return (1, 0) if value == "none" else (0, int(value))


def _runtime_span_unit_lengths(
    record_metadata: dict,
    *,
    predictor_stride: int,
    sequence_tokens: int,
) -> list[int] | None:
    dump_metadata = record_metadata.get("dump_metadata", {})
    if not isinstance(dump_metadata, dict):
        return None
    raw_spans = dump_metadata.get("scheduled_trq_frame_spans")
    if raw_spans is None:
        return None
    if not isinstance(raw_spans, list) or not raw_spans:
        raise ValueError(
            "Dump metadata reports no scheduled TRQ spans; generate at least one full "
            "quantization interval before running the residual diagnostics"
        )
    frame_seq_length = int(dump_metadata.get("frame_seq_length", predictor_stride))
    lengths = []
    expected_start_frame = 0
    covered_tokens = 0
    for span in raw_spans:
        if not isinstance(span, dict):
            raise ValueError("scheduled_trq_frame_spans entries must be mappings")
        start_frame = int(span["start_frame"])
        end_frame = int(span["end_frame"])
        if start_frame != expected_start_frame or end_frame <= start_frame:
            raise ValueError(
                "scheduled_trq_frame_spans must be a positive contiguous prefix; "
                f"got {start_frame}:{end_frame} after {expected_start_frame}"
            )
        span_tokens = (end_frame - start_frame) * frame_seq_length
        if span_tokens % predictor_stride != 0:
            raise ValueError(
                f"Runtime span has {span_tokens} tokens, not divisible by predictor "
                f"stride {predictor_stride}"
            )
        lengths.append(span_tokens // predictor_stride)
        covered_tokens += span_tokens
        expected_start_frame = end_frame
    if covered_tokens > sequence_tokens:
        raise ValueError(
            f"Runtime schedule covers {covered_tokens} tokens, but dump has {sequence_tokens}"
        )
    return lengths


def _sample_id_from_path(path: Path) -> str:
    return re.sub(r"_layer\d+$", "", path.stem)


def _limit_dump_samples(paths: list[Path], max_samples: int) -> list[Path]:
    selected_ids: list[str] = []
    for path in paths:
        sample_id = _sample_id_from_path(path)
        if sample_id not in selected_ids:
            selected_ids.append(sample_id)
        if len(selected_ids) == max_samples:
            break
    allowed = set(selected_ids)
    return [path for path in paths if _sample_id_from_path(path) in allowed]


def _bootstrap_ci(values, statistic, resamples: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [float("nan"), float("nan")]
    if values.size == 1 or resamples <= 0:
        estimate = float(statistic(values))
        return [estimate, estimate]
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = values[rng.integers(0, values.size, size=values.size)]
        estimates[index] = statistic(sample)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _normal_ci(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean
    margin = 1.96 * float(values.std(ddof=1)) / math.sqrt(values.size)
    return mean - margin, mean + margin


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(figure, base_path: Path) -> None:
    figure.savefig(base_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _git_metadata() -> dict:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], check=False, capture_output=True, text=True
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1e-30)


if __name__ == "__main__":
    main()
