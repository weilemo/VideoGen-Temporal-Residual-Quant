#!/usr/bin/env python3
"""Fit decoder-visible innovation gamma and export an S2++ runtime bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from trq.analysis.conditional_codec import codec_gamma_statistics
from trq.analysis.conditional_innovation import CrossKVModel
from trq.analysis.kv_dump import iter_kv_dump_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--unit-size", type=int, default=0)
    parser.add_argument("--key-bits", type=int, default=4)
    parser.add_argument("--value-bits", type=int, default=4)
    parser.add_argument("--anchor-bits", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--rho", type=float, default=0.95)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.rho < 1 or args.ridge < 0:
        raise ValueError("rho must be in (0,1) and ridge must be non-negative")
    e1_dir = Path(args.e1_dir).expanduser().resolve()
    manifest = _read_json(e1_dir / "manifest.json")
    params = torch.load(
        e1_dir / "conditional_innovation_params.pt", map_location="cpu", weights_only=False
    )
    numerators: dict[int, torch.Tensor] = {}
    denominators: dict[int, torch.Tensor] = {}
    observations: dict[int, int] = {}
    for path_string in manifest["calibration_dumps"]:
        for layer, key, value, metadata in iter_kv_dump_layers(Path(path_string), layers=args.layers):
            layer_params = params["layers"].get(layer, params["layers"].get(str(layer)))
            if layer_params is None or "innovation_gamma" not in layer_params:
                raise ValueError(f"missing E1 hybrid parameters for layer {layer}")
            model = CrossKVModel(layer_params["cross_weight"], layer_params["cross_bias"])
            unit_size = args.unit_size or _metadata_unit_size(metadata)
            numerator, denominator, count = codec_gamma_statistics(
                key,
                value,
                model,
                layer_params["innovation_gamma"],
                unit_size=unit_size,
                key_bits=args.key_bits,
                value_bits=args.value_bits,
                anchor_bits=args.anchor_bits,
                block_size=args.block_size,
            )
            numerators[layer] = numerators.get(layer, torch.zeros_like(numerator)) + numerator
            denominators[layer] = denominators.get(layer, torch.zeros_like(denominator)) + denominator
            observations[layer] = observations.get(layer, 0) + count

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    codec_params = {**params, "predictor_kind": "hybrid_kv_innovation"}
    gamma_values = []
    for layer in sorted(numerators):
        scale = float(denominators[layer].mean().item())
        gamma = numerators[layer] / (
            denominators[layer] + args.ridge * max(scale, 1e-12)
        )
        gamma = gamma.clamp(-args.rho, args.rho).float().contiguous()
        layer_params = codec_params["layers"].get(
            layer, codec_params["layers"].get(str(layer))
        )
        layer_params["innovation_gamma"] = gamma
        gamma_values.append(gamma.reshape(-1))
    torch.save(codec_params, output / "conditional_innovation_params.pt")

    (output / "manifest.json").write_text(
        json.dumps({**manifest, "codec_gamma_parent_e1": str(e1_dir)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    layer_ids = sorted(int(layer) for layer in codec_params["layers"])
    sample = codec_params["layers"].get(layer_ids[0], codec_params["layers"].get(str(layer_ids[0])))
    weight_shape = tuple(sample["cross_weight"].shape)
    bias_shape = tuple(sample["cross_bias"].shape)
    gamma_shape = tuple(sample["innovation_gamma"].shape)
    cross_weight = np.zeros((max(layer_ids) + 1, *weight_shape), dtype=np.float32)
    cross_bias = np.zeros((max(layer_ids) + 1, *bias_shape), dtype=np.float32)
    innovation_gamma = np.zeros((max(layer_ids) + 1, *gamma_shape), dtype=np.float32)
    for layer in layer_ids:
        layer_params = codec_params["layers"].get(layer, codec_params["layers"].get(str(layer)))
        cross_weight[layer] = layer_params["cross_weight"].float().numpy()
        cross_bias[layer] = layer_params["cross_bias"].float().numpy()
        innovation_gamma[layer] = layer_params["innovation_gamma"].float().numpy()
    np.savez(
        output / "hybrid_codec_gamma.npz",
        predictor_kind=np.asarray("hybrid_kv_innovation"),
        cross_weight=cross_weight,
        cross_bias=cross_bias,
        innovation_gamma=innovation_gamma,
        predictor_stride=np.asarray(args.unit_size or 1560, dtype=np.int64),
    )
    np.savez(
        output / "cross_codec.npz",
        predictor_kind=np.asarray("cross_kv"),
        cross_weight=cross_weight,
        cross_bias=cross_bias,
        predictor_stride=np.asarray(args.unit_size or 1560, dtype=np.int64),
    )
    all_gamma = torch.cat(gamma_values).abs()
    gamma_p95 = float(torch.quantile(all_gamma, 0.95))
    summary = {
        "status": "PASS" if torch.isfinite(all_gamma).all() and gamma_p95 < args.rho else "FAIL",
        "gamma_abs_p95": gamma_p95,
        "gamma_abs_max": float(all_gamma.max()),
        "rho": args.rho,
        "observations_by_layer": observations,
        "scientific_scope": "codec-state gamma calibration only; validation is a separate gate",
    }
    (output / "codec_gamma_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "status": "PASS_E1" if summary["status"] == "PASS" else "FAIL_CODEC_GAMMA",
                "scientific_scope": "derived E1 parameter bundle with codec-state gamma",
                "codec_gamma_calibration": summary,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


def _metadata_unit_size(metadata: dict) -> int:
    for source in (metadata, metadata.get("dump_metadata", {})):
        if isinstance(source, dict):
            for key in ("frame_seq_length", "unit_size", "predictor_stride"):
                value = source.get(key)
                if value is not None and int(value) > 0:
                    return int(value)
    raise ValueError("unit size missing from dump metadata")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
