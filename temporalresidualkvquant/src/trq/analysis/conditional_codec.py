"""Fixed-bit reconstructed-state simulation for conditional innovation coding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from ..real.s2pp import (
    _asymmetric_scale_and_zero_point,
    _dequantize_blockwise,
    _dequantize_blockwise_asymmetric,
    _quantize_blockwise,
    _quantize_blockwise_asymmetric,
    _symmetric_scales,
)
from ..real.trq import trq_state_nbytes
from .conditional_innovation import CrossKVModel, full_units


@dataclass
class _MetricAccumulator:
    squared_error: float = 0.0
    target_energy: float = 0.0
    values: int = 0
    payload_bytes: int = 0

    def update(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        payload_bytes: int,
    ) -> dict[str, float | int]:
        error = target.float() - reconstruction.float()
        squared_error = float(error.double().square().sum().item())
        target_energy = float(target.double().square().sum().item())
        values = int(target.numel())
        self.squared_error += squared_error
        self.target_energy += target_energy
        self.values += values
        self.payload_bytes += int(payload_bytes)
        return {
            "mse": squared_error / max(values, 1),
            "rel_l2": math.sqrt(squared_error / max(target_energy, 1e-30)),
            "max_abs": float(error.abs().max().item()),
            "payload_bytes": int(payload_bytes),
        }

    def summary(self) -> dict[str, float | int]:
        return {
            "squared_error": self.squared_error,
            "target_energy": self.target_energy,
            "mse": self.squared_error / max(self.values, 1),
            "rel_l2": math.sqrt(self.squared_error / max(self.target_energy, 1e-30)),
            "payload_bytes": self.payload_bytes,
            "values": self.values,
        }


def simulate_conditional_codec(
    key: torch.Tensor,
    value: torch.Tensor,
    model: CrossKVModel,
    gamma: torch.Tensor,
    *,
    unit_size: int,
    key_bits: int = 4,
    value_bits: int = 4,
    anchor_bits: int = 4,
    block_size: int = 64,
    reset_spans: tuple[int, ...] = (2, 4, 8),
    scale_precision: torch.dtype = torch.bfloat16,
    shuffled_innovations: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Compare direct, temporal, Cross-KV, oracle, and closed-loop codecs.

    K and all decoder-reproducible V predictors use only reconstructed state.
    The oracle method deliberately uses BF16 K/V history and is reported only
    as a structural upper bound. Quantized payloads use the same packed
    asymmetric residual format as TRQ v1.
    """
    if key.shape != value.shape or key.ndim != 4:
        raise ValueError("K/V must have the same BHSD shape")
    if tuple(gamma.shape) != (key.shape[1], key.shape[-1]):
        raise ValueError(f"gamma must be [H,D], got {tuple(gamma.shape)}")
    if not reset_spans or any(int(span) <= 0 for span in reset_spans):
        raise ValueError("reset spans must be positive")

    k_units = full_units(key, unit_size)
    v_units = full_units(value, unit_size)
    if len(k_units) < 2:
        raise ValueError("closed-loop E2 requires at least two complete units")
    if shuffled_innovations is None:
        raise ValueError("E2 requires reconstructed innovations from a prompt-disjoint donor")
    if len(shuffled_innovations) < len(k_units) - 1:
        raise ValueError("shuffled donor has fewer complete units than the evaluated prompt")
    for donor, expected in zip(shuffled_innovations, v_units[:-1]):
        if donor.shape != expected.shape:
            raise ValueError(
                f"shuffled donor geometry {tuple(donor.shape)} does not match "
                f"{tuple(expected.shape)}"
            )

    reset_spans = tuple(dict.fromkeys(int(span) for span in reset_spans))
    method_names = [
        "direct_v",
        "temporal",
        "cross",
        "oracle_hybrid",
        "closed_hybrid",
        "shuffled_hybrid",
    ]
    method_names.extend(f"closed_hybrid_reset_{span}" for span in reset_spans)
    metrics = {name: _MetricAccumulator() for name in method_names}
    previous_v: dict[str, torch.Tensor] = {}
    unit_rows: list[dict[str, float | int | str | bool]] = []
    gamma_view = gamma.float()[None, :, None, :]
    previous_k_reconstruction: torch.Tensor | None = None
    key_payload_bytes = 0

    for unit_index, (k_unit, v_unit) in enumerate(zip(k_units, v_units)):
        if unit_index == 0:
            k_reconstruction, k_bytes = _quantize_reconstruct(
                k_unit, anchor_bits, block_size, scale_precision, symmetric=True
            )
        else:
            assert previous_k_reconstruction is not None
            k_residual = k_unit.float() - previous_k_reconstruction.float()
            decoded_residual, k_bytes = _quantize_reconstruct(
                k_residual, key_bits, block_size, scale_precision, symmetric=False
            )
            k_reconstruction = (previous_k_reconstruction.float() + decoded_residual).to(key.dtype)
        key_payload_bytes += k_bytes

        direct_reconstruction, direct_bytes = _quantize_reconstruct(
            v_unit, value_bits, block_size, scale_precision, symmetric=False
        )
        direct_row = metrics["direct_v"].update(v_unit, direct_reconstruction, direct_bytes)
        unit_rows.append(
            {"unit": unit_index, "method": "direct_v", "reset": True, **direct_row}
        )

        if unit_index == 0:
            anchor_reconstruction, anchor_bytes = _quantize_reconstruct(
                v_unit, anchor_bits, block_size, scale_precision, symmetric=True
            )
            for name in method_names[1:]:
                previous_v[name] = anchor_reconstruction
                row = metrics[name].update(v_unit, anchor_reconstruction, anchor_bytes)
                unit_rows.append({"unit": unit_index, "method": name, "reset": True, **row})
            previous_k_reconstruction = k_reconstruction
            continue

        assert previous_k_reconstruction is not None
        cross_prediction = model.predict(k_reconstruction)
        previous_cross = model.predict(previous_k_reconstruction)
        oracle_cross = model.predict(k_unit)
        oracle_previous_cross = model.predict(k_units[unit_index - 1])
        shuffled_innovation = shuffled_innovations[unit_index - 1].float()
        predictions: dict[str, torch.Tensor] = {
            "temporal": previous_v["temporal"].float(),
            "cross": cross_prediction,
            "oracle_hybrid": oracle_cross
            + gamma_view * (v_units[unit_index - 1].float() - oracle_previous_cross),
            "closed_hybrid": cross_prediction
            + gamma_view * (previous_v["closed_hybrid"].float() - previous_cross),
            "shuffled_hybrid": cross_prediction
            + gamma_view * shuffled_innovation,
        }
        for span in reset_spans:
            name = f"closed_hybrid_reset_{span}"
            if unit_index % span:
                predictions[name] = cross_prediction + gamma_view * (
                    previous_v[name].float() - previous_cross
                )

        for name in method_names[1:]:
            is_reset = name.startswith("closed_hybrid_reset_") and name not in predictions
            if is_reset:
                reconstruction, payload_bytes = _quantize_reconstruct(
                    v_unit, anchor_bits, block_size, scale_precision, symmetric=True
                )
            else:
                prediction = predictions[name]
                residual = v_unit.float() - prediction
                decoded_residual, payload_bytes = _quantize_reconstruct(
                    residual, value_bits, block_size, scale_precision, symmetric=False
                )
                reconstruction = (prediction + decoded_residual).to(value.dtype)
            previous_v[name] = reconstruction
            row = metrics[name].update(v_unit, reconstruction, payload_bytes)
            unit_rows.append({"unit": unit_index, "method": name, "reset": is_reset, **row})

        previous_k_reconstruction = k_reconstruction

    predictor_bytes = trq_state_nbytes({"weight": model.weight, "bias": model.bias})
    gamma_bytes = trq_state_nbytes(gamma)
    summaries: dict[str, dict[str, float | int | bool]] = {}
    for name, accumulator in metrics.items():
        summary = accumulator.summary()
        method_predictor_bytes = 0 if name in {"direct_v", "temporal"} else predictor_bytes
        if "hybrid" in name:
            method_predictor_bytes += gamma_bytes
        summary["key_payload_bytes"] = key_payload_bytes
        summary["predictor_bytes"] = method_predictor_bytes
        summary["physical_bytes"] = int(
            summary["payload_bytes"] + key_payload_bytes + method_predictor_bytes
        )
        method_units = [row for row in unit_rows if row["method"] == name]
        rel_l2_values = [float(row["rel_l2"]) for row in method_units]
        summary["final_unit_rel_l2"] = rel_l2_values[-1]
        summary["max_unit_rel_l2"] = max(rel_l2_values)
        summary["monotonic_increase_fraction"] = _increase_fraction(rel_l2_values)
        summary["finite"] = all(math.isfinite(value) for value in rel_l2_values)
        summaries[name] = summary

    return {
        "complete_units": len(k_units),
        "ignored_tokens": int(key.shape[2] - len(k_units) * unit_size),
        "shared_key_codec": "identity temporal residual; reconstructed state",
        "methods": summaries,
        "unit_rows": unit_rows,
    }


def reconstruct_closed_loop_innovations(
    key: torch.Tensor,
    value: torch.Tensor,
    model: CrossKVModel,
    gamma: torch.Tensor,
    *,
    unit_size: int,
    key_bits: int = 4,
    value_bits: int = 4,
    anchor_bits: int = 4,
    block_size: int = 64,
    scale_precision: torch.dtype = torch.bfloat16,
) -> list[torch.Tensor]:
    """Reconstruct decoder-visible innovations for a shuffled donor prompt."""
    if key.shape != value.shape or key.ndim != 4:
        raise ValueError("K/V must have the same BHSD shape")
    if tuple(gamma.shape) != (key.shape[1], key.shape[-1]):
        raise ValueError(f"gamma must be [H,D], got {tuple(gamma.shape)}")
    k_units = full_units(key, unit_size)
    v_units = full_units(value, unit_size)
    if len(k_units) < 2:
        raise ValueError("shuffled donor requires at least two complete units")

    gamma_view = gamma.float()[None, :, None, :]
    innovations: list[torch.Tensor] = []
    previous_k: torch.Tensor | None = None
    previous_v: torch.Tensor | None = None
    for unit_index, (k_unit, v_unit) in enumerate(zip(k_units, v_units)):
        if unit_index == 0:
            k_reconstruction, _ = _quantize_reconstruct(
                k_unit, anchor_bits, block_size, scale_precision, symmetric=True
            )
            v_reconstruction, _ = _quantize_reconstruct(
                v_unit, anchor_bits, block_size, scale_precision, symmetric=True
            )
        else:
            assert previous_k is not None and previous_v is not None
            k_residual = k_unit.float() - previous_k.float()
            decoded_k_residual, _ = _quantize_reconstruct(
                k_residual, key_bits, block_size, scale_precision, symmetric=False
            )
            k_reconstruction = (previous_k.float() + decoded_k_residual).to(key.dtype)
            prediction = model.predict(k_reconstruction) + gamma_view * (
                previous_v.float() - model.predict(previous_k)
            )
            decoded_v_residual, _ = _quantize_reconstruct(
                v_unit.float() - prediction,
                value_bits,
                block_size,
                scale_precision,
                symmetric=False,
            )
            v_reconstruction = (prediction + decoded_v_residual).to(value.dtype)
        innovation = v_reconstruction.float() - model.predict(k_reconstruction)
        innovations.append(innovation.to(value.dtype))
        previous_k = k_reconstruction
        previous_v = v_reconstruction
    return innovations


def codec_gamma_statistics(
    key: torch.Tensor,
    value: torch.Tensor,
    model: CrossKVModel,
    seed_gamma: torch.Tensor,
    *,
    unit_size: int,
    key_bits: int = 4,
    value_bits: int = 4,
    anchor_bits: int = 4,
    block_size: int = 64,
    scale_precision: torch.dtype = torch.bfloat16,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Accumulate decoder-visible gamma regression statistics.

    The regressor is the previous reconstructed innovation and the target is
    the current BF16 innovation relative to reconstructed K. The seed gamma is
    used only to create the closed-loop calibration trajectory.
    """
    if key.shape != value.shape or key.ndim != 4:
        raise ValueError("K/V must have the same BHSD shape")
    if tuple(seed_gamma.shape) != (key.shape[1], key.shape[-1]):
        raise ValueError(f"seed gamma must be [H,D], got {tuple(seed_gamma.shape)}")
    k_units = full_units(key, unit_size)
    v_units = full_units(value, unit_size)
    if len(k_units) < 2:
        raise ValueError("codec gamma calibration requires at least two complete units")

    numerator = torch.zeros_like(seed_gamma, dtype=torch.float64)
    denominator = torch.zeros_like(seed_gamma, dtype=torch.float64)
    observations = 0
    gamma_view = seed_gamma.float()[None, :, None, :]
    previous_k: torch.Tensor | None = None
    previous_v: torch.Tensor | None = None
    for unit_index, (k_unit, v_unit) in enumerate(zip(k_units, v_units)):
        if unit_index == 0:
            _update_quantizer_diagnostics(
                diagnostics,
                k_unit,
                anchor_bits,
                block_size,
                scale_precision,
                symmetric=True,
                role="K_anchor",
            )
            k_reconstruction, _ = _quantize_reconstruct(
                k_unit, anchor_bits, block_size, scale_precision, symmetric=True
            )
            _update_quantizer_diagnostics(
                diagnostics,
                v_unit,
                anchor_bits,
                block_size,
                scale_precision,
                symmetric=True,
                role="V_anchor",
            )
            v_reconstruction, _ = _quantize_reconstruct(
                v_unit, anchor_bits, block_size, scale_precision, symmetric=True
            )
        else:
            assert previous_k is not None and previous_v is not None
            k_residual = k_unit.float() - previous_k.float()
            _update_quantizer_diagnostics(
                diagnostics,
                k_residual,
                key_bits,
                block_size,
                scale_precision,
                symmetric=False,
                role="K_residual",
            )
            decoded_k_residual, _ = _quantize_reconstruct(
                k_residual,
                key_bits,
                block_size,
                scale_precision,
                symmetric=False,
            )
            k_reconstruction = (previous_k.float() + decoded_k_residual).to(key.dtype)
            previous_innovation = previous_v.float() - model.predict(previous_k)
            target_innovation = v_unit.float() - model.predict(k_reconstruction)
            numerator += (previous_innovation.double() * target_innovation.double()).sum(
                dim=(0, 2)
            )
            denominator += previous_innovation.double().square().sum(dim=(0, 2))
            observations += int(previous_innovation.shape[0] * previous_innovation.shape[2])
            prediction = model.predict(k_reconstruction) + gamma_view * previous_innovation
            v_residual = v_unit.float() - prediction
            _update_quantizer_diagnostics(
                diagnostics,
                v_residual,
                value_bits,
                block_size,
                scale_precision,
                symmetric=False,
                role="V_residual",
            )
            decoded_v_residual, _ = _quantize_reconstruct(
                v_residual,
                value_bits,
                block_size,
                scale_precision,
                symmetric=False,
            )
            v_reconstruction = (prediction + decoded_v_residual).to(value.dtype)
            if diagnostics is not None:
                previous_true = v_units[unit_index - 1].float()
                shock = torch.linalg.vector_norm(v_unit.float() - previous_true) / torch.linalg.vector_norm(
                    previous_true
                ).clamp_min(1e-12)
                error = torch.linalg.vector_norm(v_unit.float() - v_reconstruction.float()) / torch.linalg.vector_norm(
                    v_unit.float()
                ).clamp_min(1e-12)
                diagnostics.setdefault("event_rows", []).append(
                    {
                        "unit": unit_index,
                        "shock_rel_l2": float(shock.item()),
                        "closed_rel_l2": float(error.item()),
                    }
                )
        previous_k = k_reconstruction
        previous_v = v_reconstruction
    return numerator, denominator, observations


def _update_quantizer_diagnostics(
    diagnostics: dict[str, Any] | None,
    tensor: torch.Tensor,
    bits: int,
    block_size: int,
    scale_precision: torch.dtype,
    *,
    symmetric: bool,
    role: str | None = None,
) -> None:
    if diagnostics is None:
        return
    work = tensor.float()
    diagnostics["values"] = int(diagnostics.get("values", 0)) + int(work.numel())
    diagnostics["nonfinite"] = int(diagnostics.get("nonfinite", 0)) + int(
        (~torch.isfinite(work)).sum().item()
    )
    padded = math.ceil(work.shape[-1] / block_size) * block_size
    if padded != work.shape[-1]:
        work = torch.nn.functional.pad(work, (0, padded - work.shape[-1]))
    blocks = work.reshape(*work.shape[:-1], padded // block_size, block_size)
    levels = (1 << int(bits)) - 1
    if symmetric:
        qmax = (1 << (int(bits) - 1)) - 1
        scale = _symmetric_scales(blocks, qmax, scale_precision).float()
        codes = torch.round(blocks / scale)
        low, high = -qmax, qmax
    else:
        scale, zero = _asymmetric_scale_and_zero_point(
            blocks, 0, levels, scale_precision
        )
        scale = scale.float()
        zero = zero.float()
        codes = torch.round(blocks / scale + zero)
        low, high = 0, levels
    overflow_mask = (codes < low) | (codes > high)
    endpoint_mask = (codes <= low) | (codes >= high)
    excess = torch.maximum((low - codes).clamp_min(0), (codes - high).clamp_min(0))
    overflow = int(overflow_mask.sum().item())
    endpoint = int(endpoint_mask.sum().item())
    excess_sum = float(excess.sum().item())
    excess_max = float(excess.max().item())
    diagnostics["overflow"] = int(diagnostics.get("overflow", 0)) + overflow
    diagnostics["endpoint"] = int(diagnostics.get("endpoint", 0)) + endpoint
    diagnostics["overflow_code_excess_sum"] = float(
        diagnostics.get("overflow_code_excess_sum", 0.0)
    ) + excess_sum
    diagnostics["max_code_excess"] = max(
        float(diagnostics.get("max_code_excess", 0.0)), excess_max
    )
    if role is not None:
        role_stats = diagnostics.setdefault("by_role", {}).setdefault(
            role,
            {
                "values": 0,
                "nonfinite": 0,
                "overflow": 0,
                "endpoint": 0,
                "overflow_code_excess_sum": 0.0,
                "max_code_excess": 0.0,
            },
        )
        role_stats["values"] += int(tensor.numel())
        role_stats["nonfinite"] += int((~torch.isfinite(tensor.float())).sum().item())
        role_stats["overflow"] += overflow
        role_stats["endpoint"] += endpoint
        role_stats["overflow_code_excess_sum"] += excess_sum
        role_stats["max_code_excess"] = max(role_stats["max_code_excess"], excess_max)


def _quantize_reconstruct(
    tensor: torch.Tensor,
    bits: int,
    block_size: int,
    scale_precision: torch.dtype,
    *,
    symmetric: bool,
) -> tuple[torch.Tensor, int]:
    if tensor.shape[-1] % block_size != 0:
        raise ValueError(
            f"S2++ block_size={block_size} must divide head_dim={tensor.shape[-1]}"
        )
    if symmetric:
        quantized, scales = _quantize_blockwise(
            tensor, bits, block_size, scale_precision
        )
        reconstruction = _dequantize_blockwise(
            quantized,
            scales,
            bits,
            block_size,
            tensor.shape[-1],
            tensor.dtype,
        )
        state = {"quantized": quantized, "scales": scales}
    else:
        quantized, scales, zero_points = _quantize_blockwise_asymmetric(
            tensor, bits, block_size, scale_precision
        )
        reconstruction = _dequantize_blockwise_asymmetric(
            quantized,
            scales,
            zero_points,
            bits,
            block_size,
            tensor.shape[-1],
            tensor.dtype,
        )
        state = {"quantized": quantized, "scales": scales, "zero_points": zero_points}
    return reconstruction, trq_state_nbytes(state)


def _increase_fraction(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return sum(current > previous for previous, current in zip(values, values[1:])) / (
        len(values) - 1
    )
