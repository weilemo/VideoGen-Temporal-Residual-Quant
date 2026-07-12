from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import torch

from ..real.trq import (
    _dequantize_asym,
    _predict,
    _quantize_asym,
    _resolve_scale_precision,
    trq_dequantize_tensor,
    trq_quantize_tensor,
    trq_state_nbytes,
)


ValueCallback = Callable[[str, torch.Tensor], None]


@dataclass
class DistributionAccumulator:
    """Exact moments plus a bounded, approximately uniform plotting sample."""

    sample_capacity: int = 100_000
    seed: int = 0

    def __post_init__(self) -> None:
        self.count = 0
        self.value_sum = 0.0
        self.square_sum = 0.0
        self.abs_sum = 0.0
        self.max_abs = 0.0
        self._samples = np.empty(0, dtype=np.float32)
        self._rng = np.random.default_rng(self.seed)

    @property
    def samples(self) -> np.ndarray:
        return self._samples.copy()

    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().float().reshape(-1).cpu()
        batch_count = int(flat.numel())
        if batch_count == 0:
            return

        old_count = self.count
        self.count += batch_count
        work64 = flat.to(torch.float64)
        self.value_sum += float(work64.sum().item())
        self.square_sum += float(work64.square().sum().item())
        self.abs_sum += float(work64.abs().sum().item())
        self.max_abs = max(self.max_abs, float(work64.abs().max().item()))

        if self.sample_capacity <= 0:
            return
        array = flat.numpy()
        if self.count <= self.sample_capacity:
            self._samples = np.concatenate((self._samples, array.astype(np.float32, copy=False)))
            return

        new_slots = int(round(self.sample_capacity * batch_count / self.count))
        new_slots = min(batch_count, max(1, new_slots))
        old_slots = min(self._samples.size, self.sample_capacity - new_slots)
        new_slots = min(batch_count, self.sample_capacity - old_slots)

        if old_slots:
            old_idx = self._rng.choice(self._samples.size, size=old_slots, replace=False)
            old_sample = self._samples[old_idx]
        else:
            old_sample = np.empty(0, dtype=np.float32)
        new_idx = self._rng.choice(batch_count, size=new_slots, replace=False)
        new_sample = array[new_idx].astype(np.float32, copy=False)
        self._samples = np.concatenate((old_sample, new_sample))

    def summary(self) -> dict:
        if self.count == 0:
            return {
                "count": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "rms": float("nan"),
                "mean_abs": float("nan"),
                "max_abs": float("nan"),
                "p50_abs": float("nan"),
                "p90_abs": float("nan"),
                "p95_abs": float("nan"),
                "p99_abs": float("nan"),
            }
        mean = self.value_sum / self.count
        variance = max(0.0, self.square_sum / self.count - mean * mean)
        abs_samples = np.abs(self._samples.astype(np.float64, copy=False))
        quantiles = (
            np.quantile(abs_samples, [0.50, 0.90, 0.95, 0.99])
            if abs_samples.size
            else [float("nan")] * 4
        )
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "rms": math.sqrt(max(0.0, self.square_sum / self.count)),
            "mean_abs": self.abs_sum / self.count,
            "max_abs": self.max_abs,
            "p50_abs": float(quantiles[0]),
            "p90_abs": float(quantiles[1]),
            "p95_abs": float(quantiles[2]),
            "p99_abs": float(quantiles[3]),
        }


def trace_temporal_residual_quantization(
    tensor: torch.Tensor,
    *,
    num_bits: int = 2,
    block_size: int = 64,
    anchor_bits: int = 4,
    predictor_stride: int = 1560,
    predictor_mode: str = "identity",
    predictor_params: dict | None = None,
    layer_idx: int | None = None,
    scale_precision: torch.dtype | str = torch.bfloat16,
    codec_dtype: torch.dtype = torch.bfloat16,
    value_callback: ValueCallback | None = None,
    reset_interval_units: int | None = None,
    span_unit_lengths: list[int] | tuple[int, ...] | None = None,
    verify_decoder: bool = True,
) -> list[dict]:
    """Trace a rollout, resetting the residual chain at explicit span boundaries."""
    if tensor.ndim != 4:
        raise ValueError(f"expected BHSD tensor, got {tuple(tensor.shape)}")
    if predictor_stride <= 0:
        raise ValueError("predictor_stride must be positive")
    total_units = math.ceil(tensor.shape[2] / predictor_stride)
    if span_unit_lengths is not None and reset_interval_units is not None:
        raise ValueError("span_unit_lengths and reset_interval_units are mutually exclusive")
    if span_unit_lengths is not None:
        lengths = [int(value) for value in span_unit_lengths]
        if not lengths or any(value <= 0 for value in lengths):
            raise ValueError("span_unit_lengths must contain positive values")
        if sum(lengths) > total_units:
            raise ValueError(
                f"span_unit_lengths cover {sum(lengths)} units, but tensor has {total_units}"
            )
        spans = []
        first_unit = 0
        for length in lengths:
            spans.append((first_unit, first_unit + length))
            first_unit += length
    else:
        interval = total_units if reset_interval_units is None else int(reset_interval_units)
        if interval <= 0:
            raise ValueError("reset_interval_units must be positive or None")
        spans = [
            (first_unit, min(total_units, first_unit + interval))
            for first_unit in range(0, total_units, interval)
        ]

    records: list[dict] = []
    for span_id, (first_unit, end_unit) in enumerate(spans):
        span_start = first_unit * predictor_stride
        span_end = min(tensor.shape[2], end_unit * predictor_stride)
        span_records = _trace_single_span(
            tensor[:, :, span_start:span_end, :],
            num_bits=num_bits,
            block_size=block_size,
            anchor_bits=anchor_bits,
            predictor_stride=predictor_stride,
            predictor_mode=predictor_mode,
            predictor_params=predictor_params,
            layer_idx=layer_idx,
            scale_precision=scale_precision,
            codec_dtype=codec_dtype,
            value_callback=value_callback,
            verify_decoder=verify_decoder,
        )
        for local_record in span_records:
            chain_position = int(local_record["step"])
            rollout_unit = first_unit + chain_position
            local_record["span_id"] = span_id
            local_record["chain_position"] = chain_position
            local_record["rollout_unit"] = rollout_unit
            local_record["step"] = rollout_unit
            local_record["start_token"] += span_start
            local_record["end_token"] += span_start
            records.append(local_record)
    return records


def _trace_single_span(
    tensor: torch.Tensor,
    *,
    num_bits: int,
    block_size: int,
    anchor_bits: int,
    predictor_stride: int,
    predictor_mode: str,
    predictor_params: dict | None,
    layer_idx: int | None,
    scale_precision: torch.dtype | str,
    codec_dtype: torch.dtype,
    value_callback: ValueCallback | None,
    verify_decoder: bool,
) -> list[dict]:
    """Trace raw-direct, teacher-forced, and chained quantization per unit.

    ``tensor`` is BHSD. Raw and residual distributions are sampled from the
    same non-anchor target units. The chained reconstruction is the actual TRQ
    encoder reconstruction, while teacher forcing predicts from exact previous
    KV and isolates local residual quantization error.
    """
    if tensor.shape[2] <= 0:
        raise ValueError("sequence length must be positive")

    codec_input = tensor.to(dtype=codec_dtype)
    state, chained_full = trq_quantize_tensor(
        codec_input,
        num_bits=num_bits,
        block_size=block_size,
        anchor_bits=anchor_bits,
        predictor_stride=predictor_stride,
        predictor_mode=predictor_mode,
        predictor_params=predictor_params,
        layer_idx=layer_idx,
        scale_precision=scale_precision,
        return_reconstruction=True,
    )
    predictor = state["predictor"]
    state_nbytes = trq_state_nbytes(state)
    effective_bits = 8.0 * state_nbytes / codec_input.numel()
    if verify_decoder:
        decoder_full = trq_dequantize_tensor(state, output_dtype=codec_dtype)
        decoder_parity_max_abs = float(
            (decoder_full.float() - chained_full.float()).abs().max().item()
        )
    else:
        decoder_parity_max_abs = float("nan")
    padded_dim = int(state["padded_dim"])
    scale_dtype = _resolve_scale_precision(scale_precision)
    unit_lengths = [int(value) for value in state["unit_lengths"]]

    records: list[dict] = []
    current_start = 0
    previous_start = 0
    previous_chain_error = None
    for step, unit_len in enumerate(unit_lengths):
        current = codec_input[:, :, current_start:current_start + unit_len, :]
        chained_current = chained_full[:, :, current_start:current_start + unit_len, :]
        raw_recon = _asymmetric_roundtrip(
            current,
            bits=num_bits,
            block_size=block_size,
            padded_dim=padded_dim,
            scale_precision=scale_dtype,
            output_dtype=codec_dtype,
        )

        if step == 0:
            teacher_current = chained_current
            chain_residual = None
            reference_residual = None
            parity_max_abs = 0.0
            prediction_perturbation_sse = float("nan")
            residual_inflation_ratio = float("nan")
        else:
            previous_exact = codec_input[:, :, previous_start:previous_start + unit_len, :]
            previous_chained = chained_full[:, :, previous_start:previous_start + unit_len, :]
            chain_prediction = _predict(previous_chained, predictor)
            teacher_prediction = _predict(previous_exact, predictor)
            chain_residual = current.float() - chain_prediction
            reference_residual = current.float() - teacher_prediction
            chain_residual_recon = _asymmetric_roundtrip(
                chain_residual,
                bits=num_bits,
                block_size=block_size,
                padded_dim=padded_dim,
                scale_precision=scale_dtype,
                output_dtype=torch.float32,
            )
            teacher_residual_recon = _asymmetric_roundtrip(
                reference_residual,
                bits=num_bits,
                block_size=block_size,
                padded_dim=padded_dim,
                scale_precision=scale_dtype,
                output_dtype=torch.float32,
            )
            recomputed_chain = (chain_prediction + chain_residual_recon).to(codec_dtype)
            teacher_current = (teacher_prediction + teacher_residual_recon).to(codec_dtype)
            parity_max_abs = float(
                (recomputed_chain.float() - chained_current.float()).abs().max().item()
            )
            prediction_perturbation_sse = _sum_squares(chain_prediction - teacher_prediction)
            residual_inflation_ratio = math.sqrt(
                _sum_squares(chain_residual) / max(_sum_squares(reference_residual), 1e-30)
            )
            if value_callback is not None:
                value_callback("raw", current.float())
                value_callback("residual", chain_residual)
                value_callback("reference_residual", reference_residual)

        target = current.float()
        target_sse = _sum_squares(target)
        chain_error = chained_current.float() - target
        if previous_chain_error is None:
            error_cosine_with_previous = float("nan")
        else:
            previous_slice = previous_chain_error[:, :, :unit_len, :]
            error_cosine_with_previous = _cosine(chain_error, previous_slice)
        record = {
            "step": step,
            "start_token": current_start,
            "end_token": current_start + unit_len,
            "unit_length": unit_len,
            "num_values": int(target.numel()),
            "target_sse": target_sse,
            "raw_error_sse": _sum_squares(raw_recon.float() - target),
            "chain_error_sse": _sum_squares(chain_error),
            "teacher_error_sse": _sum_squares(teacher_current.float() - target),
            "chain_teacher_delta_sse": _sum_squares(
                chained_current.float() - teacher_current.float()
            ),
            "codec_parity_max_abs": parity_max_abs,
            "decoder_parity_max_abs": decoder_parity_max_abs,
            "prediction_perturbation_sse": prediction_perturbation_sse,
            "prediction_perturbation_rel_l2": (
                math.sqrt(prediction_perturbation_sse / max(target_sse, 1e-30))
                if math.isfinite(prediction_perturbation_sse)
                else float("nan")
            ),
            "residual_inflation_ratio": residual_inflation_ratio,
            "error_cosine_with_previous": error_cosine_with_previous,
            "span_state_nbytes": state_nbytes if step == 0 else 0,
            "span_effective_bits_per_value": effective_bits,
            "raw_rel_l2": _relative_l2(raw_recon.float() - target, target),
            "chain_rel_l2": _relative_l2(chained_current.float() - target, target),
            "teacher_rel_l2": _relative_l2(teacher_current.float() - target, target),
        }
        if chain_residual is not None and reference_residual is not None:
            record["residual_sse"] = _sum_squares(chain_residual)
            record["reference_residual_sse"] = _sum_squares(reference_residual)
            record["residual_rms_ratio"] = math.sqrt(record["residual_sse"] / max(target_sse, 1e-30))
            record["reference_residual_rms_ratio"] = math.sqrt(
                record["reference_residual_sse"] / max(target_sse, 1e-30)
            )
        else:
            record["residual_sse"] = float("nan")
            record["reference_residual_sse"] = float("nan")
            record["residual_rms_ratio"] = float("nan")
            record["reference_residual_rms_ratio"] = float("nan")
        records.append(record)

        previous_chain_error = chain_error
        previous_start = current_start
        current_start += unit_len

    return records


def aggregate_error_records(records: list[dict]) -> dict:
    if not records:
        return {"count": 0}
    target_sse = sum(float(record["target_sse"]) for record in records)
    result = {
        "count": len(records),
        "num_values": sum(int(record["num_values"]) for record in records),
        "target_sse": target_sse,
    }
    for prefix in ("raw", "chain", "teacher"):
        error_sse = sum(float(record[f"{prefix}_error_sse"]) for record in records)
        result[f"{prefix}_error_sse"] = error_sse
        result[f"{prefix}_rel_l2"] = math.sqrt(error_sse / max(target_sse, 1e-30))
        result[f"{prefix}_sqnr_db"] = _sqnr_db(target_sse, error_sse)
    delta_sse = sum(float(record["chain_teacher_delta_sse"]) for record in records)
    result["chain_teacher_delta_sse"] = delta_sse
    result["chain_teacher_delta_rel_l2"] = math.sqrt(delta_sse / max(target_sse, 1e-30))
    result["chain_over_teacher_error_ratio"] = (
        result["chain_error_sse"] / max(result["teacher_error_sse"], 1e-30)
    )
    result["residual_over_raw_error_ratio"] = (
        result["chain_error_sse"] / max(result["raw_error_sse"], 1e-30)
    )
    result["raw_over_residual_error_gain"] = (
        result["raw_error_sse"] / max(result["chain_error_sse"], 1e-30)
    )
    result["codec_parity_max_abs"] = max(float(record["codec_parity_max_abs"]) for record in records)
    finite_decoder_parity = [
        float(record["decoder_parity_max_abs"])
        for record in records
        if math.isfinite(float(record["decoder_parity_max_abs"]))
    ]
    result["decoder_parity_max_abs"] = max(finite_decoder_parity, default=float("nan"))
    result["state_nbytes"] = sum(int(record.get("span_state_nbytes", 0)) for record in records)
    return result


def _asymmetric_roundtrip(
    tensor: torch.Tensor,
    *,
    bits: int,
    block_size: int,
    padded_dim: int,
    scale_precision: torch.dtype,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    q, scales, zero_points = _quantize_asym(
        tensor,
        bits,
        block_size,
        padded_dim,
        scale_precision,
    )
    return _dequantize_asym(
        q,
        scales,
        zero_points,
        bits,
        block_size,
        tensor.shape[-1],
        padded_dim,
        output_dtype,
    )


def _sum_squares(tensor: torch.Tensor) -> float:
    return float(tensor.double().square().sum().item())


def _relative_l2(error: torch.Tensor, target: torch.Tensor) -> float:
    return math.sqrt(_sum_squares(error) / max(_sum_squares(target), 1e-30))


def _sqnr_db(signal_sse: float, error_sse: float) -> float:
    if error_sse <= 0:
        return float("inf")
    return 10.0 * math.log10(max(signal_sse, 1e-30) / error_sse)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.double().reshape(-1)
    right_flat = right.double().reshape(-1)
    denominator = left_flat.norm() * right_flat.norm()
    if denominator.item() <= 0:
        return float("nan")
    return float(torch.dot(left_flat, right_flat).item() / denominator.item())
