"""Shared metrics for identity-codec and online-snapshot parity experiments."""

from __future__ import annotations

from contextlib import contextmanager
import math
import os
from typing import Any, Iterator

import torch

from ..real.trq import trq_dequantize_tensor, trq_quantize_tensor, trq_state_nbytes


def tensor_error_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | int]:
    """Return stable reconstruction metrics without moving full tensors to CPU."""
    if tuple(reference.shape) != tuple(actual.shape):
        raise ValueError(
            f"tensor shapes differ: reference={tuple(reference.shape)}, actual={tuple(actual.shape)}"
        )
    ref = reference.detach().float()
    out = actual.detach().to(device=reference.device).float()
    error = out - ref
    ref_sse = float(ref.square().sum().item())
    error_sse = float(error.square().sum().item())
    ref_norm = math.sqrt(max(ref_sse, 0.0))
    actual_norm = float(out.norm().item())
    dot = float((ref * out).sum().item())
    denominator = max(ref_norm * actual_norm, 1e-30)
    return {
        "num_values": int(ref.numel()),
        "rel_l2": math.sqrt(error_sse) / max(ref_norm, 1e-30),
        "nmse": error_sse / max(ref_sse, 1e-30),
        "cosine": dot / denominator,
        "max_abs": float(error.abs().max().item()) if error.numel() else 0.0,
        "mean_abs": float(error.abs().mean().item()) if error.numel() else 0.0,
    }


def state_metadata(state: dict[str, Any]) -> dict[str, Any]:
    """Extract comparable codec metadata while excluding tensor payloads."""
    predictor = state.get("predictor")
    predictor_kind = (
        predictor.get("kind") if isinstance(predictor, dict) else None
    ) or state.get("predictor_kind") or state.get("predictor_mode") or state.get("mode")
    return {
        "format": state.get("format") or state.get("method"),
        "version": state.get("version"),
        "shape": [int(value) for value in state.get("shape", ())],
        "unit_lengths": [int(value) for value in state.get("unit_lengths", ())],
        "num_bits": _optional_int(state.get("num_bits")),
        "anchor_bits": _optional_int(state.get("anchor_bits")),
        "block_size": _optional_int(state.get("block_size")),
        "predictor_stride": _optional_int(state.get("predictor_stride")),
        "predictor_kind": predictor_kind,
        "scale_precision": state.get("scale_precision"),
        "residual_quant_mode": state.get("residual_quant_mode"),
        "state_bytes": int(trq_state_nbytes(state)),
    }


def compare_identity_codecs(
    tensor: torch.Tensor,
    *,
    student_module,
    num_bits: int,
    anchor_bits: int,
    block_size: int,
    predictor_stride: int,
    layer_idx: int | None = None,
    scale_precision: torch.dtype = torch.bfloat16,
    codec_dtype: torch.dtype = torch.bfloat16,
    run_student_triton: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run TRQ and the external QVG S2++ identity codec on one BHSD tensor."""
    codec_input = tensor.to(dtype=codec_dtype)
    trq_state, trq_encoder = trq_quantize_tensor(
        codec_input,
        num_bits=num_bits,
        block_size=block_size,
        anchor_bits=anchor_bits,
        predictor_stride=predictor_stride,
        predictor_mode="identity",
        layer_idx=layer_idx,
        scale_precision=scale_precision,
        residual_quant_mode="asym_zero_point",
        return_reconstruction=True,
    )
    trq_decoder = trq_dequantize_tensor(trq_state, output_dtype=codec_dtype)

    student_state = student_module.s2pp_quantize_tensor(
        codec_input,
        num_bits=num_bits,
        block_size=block_size,
        anchor_bits=anchor_bits,
        predictor_stride=predictor_stride,
        affine_path="",
        mode="identity",
        scale_precision=scale_precision,
        residual_quant_mode="asym_zero_point",
        layer_idx=layer_idx,
    )
    with temporary_environment(S2PP_USE_TRITON="0"):
        student_torch = student_module.s2pp_dequantize_tensor(
            student_state, output_dtype=codec_dtype
        )
    student_via_trq = trq_dequantize_tensor(student_state, output_dtype=codec_dtype)

    student_triton = None
    if run_student_triton:
        if codec_input.device.type != "cuda":
            raise ValueError("student Triton parity requires a CUDA device")
        with temporary_environment(S2PP_USE_TRITON="1", S2PP_TRITON_STRICT="1"):
            student_triton = student_module.s2pp_dequantize_tensor(
                student_state, output_dtype=codec_dtype
            )

    trq_meta = state_metadata(trq_state)
    student_meta = state_metadata(student_state)
    summary: dict[str, Any] = {
        "shape": list(codec_input.shape),
        "config": {
            "num_bits": int(num_bits),
            "anchor_bits": int(anchor_bits),
            "block_size": int(block_size),
            "predictor_stride": int(predictor_stride),
            "predictor_mode": "identity",
            "scale_precision": str(scale_precision).replace("torch.", ""),
            "codec_dtype": str(codec_dtype).replace("torch.", ""),
        },
        "trq_state": trq_meta,
        "student_state": student_meta,
        "state_unit_lengths_match": trq_meta["unit_lengths"] == student_meta["unit_lengths"],
        "trq_vs_raw": tensor_error_metrics(codec_input, trq_decoder),
        "student_torch_vs_raw": tensor_error_metrics(codec_input, student_torch),
        "trq_encoder_decoder_parity": tensor_error_metrics(trq_encoder, trq_decoder),
        "student_torch_vs_trq_compat_decoder": tensor_error_metrics(
            student_torch, student_via_trq
        ),
        "trq_vs_student_torch": tensor_error_metrics(trq_decoder, student_torch),
    }
    if student_triton is not None:
        summary["student_torch_vs_triton"] = tensor_error_metrics(
            student_torch, student_triton
        )
        summary["student_triton_vs_raw"] = tensor_error_metrics(
            codec_input, student_triton
        )

    unit_rows: list[dict[str, Any]] = []
    boundaries = _unit_boundaries(trq_meta["unit_lengths"])
    if trq_meta["unit_lengths"] == student_meta["unit_lengths"]:
        for unit_index, (start, end) in enumerate(boundaries):
            unit_rows.append(
                {
                    "unit_index": unit_index,
                    "start_token": start,
                    "end_token": end,
                    "unit_length": end - start,
                    "is_anchor": unit_index == 0,
                    **_prefixed_metrics(
                        "trq_vs_raw",
                        tensor_error_metrics(
                            codec_input[:, :, start:end, :], trq_decoder[:, :, start:end, :]
                        ),
                    ),
                    **_prefixed_metrics(
                        "student_vs_raw",
                        tensor_error_metrics(
                            codec_input[:, :, start:end, :], student_torch[:, :, start:end, :]
                        ),
                    ),
                    **_prefixed_metrics(
                        "trq_vs_student",
                        tensor_error_metrics(
                            trq_decoder[:, :, start:end, :], student_torch[:, :, start:end, :]
                        ),
                    ),
                }
            )
    return summary, unit_rows


@contextmanager
def temporary_environment(**updates: str) -> Iterator[None]:
    old_values = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _unit_boundaries(lengths: list[int]) -> list[tuple[int, int]]:
    result = []
    start = 0
    for length in lengths:
        end = start + int(length)
        result.append((start, end))
        start = end
    return result


def _prefixed_metrics(prefix: str, metrics: dict[str, float | int]) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
