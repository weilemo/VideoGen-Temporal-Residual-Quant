"""Compatibility adapter for stable S2++ identity/affine call sites."""

from __future__ import annotations

import numpy as np
import torch

from .trq import trq_dequantize_tensor, trq_quantize_tensor


def s2pp_quantize_tensor(
    tensor: torch.Tensor,
    num_bits: int,
    block_size: int,
    anchor_bits: int = 4,
    predictor_stride: int = 1560,
    affine_path: str = "",
    mode: str = "affine",
    scale_precision: torch.dtype | str = torch.bfloat16,
    residual_quant_mode: str = "asym_zero_point",
    layer_idx: int | None = None,
    use_lloyd_max: bool = False,
    source_tensor: torch.Tensor | None = None,
    source_state: dict | None = None,
    **unsupported,
) -> dict:
    if use_lloyd_max or source_tensor is not None or source_state is not None:
        raise NotImplementedError("Lloyd-Max and Cross-KV remain experimental and are not part of TRQ v1")
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise NotImplementedError(f"S2++ experimental options are not supported by TRQ v1: {names}")
    params = None
    if mode not in {"identity"}:
        if not affine_path:
            raise ValueError("S2++ affine compatibility mode requires affine_path")
        with np.load(affine_path, allow_pickle=False) as data:
            if "alpha" not in data or "beta" not in data:
                raise ValueError(f"S2++ affine file must contain alpha and beta: {affine_path}")
            params = {
                "alpha": torch.from_numpy(data["alpha"].astype(np.float32)),
                "beta": torch.from_numpy(data["beta"].astype(np.float32)),
            }
    return trq_quantize_tensor(
        tensor,
        num_bits=num_bits,
        block_size=block_size,
        anchor_bits=anchor_bits,
        predictor_stride=predictor_stride,
        predictor_mode=mode,
        predictor_params=params,
        layer_idx=layer_idx,
        scale_precision=scale_precision,
        residual_quant_mode=residual_quant_mode,
    )


def s2pp_dequantize_tensor(
    packed_state: dict,
    output_dtype: torch.dtype = torch.bfloat16,
    **unsupported,
) -> torch.Tensor:
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise NotImplementedError(f"S2++ experimental decode options are not supported by TRQ v1: {names}")
    return trq_dequantize_tensor(packed_state, output_dtype=output_dtype)


__all__ = ["s2pp_quantize_tensor", "s2pp_dequantize_tensor"]
