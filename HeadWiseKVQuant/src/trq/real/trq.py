from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import torch


TRQ_FORMAT = "trq"
TRQ_VERSION = 1
SUPPORTED_BITS = (2, 4, 8)
STABLE_PREDICTORS = ("identity", "affine_channel")


def extract_trq_bits(quant_type: str) -> int:
    match = re.search(r"int(\d+)", quant_type)
    if match is None:
        raise ValueError(f"Cannot identify num_bits from {quant_type}")
    return _validate_bits(int(match.group(1)))


def normalize_trq_predictor_mode(mode: str) -> str:
    """Return a stable v1 predictor name or reject experimental modes."""
    return _normalize_predictor_mode(mode)


def trq_quantize_tensor(
    tensor: torch.Tensor,
    *,
    num_bits: int,
    block_size: int,
    anchor_bits: int = 4,
    predictor_stride: int = 1560,
    predictor_mode: str = "identity",
    predictor_params: dict | None = None,
    layer_idx: int | None = None,
    scale_precision: torch.dtype | str = torch.bfloat16,
    residual_quant_mode: str = "asym_zero_point",
    return_reconstruction: bool = False,
) -> dict | tuple[dict, torch.Tensor]:
    """Encode a BHSD tensor with temporal predictive residual quantization.

    TRQ v1 is deliberately small: it supports identity and per-channel affine
    prediction, and asymmetric packed residuals. Predictor parameters required
    by decoding are embedded in the state so the codec is self-contained.
    """
    if tensor.ndim != 4:
        raise ValueError(f"TRQ expects [B, H, S, D], got {tuple(tensor.shape)}")
    if tensor.shape[2] <= 0 or tensor.shape[3] <= 0:
        raise ValueError(f"TRQ requires non-empty sequence and head dimensions, got {tuple(tensor.shape)}")

    num_bits = _validate_bits(num_bits)
    anchor_bits = _validate_bits(anchor_bits)
    block_size = _validate_block_size(block_size)
    predictor_stride = _validate_stride(predictor_stride, tensor.shape[2])
    predictor_mode = _normalize_predictor_mode(predictor_mode)
    residual_quant_mode = _validate_residual_quant_mode(residual_quant_mode)
    scale_precision = _resolve_scale_precision(scale_precision)

    bsz, heads, seq_len, head_dim = tensor.shape
    alignment = math.lcm(block_size, 8 // num_bits, 8 // anchor_bits)
    padded_dim = math.ceil(head_dim / alignment) * alignment
    predictor = _build_predictor_state(
        predictor_mode,
        predictor_params,
        heads=heads,
        head_dim=head_dim,
        layer_idx=layer_idx,
        device=tensor.device,
    )

    anchor_len = min(predictor_stride, seq_len)
    anchor_q, anchor_scales = _quantize_sym(
        tensor[:, :, :anchor_len, :], anchor_bits, block_size, padded_dim, scale_precision
    )
    previous = _dequantize_sym(
        anchor_q,
        anchor_scales,
        anchor_bits,
        block_size,
        head_dim,
        padded_dim,
        tensor.dtype,
    )

    residual_qs: list[torch.Tensor] = []
    residual_scales: list[torch.Tensor] = []
    residual_zps: list[torch.Tensor] = []
    reconstruction_parts = [previous]
    unit_lengths = [anchor_len]
    offset = anchor_len

    while offset < seq_len:
        unit_len = min(predictor_stride, seq_len - offset)
        current = tensor[:, :, offset:offset + unit_len, :]
        prediction = _predict(previous[:, :, :unit_len, :], predictor)
        residual = current.float() - prediction
        q, scales, zero_points = _quantize_asym(
            residual, num_bits, block_size, padded_dim, scale_precision
        )
        residual_recon = _dequantize_asym(
            q,
            scales,
            zero_points,
            num_bits,
            block_size,
            head_dim,
            padded_dim,
            torch.float32,
        )
        previous = (prediction + residual_recon).to(tensor.dtype)
        reconstruction_parts.append(previous)
        residual_qs.append(q)
        residual_scales.append(scales)
        residual_zps.append(zero_points)
        unit_lengths.append(unit_len)
        offset += unit_len

    state = {
        "format": TRQ_FORMAT,
        "version": TRQ_VERSION,
        "layout": "BHSD",
        "shape": (int(bsz), int(heads), int(seq_len), int(head_dim)),
        "padded_dim": int(padded_dim),
        "anchor_quant": anchor_q,
        "anchor_scales": anchor_scales,
        "residual_quant": torch.cat(residual_qs, dim=2).contiguous() if residual_qs else None,
        "residual_scales": torch.cat(residual_scales, dim=2).contiguous() if residual_scales else None,
        "residual_zero_points": torch.cat(residual_zps, dim=2).contiguous() if residual_zps else None,
        "unit_lengths": unit_lengths,
        "num_bits": int(num_bits),
        "anchor_bits": int(anchor_bits),
        "block_size": int(block_size),
        "predictor_stride": int(predictor_stride),
        "predictor": predictor,
        "predictor_mode": predictor["kind"],
        "dependency": "previous_reconstruction",
        "scale_precision": str(scale_precision).replace("torch.", ""),
        "residual_quant_mode": residual_quant_mode,
    }
    if layer_idx is not None:
        state["layer_idx"] = int(layer_idx)

    if return_reconstruction:
        return state, torch.cat(reconstruction_parts, dim=2).reshape(tensor.shape)
    return state


def trq_dequantize_tensor(
    packed_state: dict,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    predictor_params: dict | None = None,
) -> torch.Tensor:
    """Decode TRQ v1 plus stable legacy HRQ/S2++ states."""
    state = _coerce_legacy_state(packed_state, predictor_params)
    _validate_state(state)

    bsz, heads, seq_len, head_dim = [int(value) for value in state["shape"]]
    padded_dim = int(state.get("padded_dim", head_dim))
    block_size = int(state["block_size"])
    anchor_bits = int(state["anchor_bits"])
    num_bits = int(state["num_bits"])
    unit_lengths = [int(value) for value in state["unit_lengths"]]
    predictor = state["predictor"]

    anchor = _dequantize_sym(
        state["anchor_quant"],
        state["anchor_scales"],
        anchor_bits,
        block_size,
        head_dim,
        padded_dim,
        output_dtype,
    )
    outputs = [anchor]
    previous = anchor
    q_all = state.get("residual_quant")
    scales_all = state.get("residual_scales")
    zps_all = state.get("residual_zero_points")
    residual_offset = 0

    for unit_len in unit_lengths[1:]:
        q = q_all[:, :, residual_offset:residual_offset + unit_len, :]
        scales = scales_all[:, :, residual_offset:residual_offset + unit_len, :]
        zero_points = zps_all[:, :, residual_offset:residual_offset + unit_len, :]
        residual_offset += unit_len
        prediction = _predict(previous[:, :, :unit_len, :], predictor)
        residual = _dequantize_asym(
            q,
            scales,
            zero_points,
            num_bits,
            block_size,
            head_dim,
            padded_dim,
            torch.float32,
        )
        current = (prediction + residual).to(output_dtype)
        outputs.append(current)
        previous = current

    return torch.cat(outputs, dim=2).reshape(bsz, heads, seq_len, head_dim)


def trq_state_nbytes(state: Any) -> int:
    """Count bytes physically held by tensors in an encoded state."""
    if isinstance(state, torch.Tensor):
        return state.numel() * state.element_size()
    if isinstance(state, dict):
        return sum(trq_state_nbytes(value) for value in state.values())
    if isinstance(state, (list, tuple)):
        return sum(trq_state_nbytes(value) for value in state)
    return 0


def _build_predictor_state(
    mode: str,
    params: dict | None,
    *,
    heads: int,
    head_dim: int,
    layer_idx: int | None,
    device: torch.device,
) -> dict:
    if mode == "identity":
        return {"kind": "identity", "version": 1, "id": "identity:v1", "params": {}}
    if params is None:
        raise ValueError("TRQ affine_channel predictor requires alpha and beta parameters")
    if "alpha" not in params or "beta" not in params:
        raise ValueError("TRQ affine_channel predictor requires both alpha and beta")

    alpha = _select_affine_param(params["alpha"], "alpha", heads, head_dim, layer_idx, device)
    beta = _select_affine_param(params["beta"], "beta", heads, head_dim, layer_idx, device)
    if alpha.shape != beta.shape:
        raise ValueError(f"TRQ affine alpha/beta shapes differ: {tuple(alpha.shape)} vs {tuple(beta.shape)}")
    predictor_id = _predictor_id(alpha, beta)
    return {
        "kind": "affine_channel",
        "version": 1,
        "id": predictor_id,
        "params": {"alpha": alpha, "beta": beta},
    }


def _select_affine_param(value, name, heads, head_dim, layer_idx, device):
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device).detach().contiguous()
    if tensor.ndim == 3:
        if layer_idx is None:
            raise ValueError(f"TRQ {name} has shape {tuple(tensor.shape)}; layer_idx is required")
        if not 0 <= int(layer_idx) < tensor.shape[0]:
            raise ValueError(f"TRQ {name} layer_idx={layer_idx} is outside [0, {tensor.shape[0] - 1}]")
        tensor = tensor[int(layer_idx)].contiguous()
    if tensor.ndim == 1:
        if tensor.shape[0] != head_dim:
            raise ValueError(f"TRQ {name} must end in head_dim={head_dim}, got {tuple(tensor.shape)}")
        return tensor
    if tensor.ndim == 2:
        if tuple(tensor.shape) != (heads, head_dim):
            raise ValueError(f"TRQ {name} must be [D] or [H,D]={heads, head_dim}, got {tuple(tensor.shape)}")
        return tensor
    raise ValueError(f"TRQ {name} must be [D], [H,D], or [L,H,D], got {tuple(tensor.shape)}")


def _predictor_id(alpha: torch.Tensor, beta: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in (alpha, beta):
        cpu = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(cpu.numpy().tobytes())
    return f"affine-channel:v1:{digest.hexdigest()[:16]}"


def _predict(previous: torch.Tensor, predictor: dict) -> torch.Tensor:
    kind = predictor.get("kind")
    if kind == "identity":
        return previous.float()
    if kind != "affine_channel":
        raise ValueError(f"Unsupported TRQ predictor in encoded state: {kind!r}")
    params = predictor.get("params") or {}
    if "alpha" not in params or "beta" not in params:
        raise ValueError("TRQ affine_channel encoded state is missing alpha or beta")
    alpha = params["alpha"].to(previous.device, torch.float32)
    beta = params["beta"].to(previous.device, torch.float32)
    if alpha.ndim == 1:
        view_shape = (1, 1, 1, previous.shape[-1])
    elif alpha.ndim == 2:
        view_shape = (1, previous.shape[1], 1, previous.shape[-1])
    else:
        raise ValueError(f"TRQ affine state must contain [D] or [H,D], got {tuple(alpha.shape)}")
    return previous.float() * alpha.view(view_shape) + beta.view(view_shape)


def _coerce_legacy_state(state: dict, predictor_params: dict | None) -> dict:
    if state.get("format") == TRQ_FORMAT:
        return state

    if state.get("format") == "hrq":
        result = dict(state)
        mode = _normalize_predictor_mode(result.get("predictor_mode", "identity"))
        _, heads, _, head_dim = [int(value) for value in result["shape"]]
        result["format"] = TRQ_FORMAT
        result["version"] = TRQ_VERSION
        result["layout"] = "BHSD"
        result["dependency"] = "previous_reconstruction"
        result["predictor"] = _build_predictor_state(
            mode,
            predictor_params,
            heads=heads,
            head_dim=head_dim,
            layer_idx=result.get("layer_idx"),
            device=result["anchor_quant"].device,
        )
        return result

    if state.get("method") == "s2pp":
        kind = state.get("predictor_kind", state.get("predictor_mode", "identity"))
        mode = _normalize_predictor_mode(kind)
        result = dict(state)
        _, heads, _, head_dim = [int(value) for value in result["shape"]]
        params = predictor_params
        if params is None and mode == "affine_channel":
            params = {"alpha": state.get("alpha"), "beta": state.get("beta")}
        result["format"] = TRQ_FORMAT
        result["version"] = TRQ_VERSION
        result["layout"] = "BHSD"
        result["padded_dim"] = int(head_dim)
        result["dependency"] = "previous_reconstruction"
        result["predictor"] = _build_predictor_state(
            mode,
            params,
            heads=heads,
            head_dim=head_dim,
            layer_idx=result.get("layer_idx"),
            device=result["anchor_quant"].device,
        )
        return result

    raise ValueError(
        f"Unsupported residual codec state: format={state.get('format')!r}, method={state.get('method')!r}"
    )


def _validate_state(state: dict) -> None:
    if state.get("format") != TRQ_FORMAT or int(state.get("version", -1)) != TRQ_VERSION:
        raise ValueError(f"Unsupported TRQ state version: {state.get('format')!r} v{state.get('version')!r}")
    if state.get("layout", "BHSD") != "BHSD":
        raise ValueError(f"TRQ v1 only supports BHSD states, got {state.get('layout')!r}")
    shape = tuple(int(value) for value in state["shape"])
    if len(shape) != 4:
        raise ValueError(f"TRQ state shape must have four dimensions, got {shape}")
    unit_lengths = [int(value) for value in state["unit_lengths"]]
    if not unit_lengths or any(value <= 0 for value in unit_lengths) or sum(unit_lengths) != shape[2]:
        raise ValueError(f"TRQ unit_lengths {unit_lengths} do not cover sequence length {shape[2]}")
    _validate_bits(state["num_bits"])
    _validate_bits(state["anchor_bits"])
    block_size = _validate_block_size(state["block_size"])
    padded_dim = int(state.get("padded_dim", shape[3]))
    alignment = math.lcm(
        block_size,
        8 // int(state["num_bits"]),
        8 // int(state["anchor_bits"]),
    )
    if padded_dim < shape[3] or padded_dim % alignment != 0:
        raise ValueError(
            f"TRQ padded_dim={padded_dim} must cover D={shape[3]} and be divisible by {alignment}"
        )
    _validate_residual_quant_mode(state.get("residual_quant_mode", "asym_zero_point"))
    predictor = state.get("predictor")
    if not isinstance(predictor, dict) or predictor.get("kind") not in STABLE_PREDICTORS:
        raise ValueError(f"TRQ state has invalid predictor metadata: {predictor!r}")
    if len(unit_lengths) > 1:
        for key in ("residual_quant", "residual_scales", "residual_zero_points"):
            if state.get(key) is None:
                raise ValueError(f"TRQ state is missing {key}")


def _quantize_sym(x, bits, block_size, padded_dim, scale_precision):
    x = _pad(x.float(), padded_dim)
    blocks = padded_dim // block_size
    grouped = x.reshape(*x.shape[:-1], blocks, block_size)
    qmax = (1 << (bits - 1)) - 1
    scales = (grouped.abs().amax(dim=-1).clamp_min(1e-8) / qmax).to(scale_precision)
    q = torch.round(grouped / scales.float().unsqueeze(-1)).clamp(-qmax, qmax)
    q = q.to(torch.int16).reshape(*x.shape)
    return _pack_signed(q, bits), scales.contiguous()


def _dequantize_sym(q, scales, bits, block_size, head_dim, padded_dim, output_dtype):
    q = _unpack_signed(q, bits, padded_dim)
    scales = _expand(scales, block_size, padded_dim)
    return (q.float() * scales)[..., :head_dim].to(output_dtype)


def _quantize_asym(x, bits, block_size, padded_dim, scale_precision):
    x = _pad(x.float(), padded_dim)
    blocks = padded_dim // block_size
    grouped = x.reshape(*x.shape[:-1], blocks, block_size)
    qmax = (1 << bits) - 1
    xmin = grouped.amin(dim=-1)
    xmax = grouped.amax(dim=-1)
    value_range = xmax - xmin
    range_scale = value_range.clamp_min(1e-8) / qmax
    constant_scale = torch.maximum(xmin.abs(), xmax.abs()).clamp_min(1e-8) / qmax
    is_constant = value_range < 1e-8
    scales_float = torch.where(is_constant, constant_scale, range_scale)
    zps_float = torch.round(-xmin / scales_float).clamp(0, qmax)
    zps_float = torch.where(is_constant & (xmin < 0), torch.full_like(zps_float, qmax), zps_float)
    zps_float = torch.where(is_constant & (xmin >= 0), torch.zeros_like(zps_float), zps_float)
    scales = scales_float.to(scale_precision)
    zps = zps_float.to(torch.uint8)
    q = torch.round(grouped / scales.float().unsqueeze(-1) + zps.float().unsqueeze(-1))
    q = q.clamp(0, qmax).to(torch.uint8).reshape(*x.shape)
    return _pack_unsigned(q, bits), scales.contiguous(), zps.contiguous()


def _dequantize_asym(q, scales, zps, bits, block_size, head_dim, padded_dim, output_dtype):
    q = _unpack_unsigned(q, bits, padded_dim)
    scales = _expand(scales, block_size, padded_dim)
    zps = _expand(zps.float(), block_size, padded_dim)
    return ((q.float() - zps) * scales)[..., :head_dim].to(output_dtype)


def _pack_unsigned(q, bits):
    if bits == 8:
        return q.to(torch.uint8).contiguous()
    per_byte = 8 // bits
    grouped = q.to(torch.uint8).reshape(*q.shape[:-1], q.shape[-1] // per_byte, per_byte)
    packed = torch.zeros(*grouped.shape[:-1], device=q.device, dtype=torch.uint8)
    mask = (1 << bits) - 1
    for idx in range(per_byte):
        packed |= (grouped[..., idx] & mask) << (bits * (per_byte - 1 - idx))
    return packed.contiguous()


def _unpack_unsigned(q, bits, padded_dim):
    if bits == 8:
        return q[..., :padded_dim].to(torch.uint8)
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    parts = [
        ((q >> (bits * (per_byte - 1 - idx))) & mask).to(torch.uint8)
        for idx in range(per_byte)
    ]
    return torch.stack(parts, dim=-1).reshape(*q.shape[:-1], padded_dim)


def _pack_signed(q, bits):
    if bits == 8:
        return q.to(torch.int8).contiguous()
    qmax = (1 << (bits - 1)) - 1
    return _pack_unsigned((q + qmax).to(torch.uint8), bits)


def _unpack_signed(q, bits, padded_dim):
    if bits == 8:
        return q[..., :padded_dim].to(torch.int16)
    qmax = (1 << (bits - 1)) - 1
    return _unpack_unsigned(q, bits, padded_dim).to(torch.int16) - qmax


def _pad(x, padded_dim):
    if x.shape[-1] == padded_dim:
        return x
    return torch.nn.functional.pad(x, (0, padded_dim - x.shape[-1]), value=0)


def _expand(params, block_size, padded_dim):
    return (
        params.float()
        .unsqueeze(-1)
        .expand(*params.shape, block_size)
        .reshape(*params.shape[:-1], padded_dim)
    )


def _validate_bits(bits):
    bits = int(bits)
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"TRQ only supports {SUPPORTED_BITS}, got int{bits}")
    return bits


def _validate_block_size(block_size):
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    return block_size


def _validate_stride(stride, seq_len):
    stride = int(stride)
    if stride <= 0:
        raise ValueError(f"predictor_stride must be positive, got {stride}")
    return min(stride, int(seq_len))


def _normalize_predictor_mode(mode):
    aliases = {"affine": "affine_channel", "channel_affine": "affine_channel"}
    mode = aliases.get(str(mode), str(mode))
    if mode in {"rope", "rope_affine", "rope_rotation", "rope_rotation_affine"}:
        raise NotImplementedError("RoPE prediction is experimental and is not part of the TRQ v1 codec")
    if mode in {"tiny_mlp", "cross_kv", "block_var", "error_feedback", "ar2"}:
        raise NotImplementedError(f"Predictor {mode!r} is experimental and is not part of the TRQ v1 codec")
    if mode not in STABLE_PREDICTORS:
        raise ValueError(f"TRQ predictor_mode must be one of {STABLE_PREDICTORS}, got {mode!r}")
    return mode


def _validate_residual_quant_mode(mode):
    aliases = {
        "asym": "asym_zero_point",
        "asymmetric": "asym_zero_point",
        "zero_point": "asym_zero_point",
        "zp": "asym_zero_point",
    }
    mode = aliases.get(str(mode), str(mode))
    if mode != "asym_zero_point":
        raise ValueError(f"TRQ v1 only supports residual_quant_mode=asym_zero_point, got {mode}")
    return mode


def _resolve_scale_precision(scale_precision):
    if isinstance(scale_precision, torch.dtype):
        return scale_precision
    if scale_precision in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if scale_precision in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if scale_precision in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"unsupported scale precision: {scale_precision}")
