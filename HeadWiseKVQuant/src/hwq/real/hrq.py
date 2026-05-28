from __future__ import annotations

import math
import re
import torch
import torch.nn.functional as F

SUPPORTED_BITS = (2, 4, 8)
_SUPPORTED_PREDICTOR_MODES = ("identity", "affine_channel", "tiny_mlp")

def extract_hrq_bits(quant_type: str) -> int:
    m = re.search(r"int(\d+)", quant_type)
    if m is None:
        raise ValueError(f"Cannot identify num_bits from {quant_type}")
    return _validate_bits(int(m.group(1)))

def hrq_quantize_tensor(
    tensor: torch.Tensor,
    *,
    num_bits: int,
    block_size: int,
    anchor_bits: int = 4,
    predictor_stride: int = 1560,
    predictor_mode: str = "identity",
    predictor_params: dict | None = None,
    scale_precision: torch.dtype | str = torch.bfloat16,
    residual_quant_mode: str = "asym_zero_point",
) -> dict:
    if tensor.ndim != 4:
        raise ValueError(f"hrq expects [B, H, S, D], got {tuple(tensor.shape)}")
    num_bits = _validate_bits(num_bits)
    anchor_bits = _validate_bits(anchor_bits)
    block_size = _validate_block_size(block_size)
    predictor_stride = _validate_stride(predictor_stride, tensor.shape[2])
    predictor_mode = _validate_predictor_mode(predictor_mode)
    residual_quant_mode = _validate_residual_quant_mode(residual_quant_mode)
    scale_precision = _resolve_scale_precision(scale_precision)
    bsz, heads, seq_len, head_dim = tensor.shape
    padded_dim = math.ceil(head_dim / block_size) * block_size

    anchor_len = min(predictor_stride, seq_len)
    anchor_q, anchor_scales = _quantize_sym(tensor[:, :, :anchor_len, :], anchor_bits, block_size, padded_dim, scale_precision)
    prev = _dequantize_sym(anchor_q, anchor_scales, anchor_bits, block_size, head_dim, padded_dim, tensor.dtype)

    residual_qs, residual_scales, residual_zps = [], [], []
    unit_lengths = [anchor_len]
    offset = anchor_len
    while offset < seq_len:
        unit_len = min(predictor_stride, seq_len - offset)
        current = tensor[:, :, offset:offset + unit_len, :]
        pred = _predict(prev[:, :, :unit_len, :], predictor_mode, predictor_params)
        residual = current.float() - pred
        q, scales, zps = _quantize_asym(residual, num_bits, block_size, padded_dim, scale_precision)
        residual_recon = _dequantize_asym(q, scales, zps, num_bits, block_size, head_dim, padded_dim, torch.float32)
        prev = (pred + residual_recon).to(tensor.dtype)
        residual_qs.append(q); residual_scales.append(scales); residual_zps.append(zps)
        unit_lengths.append(unit_len)
        offset += unit_len

    return {
        "format": "hrq",
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
        "shape": (int(bsz), int(heads), int(seq_len), int(head_dim)),
        "padded_dim": int(padded_dim),
        "predictor_mode": predictor_mode,
        "scale_precision": str(scale_precision).replace("torch.", ""),
        "residual_quant_mode": residual_quant_mode,
    }

def hrq_dequantize_tensor(
    packed_state: dict,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    predictor_params: dict | None = None,
) -> torch.Tensor:
    if packed_state.get("format") != "hrq":
        raise ValueError(f"Unsupported HRQ format: {packed_state.get(format)}")
    bsz, heads, seq_len, head_dim = [int(x) for x in packed_state["shape"]]
    padded_dim = int(packed_state.get("padded_dim", head_dim))
    block_size = int(packed_state["block_size"])
    anchor_bits = int(packed_state["anchor_bits"])
    num_bits = int(packed_state["num_bits"])
    predictor_mode = _validate_predictor_mode(packed_state.get("predictor_mode", "identity"))
    unit_lengths = [int(x) for x in packed_state["unit_lengths"]]
    anchor = _dequantize_sym(packed_state["anchor_quant"], packed_state["anchor_scales"], anchor_bits, block_size, head_dim, padded_dim, output_dtype)
    outputs = [anchor]
    prev = anchor
    q_all = packed_state.get("residual_quant")
    s_all = packed_state.get("residual_scales")
    z_all = packed_state.get("residual_zero_points")
    residual_offset = 0
    for unit_len in unit_lengths[1:]:
        q = q_all[:, :, residual_offset:residual_offset + unit_len, :]
        s = s_all[:, :, residual_offset:residual_offset + unit_len, :]
        z = z_all[:, :, residual_offset:residual_offset + unit_len, :]
        residual_offset += unit_len
        pred = _predict(prev[:, :, :unit_len, :], predictor_mode, predictor_params)
        residual = _dequantize_asym(q, s, z, num_bits, block_size, head_dim, padded_dim, torch.float32)
        current = (pred + residual).to(output_dtype)
        outputs.append(current)
        prev = current
    return torch.cat(outputs, dim=2).reshape(bsz, heads, seq_len, head_dim)

def _quantize_sym(x, bits, block_size, padded_dim, scale_precision):
    x = _pad(x.float(), padded_dim)
    blocks = padded_dim // block_size
    grouped = x.reshape(*x.shape[:-1], blocks, block_size)
    qmax = (1 << (bits - 1)) - 1
    scales = (grouped.abs().amax(dim=-1).clamp_min(1e-8) / qmax).to(scale_precision)
    q = torch.round(grouped / scales.float().unsqueeze(-1)).clamp(-qmax, qmax).to(torch.int16).reshape(*x.shape)
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
    scales = ((xmax - xmin).clamp_min(1e-8) / qmax).to(scale_precision)
    zps = torch.round(-xmin / scales.float()).clamp(0, qmax).to(torch.uint8)
    q = torch.round(grouped / scales.float().unsqueeze(-1) + zps.float().unsqueeze(-1)).clamp(0, qmax).to(torch.uint8).reshape(*x.shape)
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
    packed_dim = q.shape[-1] // per_byte
    grouped = q.to(torch.uint8).reshape(*q.shape[:-1], packed_dim, per_byte)
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
    parts = [((q >> (bits * (per_byte - 1 - idx))) & mask).to(torch.uint8) for idx in range(per_byte)]
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
    return params.float().unsqueeze(-1).expand(*params.shape, block_size).reshape(*params.shape[:-1], padded_dim)

def _predict(prev: torch.Tensor, predictor_mode: str, predictor_params: dict | None) -> torch.Tensor:
    """Return float32 prediction of the next chunk given the previous chunk.

    Args:
        prev: [B, H, S, D] in the model's native dtype.
        predictor_mode: one of "identity", "affine_channel", "tiny_mlp".
        predictor_params: dict of pre-fitted params (ignored for identity).
    """
    if predictor_mode == "identity":
        return prev.float()

    if predictor_mode == "affine_channel":
        if predictor_params is None:
            return prev.float()
        alpha = predictor_params["alpha"].to(prev.device, torch.float32)  # [H, D]
        beta = predictor_params["beta"].to(prev.device, torch.float32)   # [H, D]
        # Broadcast [H, D] → [1, H, 1, D]
        return alpha.unsqueeze(0).unsqueeze(2) * prev.float() + beta.unsqueeze(0).unsqueeze(2)

    if predictor_mode == "tiny_mlp":
        if predictor_params is None:
            return prev.float()
        B, H, S, D = prev.shape
        x = prev.float().reshape(-1, D)
        w1 = predictor_params["fc1_weight"].to(prev.device, torch.float32)
        b1 = predictor_params["fc1_bias"].to(prev.device, torch.float32)
        w2 = predictor_params["fc2_weight"].to(prev.device, torch.float32)
        b2 = predictor_params["fc2_bias"].to(prev.device, torch.float32)
        h = F.gelu(F.linear(x, w1, b1))
        out = F.linear(h, w2, b2)
        return out.reshape(B, H, S, D)

    raise ValueError(f"Unknown predictor_mode: {predictor_mode}")

def _validate_bits(bits):
    bits = int(bits)
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"HRQ only supports {SUPPORTED_BITS}, got int{bits}")
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

def _validate_predictor_mode(mode):
    mode = str(mode)
    if mode not in _SUPPORTED_PREDICTOR_MODES:
        raise ValueError(
            f"hrq predictor_mode must be one of {_SUPPORTED_PREDICTOR_MODES}, got {mode!r}"
        )
    return mode

def _validate_residual_quant_mode(mode):
    aliases = {"asym": "asym_zero_point", "asymmetric": "asym_zero_point", "zero_point": "asym_zero_point", "zp": "asym_zero_point"}
    mode = aliases.get(str(mode), str(mode))
    if mode != "asym_zero_point":
        raise ValueError(f"hrq v1 only supports residual_quant_mode=asym_zero_point, got {mode}")
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
