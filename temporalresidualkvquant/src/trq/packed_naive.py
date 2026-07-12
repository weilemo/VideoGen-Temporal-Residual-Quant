from __future__ import annotations

import math

import torch


SUPPORTED_BITS = (2, 4, 8)


def _validate_num_bits(num_bits: int) -> None:
    if num_bits not in SUPPORTED_BITS:
        raise ValueError(f"packed-naive only supports {SUPPORTED_BITS}, got int{num_bits}")


def _pack_lowbit(qvalues: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Pack uint8 quantized values along the last dimension."""
    _validate_num_bits(num_bits)

    if num_bits == 8:
        return qvalues.contiguous()

    values_per_byte = 8 // num_bits
    mask = (1 << num_bits) - 1
    flat_shape = qvalues.shape[:-1]
    packed_dim = qvalues.shape[-1] // values_per_byte
    q = qvalues.reshape(*flat_shape, packed_dim, values_per_byte).to(torch.uint8)

    packed = torch.zeros((*flat_shape, packed_dim), dtype=torch.uint8, device=qvalues.device)
    for idx in range(values_per_byte):
        shift = 8 - num_bits * (idx + 1)
        packed |= (q[..., idx] & mask) << shift

    return packed.contiguous()


def _unpack_lowbit(packed: torch.Tensor, num_bits: int, padded_dim: int) -> torch.Tensor:
    """Unpack uint8 values along the last dimension."""
    _validate_num_bits(num_bits)

    if num_bits == 8:
        return packed[..., :padded_dim].to(torch.uint8)

    values_per_byte = 8 // num_bits
    mask = (1 << num_bits) - 1
    parts = []
    for idx in range(values_per_byte):
        shift = 8 - num_bits * (idx + 1)
        parts.append((packed >> shift) & mask)

    stacked = torch.stack(parts, dim=-1)
    return stacked.flatten(-2)[..., :padded_dim].to(torch.uint8)


def packed_naive_quantize_tensor(
    tensor: torch.Tensor,
    *,
    num_bits: int,
    block_size: int,
) -> dict:
    """Blockwise min-max quantization with packed low-bit storage.

    Input and reconstructed output use the regular HWQ layout [B, H, S, D].
    Codes are packed into uint8 along D, while one scale and one min value are
    stored per block.  This is a real storage-compression path, unlike
    ``naive-int*`` fake quantization which returns a BF16 tensor.
    """
    _validate_num_bits(num_bits)
    if tensor.ndim != 4:
        raise ValueError(f"packed-naive expects [B, H, S, D], got {tuple(tensor.shape)}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    original_shape = tuple(tensor.shape)
    bsz, num_heads, seq_len, head_dim = original_shape
    num_blocks = math.ceil(head_dim / block_size)
    padded_dim = num_blocks * block_size

    x = tensor.detach()
    if padded_dim != head_dim:
        pad = padded_dim - head_dim
        x = torch.nn.functional.pad(x, (0, pad), mode="constant", value=0)

    x_blocks = x.float().reshape(bsz, num_heads, seq_len, num_blocks, block_size)
    valid_mask = torch.arange(padded_dim, device=tensor.device) < head_dim
    valid_mask = valid_mask.reshape(num_blocks, block_size)
    mins = torch.where(valid_mask, x_blocks, torch.inf).amin(dim=-1)
    maxs = torch.where(valid_mask, x_blocks, -torch.inf).amax(dim=-1)
    levels = (1 << num_bits) - 1
    scales = (maxs - mins) / levels
    scales = torch.clamp(scales, min=1e-8)

    q = torch.round((x_blocks - mins.unsqueeze(-1)) / scales.unsqueeze(-1))
    q = torch.clamp(q, 0, levels).to(torch.uint8)
    q = q.reshape(bsz, num_heads, seq_len, padded_dim)
    packed = _pack_lowbit(q, num_bits)

    return {
        "format": "packed-naive",
        "num_bits": num_bits,
        "block_size": block_size,
        "original_shape": original_shape,
        "padded_dim": padded_dim,
        "packed_codes": packed,
        "scales": scales.to(torch.float16),
        "mins": mins.to(torch.float16),
    }


def packed_naive_dequantize_tensor(
    packed_state: dict,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if packed_state.get("format") != "packed-naive":
        raise ValueError(f"Unsupported packed-naive format: {packed_state.get('format')}")

    num_bits = int(packed_state["num_bits"])
    block_size = int(packed_state["block_size"])
    original_shape = tuple(packed_state["original_shape"])
    padded_dim = int(packed_state["padded_dim"])
    bsz, num_heads, seq_len, head_dim = original_shape

    q = _unpack_lowbit(packed_state["packed_codes"], num_bits, padded_dim)
    num_blocks = padded_dim // block_size
    q = q.reshape(bsz, num_heads, seq_len, num_blocks, block_size).float()

    scales = packed_state["scales"].float().unsqueeze(-1)
    mins = packed_state["mins"].float().unsqueeze(-1)
    out = q * scales + mins
    out = out.reshape(bsz, num_heads, seq_len, padded_dim)[..., :head_dim]
    return out.to(dtype=output_dtype)
