"""
Uncompress and dequantize for triton-xxx quant types (triton-nstages-kmeans-int2/int4).
Mirrors the structure of compress.py: get_dequantize_fn + uncompress_kv_cache.
"""

import re
from types import SimpleNamespace
import torch

from .compress import get_quantize_type, QuantizeFunctions
from .sim.quant.quantize_config import QuantizeConfig
from .functions import triton_prq_dequantize_tensor


########################################################
# Entrypoints (mirror compress.py)
########################################################

def extract_num_bits(quant_config: QuantizeConfig):
    m = re.search(r'int(\d+)', quant_config.quant_type)
    if m is None:
        raise ValueError(f"Cannot identify num_bits from {quant_config.quant_type}")
    return int(m.group(1))


def _coerce_quant_config(quant_config):
    if isinstance(quant_config, dict):
        return SimpleNamespace(**quant_config)
    return quant_config


def _dequantize_single_cache(
    packed_state: torch.Tensor | dict,
    quant_config,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(packed_state, dict):
        return packed_state

    quant_config = _coerce_quant_config(quant_config)
    quantize_type = get_quantize_type(quant_config.quant_type)
    num_bits = extract_num_bits(quant_config)

    if quantize_type in (QuantizeFunctions.TRITON_PRQ, QuantizeFunctions.TRITON_PRQ_CLIP):
        return triton_prq_dequantize_tensor(
            packed_state,
            quant_config.quant_block_size,
            num_bits,
            output_dtype=output_dtype,
        )

    raise ValueError(f"Unsupported packed quant type during dequantization: {quant_config.quant_type}")


def uncompress_single_cache(cache: torch.Tensor | dict) -> torch.Tensor:
    if not isinstance(cache, dict):
        return cache

    info = cache["info"]
    output_dtype = info["output_dtype"]

    if "groups" in cache:
        groups = cache["groups"]
        if len(groups) == 0:
            raise ValueError("Mixed head-wise cache contains no groups")

        parts = []
        for group in groups:
            dec = _dequantize_single_cache(
                group["payload"],
                group["quant_config"],
                output_dtype=output_dtype,
            )
            parts.append((group["head_ids"], dec))

        first_dec = parts[0][1]
        bsz, _, seq_len, head_dim = first_dec.shape
        num_heads = info["num_heads"]
        merged = torch.empty(
            [bsz, num_heads, seq_len, head_dim],
            dtype=output_dtype,
            device=first_dec.device,
        )

        for head_ids, dec in parts:
            merged[:, head_ids, :, :] = dec

        return merged

    quant_config = info["quant_config"]
    return _dequantize_single_cache(cache, quant_config, output_dtype=output_dtype)

def uncompress_kv_cache(
    k_cache: torch.Tensor | dict,
    v_cache: torch.Tensor | dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Uncompress (and dequantize) one layer of KV cache, mirroring compress_kv_cache.

    Same I/O pattern as compress_kv_cache:
      - compress_kv_cache(k, v, ...) -> (k_quant, v_quant)   # one layer
      - uncompress_kv_cache(k_cached, v_cached, ...) -> (k, v)  # one layer

    Args:
        k: Cached K for this layer (tensor for non-triton, packed_state dict for triton-xxx).
        v: Cached V for this layer (tensor for non-triton, packed_state dict for triton-xxx).
        quant_type: Same quant_type used when compressing.
        quant_config: Same config used when compressing.
        dequantize_fn: From get_dequantize_fn(quant_type, quant_config).
        device: If set, move packed state to this device before dequantizing.
        output_dtype: Dtype for reconstructed K/V tensors.

    Returns:
        (k_tensor, v_tensor) for this layer, ready for attention.
    """

    if not isinstance(k_cache, dict) or not isinstance(v_cache, dict):
        return k_cache, v_cache

    k_tensor = uncompress_single_cache(k_cache)
    v_tensor = uncompress_single_cache(v_cache)
    return k_tensor, v_tensor
