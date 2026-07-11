from __future__ import annotations

import torch

from .headwise import RandomHeadPolicy, TopKHeadPolicy, compress_headwise_kv_cache


def compress_self_forcing_cache_span(
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    base_quant_config,
    policy: RandomHeadPolicy | TopKHeadPolicy,
    layer_idx: int | None = None,
) -> tuple[dict, dict]:
    """Compress a Self-Forcing KV span stored as [B, S, H, D].

    Self-Forcing keeps cache chunks in BSHD layout, while the quantizers operate
    on BHSD.  This helper makes the layout conversion explicit for adapters.
    """
    if k_bshd.ndim != 4 or v_bshd.ndim != 4:
        raise ValueError("Self-Forcing cache tensors must be [B, S, H, D]")

    k_bhsd = k_bshd.permute(0, 2, 1, 3).contiguous()
    v_bhsd = v_bshd.permute(0, 2, 1, 3).contiguous()
    return compress_headwise_kv_cache(k_bhsd, v_bhsd, base_quant_config, policy, layer_idx=layer_idx)
