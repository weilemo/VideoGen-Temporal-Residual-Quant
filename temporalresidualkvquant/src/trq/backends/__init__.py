"""Adapters for model-specific KV-cache layouts and lifecycles."""

from .causal_forcing import (
    CausalChunkedKVCache,
    create_cache as create_causal_forcing_cache,
    quantize_cache_pair as quantize_causal_forcing_cache_pair,
)
from .longcat_video import (
    LongCatKVCacheCodec,
    QuantizedLongCatCache,
    install_longcat_kv_quant,
    longcat_quant_config,
)

__all__ = [
    "CausalChunkedKVCache",
    "LongCatKVCacheCodec",
    "QuantizedLongCatCache",
    "create_causal_forcing_cache",
    "install_longcat_kv_quant",
    "longcat_quant_config",
    "quantize_causal_forcing_cache_pair",
]
