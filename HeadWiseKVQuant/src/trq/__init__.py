"""Public API for HeadWiseKVQuant.

Heavy GPU modules are imported lazily so the CPU reference codecs and config
objects remain usable in environments without Triton.
"""

from importlib import import_module


_EXPORTS = {
    "compress_kv_cache": (".compress", "compress_kv_cache"),
    "get_quantize_fn": (".compress", "get_quantize_fn"),
    "get_quantize_type": (".compress", "get_quantize_type"),
    "HeadGroup": (".headwise", "HeadGroup"),
    "RandomHeadPolicy": (".headwise", "RandomHeadPolicy"),
    "TopKHeadPolicy": (".headwise", "TopKHeadPolicy"),
    "compress_headwise_kv_cache": (".headwise", "compress_headwise_kv_cache"),
    "load_topk_head_policy": (".headwise", "load_topk_head_policy"),
    "uncompress_kv_cache": (".uncompress", "uncompress_kv_cache"),
    "uncompress_single_cache": (".uncompress", "uncompress_single_cache"),
    "build_topk_policy_from_focused_forcing": (".head_importance", "build_topk_policy_from_focused_forcing"),
    "load_focused_forcing_head_losses": (".head_importance", "load_focused_forcing_head_losses"),
    "mean_head_scores": (".head_importance", "mean_head_scores"),
    "select_top_heads_by_layer": (".head_importance", "select_top_heads_by_layer"),
    "write_topk_policy": (".head_importance", "write_topk_policy"),
    "ChunkedKVCache": (".kv_cache", "ChunkedKVCache"),
    "offload_kv_cache_layer": (".kv_cache", "offload_kv_cache_layer"),
    "onload_kv_cache_layer": (".kv_cache", "onload_kv_cache_layer"),
    "QuantizeConfig": (".sim.quant.quantize_config", "QuantizeConfig"),
    "TimeLoggingContext": (".timer", "TimeLoggingContext"),
    "time_logging_decorator": (".timer", "time_logging_decorator"),
    "logger": (".logger", "logger"),
    "trq_quantize_tensor": (".real.trq", "trq_quantize_tensor"),
    "trq_dequantize_tensor": (".real.trq", "trq_dequantize_tensor"),
    "trq_state_nbytes": (".real.trq", "trq_state_nbytes"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
