"""Runtime storage and operation summaries for online TRQ experiments."""

from __future__ import annotations

from typing import Any

import torch

from .kv_cache import ChunkedKVCache, ChunkState


def nested_tensor_nbytes(value: Any, *, _seen: set[int] | None = None) -> int:
    """Count logical tensor payload bytes without double-counting tensor objects."""
    seen = set() if _seen is None else _seen
    if isinstance(value, torch.Tensor):
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(nested_tensor_nbytes(item, _seen=seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(nested_tensor_nbytes(item, _seen=seen) for item in value)
    return 0


def summarize_chunked_cache(cache: ChunkedKVCache) -> dict[str, int | float]:
    raw_bytes = sum(
        nested_tensor_nbytes(chunk)
        for chunk in cache.chunks
        if chunk is not None
    )
    quantized_bytes = sum(
        nested_tensor_nbytes(span.get("quant_data"))
        for span in cache.quantized_spans
    )
    filled_chunks = sum(state != ChunkState.EMPTY for state in cache.chunk_state)
    values_per_chunk = (
        cache.batch_size
        * cache.frame_seq_length
        * cache.num_heads
        * cache.head_dim
    )
    represented_values = int(filled_chunks * values_per_chunk)
    source_element_size = torch.empty((), dtype=cache.dtype).element_size()
    bf16_equivalent_bytes = int(represented_values * source_element_size)
    physical_bytes = int(raw_bytes + quantized_bytes)
    effective_bits = (
        8.0 * physical_bytes / represented_values
        if represented_values
        else 0.0
    )
    return {
        "raw_bytes": int(raw_bytes),
        "quantized_bytes": int(quantized_bytes),
        "physical_bytes": physical_bytes,
        "bf16_equivalent_bytes": bf16_equivalent_bytes,
        "represented_values": represented_values,
        "filled_chunks": int(filled_chunks),
        "quantized_spans": int(len(cache.quantized_spans)),
        "effective_bits_per_value": float(effective_bits),
    }


def summarize_kv_cache(layers: list[dict]) -> dict[str, Any]:
    by_kind: dict[str, dict[str, int | float]] = {}
    for kind in ("k", "v"):
        summaries = [
            summarize_chunked_cache(layer[kind])
            for layer in layers
            if isinstance(layer.get(kind), ChunkedKVCache)
        ]
        physical_bytes = sum(int(item["physical_bytes"]) for item in summaries)
        bf16_bytes = sum(int(item["bf16_equivalent_bytes"]) for item in summaries)
        represented_values = sum(int(item["represented_values"]) for item in summaries)
        by_kind[kind] = {
            "physical_bytes": physical_bytes,
            "bf16_equivalent_bytes": bf16_bytes,
            "represented_values": represented_values,
            "effective_bits_per_value": (
                8.0 * physical_bytes / represented_values
                if represented_values
                else 0.0
            ),
            "quantized_spans": sum(int(item["quantized_spans"]) for item in summaries),
        }

    physical_bytes = sum(int(item["physical_bytes"]) for item in by_kind.values())
    bf16_bytes = sum(int(item["bf16_equivalent_bytes"]) for item in by_kind.values())
    represented_values = sum(int(item["represented_values"]) for item in by_kind.values())
    return {
        "k": by_kind["k"],
        "v": by_kind["v"],
        "total": {
            "physical_bytes": physical_bytes,
            "bf16_equivalent_bytes": bf16_bytes,
            "represented_values": represented_values,
            "effective_bits_per_value": (
                8.0 * physical_bytes / represented_values
                if represented_values
                else 0.0
            ),
            "compression_ratio": (
                bf16_bytes / physical_bytes
                if physical_bytes
                else 0.0
            ),
            "saving_fraction": (
                1.0 - physical_bytes / bf16_bytes
                if bf16_bytes
                else 0.0
            ),
        },
    }


def summarize_cache_profile(layers: list[dict], *, synchronize: bool = False) -> dict[str, float | int]:
    calls = 0
    elapsed_ms = 0.0
    synchronized = False
    for layer in layers:
        for kind in ("k", "v"):
            cache = layer.get(kind)
            if not isinstance(cache, ChunkedKVCache):
                continue
            summary = cache.profile_summary(
                synchronize=synchronize and not synchronized
            )
            synchronized = synchronized or bool(synchronize)
            calls += int(summary["dequantize_calls"])
            elapsed_ms += float(summary["dequantize_ms"])
    return {
        "dequantize_calls": calls,
        "dequantize_ms": elapsed_ms,
    }
