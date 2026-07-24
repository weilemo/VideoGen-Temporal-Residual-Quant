"""Packed KV-cache integration helpers for the official Causal Forcing backend."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from trq.compress import compress_kv_cache, get_quantize_fn
from trq.kv_cache import ChunkedKVCache


SUPPORTED_QUANT_TYPES = {
    "none",
    "trq-int4",
    "trq-int2",
    "packed-naive-int4",
    "packed-naive-int2",
}


def validate_quant_type(quant_type: str) -> str:
    quant_type = str(quant_type)
    if quant_type not in SUPPORTED_QUANT_TYPES:
        raise ValueError(
            f"unsupported Causal Forcing KV quant type {quant_type!r}; "
            f"expected one of {sorted(SUPPORTED_QUANT_TYPES)}"
        )
    return quant_type


class CausalChunkedKVCache(ChunkedKVCache):
    """Expose the tensor slice API used by Causal Forcing's BSHD cache."""

    @property
    def shape(self):
        return (
            self.batch_size,
            self.kv_cache_size,
            self.num_heads,
            self.head_dim,
        )

    @staticmethod
    def _sequence_bounds(key, size):
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("Causal Forcing cache expects cache[:, start:end]")
        batch_key, sequence_key = key
        if batch_key != slice(None) or not isinstance(sequence_key, slice):
            raise TypeError("Only full-batch sequence slices are supported")
        if sequence_key.step not in (None, 1):
            raise ValueError("Strided cache slices are not supported")
        start = 0 if sequence_key.start is None else int(sequence_key.start)
        end = size if sequence_key.stop is None else int(sequence_key.stop)
        return start, end

    def __getitem__(self, key):
        start, end = self._sequence_bounds(key, self.kv_cache_size)
        if end <= start:
            return torch.empty(
                (self.batch_size, 0, self.num_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
        return self.read(start, end)

    def __setitem__(self, key, value):
        start, end = self._sequence_bounds(key, self.kv_cache_size)
        self.write(start, end, value)


def create_cache(
    batch_size: int,
    frame_seq_length: int,
    num_heads: int,
    head_dim: int,
    max_frames: int,
    dtype: torch.dtype,
    device: torch.device,
    quant_type: str,
):
    """Create the native BF16 tensor or a frame-aligned packed cache."""
    quant_type = validate_quant_type(quant_type)
    if quant_type == "none":
        return torch.zeros(
            [batch_size, frame_seq_length * max_frames, num_heads, head_dim],
            dtype=dtype,
            device=device,
        )
    return CausalChunkedKVCache(
        batch_size,
        frame_seq_length,
        num_heads,
        head_dim,
        max_frames,
        dtype,
        device,
        layout="BSHD",
    )


def cache_size(cache) -> int:
    if isinstance(cache, ChunkedKVCache):
        return cache.kv_cache_size
    return int(cache.shape[1])


def read_cache(cache, start: int, end: int) -> torch.Tensor:
    if isinstance(cache, ChunkedKVCache):
        return cache.read(start, end)
    return cache[:, start:end]


def write_cache(cache, start: int, end: int, value: torch.Tensor) -> None:
    if isinstance(cache, ChunkedKVCache):
        cache.write(start, end, value)
    else:
        cache[:, start:end] = value


def evict_cache_pair(
    layer: dict,
    *,
    num_tokens: int,
    sink_tokens: int,
    local_end_index: int,
) -> None:
    """Evict a middle prefix while preserving Causal Forcing's sink tokens."""
    if isinstance(layer["k"], ChunkedKVCache):
        layer["k"].evict_prefix(num_tokens, sink_tokens=sink_tokens)
        layer["v"].evict_prefix(num_tokens, sink_tokens=sink_tokens)
        return

    num_rolled_tokens = local_end_index - num_tokens - sink_tokens
    layer["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = layer["k"][
        :, sink_tokens + num_tokens:sink_tokens + num_tokens + num_rolled_tokens
    ].clone()
    layer["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = layer["v"][
        :, sink_tokens + num_tokens:sink_tokens + num_tokens + num_rolled_tokens
    ].clone()


def reset_cache_pair(layer: dict) -> None:
    """Drop packed spans between prompts while retaining allocated metadata."""
    for name in ("k", "v"):
        cache = layer[name]
        if isinstance(cache, ChunkedKVCache):
            cache.clear()


def _quant_config(meta: dict) -> SimpleNamespace:
    quant_type = validate_quant_type(meta.get("kv_quant_type", "none"))
    return SimpleNamespace(
        quant_type=quant_type,
        quant_block_size=int(meta.get("kv_quant_block_size", 64)),
        trq_anchor_bits=int(meta.get("trq_anchor_bits", 4)),
        trq_predictor_stride=int(meta.get("trq_predictor_stride", 1560)),
        trq_predictor_mode=str(meta.get("trq_predictor_mode", "identity")),
        trq_k_bits=int(meta.get("trq_k_bits", 0)),
        trq_v_bits=int(meta.get("trq_v_bits", 0)),
    )


def _attach_decode_info(state, config: SimpleNamespace, dtype: torch.dtype):
    if not isinstance(state, dict):
        return state
    state = dict(state)
    state["info"] = {
        "quant_config": vars(config).copy(),
        "output_dtype": dtype,
    }
    return state


def quantize_cache_pair(
    layer: dict,
    start: int,
    end: int,
    *,
    meta: dict,
    layer_idx: int,
) -> None:
    """Pack one frame-aligned K/V span after the official cache update."""
    if not isinstance(layer["k"], ChunkedKVCache):
        return
    config = _quant_config(meta)
    if config.quant_type == "none":
        return

    k = layer["k"].read(start, end).permute(0, 2, 1, 3).contiguous()
    v = layer["v"].read(start, end).permute(0, 2, 1, 3).contiguous()
    quantize_fn = get_quantize_fn(config.quant_type, config)
    k_state, v_state = compress_kv_cache(
        k,
        v,
        config.quant_type,
        config,
        quantize_fn,
        layer_idx=layer_idx,
    )
    layer["k"].store_quantized(
        start, end, _attach_decode_info(k_state, config, k.dtype)
    )
    layer["v"].store_quantized(
        start, end, _attach_decode_info(v_state, config, v.dtype)
    )


def packed_cache_bytes(layers: list[dict]) -> tuple[int, int]:
    """Return native-equivalent and currently allocated packed byte counts."""
    native = 0
    packed = 0
    for layer in layers:
        for name in ("k", "v"):
            cache = layer[name]
            if isinstance(cache, ChunkedKVCache):
                native += sum(
                    chunk.numel() * chunk.element_size()
                    for chunk in cache.chunks
                    if chunk is not None
                )
                for span in cache.quantized_spans:
                    native += (
                        (span["end_chunk"] - span["start_chunk"])
                        * cache.frame_seq_length
                        * cache.batch_size
                        * cache.num_heads
                        * cache.head_dim
                        * torch.tensor([], dtype=cache.dtype).element_size()
                    )
                    state = span["quant_data"]
                    packed += _state_nbytes(state)
                packed += sum(
                    chunk.numel() * chunk.element_size()
                    for chunk in cache.chunks
                    if chunk is not None
                )
    return native, packed


def _state_nbytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_state_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_state_nbytes(item) for item in value)
    return 0
