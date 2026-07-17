"""
Chunk-based KV cache: manages a *single* K or V tensor in frame-aligned
chunks, supporting mixed BF16 / quantized storage.
"""

from enum import IntEnum
from typing import Optional
import time

import torch

from .uncompress import uncompress_single_cache

class ChunkState(IntEnum):
    EMPTY = 0
    BF16 = 1
    QUANTIZED = 2



class ChunkedKVCache:
    """
    Manages **one** tensor (either K or V) in frame-aligned chunks.

    * BF16 chunks are stored individually; memory is allocated lazily with
      ``torch.empty`` on first write.
    * Real-quantized spans (dicts from ``compress_kv_cache``) can cover
      multiple contiguous chunks and are stored as a single compressed object.
    * ``read()`` returns a full-precision tensor in the configured *layout*,
      decompressing quantized spans on the fly.

    Args:
        layout: ``"BHSD"`` stores chunks as ``[B, H, S, D]`` (default);
                ``"BSHD"`` stores chunks as ``[B, S, H, D]``.
                All ``write`` / ``read`` / ``store_quantized`` calls use
                the same layout — no implicit permutes.
    """

    def __init__(
        self,
        batch_size: int,
        frame_seq_length: int,
        num_heads: int,
        head_dim: int,
        max_num_chunks: int,
        dtype: torch.dtype,
        device: torch.device,
        layout: str = "BHSD",
    ):
        assert layout in ("BHSD", "BSHD"), f"layout must be BHSD or BSHD, got {layout}"
        self.batch_size = batch_size
        self.frame_seq_length = frame_seq_length
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_num_chunks = max_num_chunks
        self.dtype = dtype
        self.device = device
        self.layout = layout
        self.seq_dim = 2 if layout == "BHSD" else 1

        # Per-chunk BF16 storage (None → not yet allocated, zero GPU memory)
        self.chunks: list[Optional[torch.Tensor]] = [None] * max_num_chunks
        self.chunk_state: list[ChunkState] = [ChunkState.EMPTY] * max_num_chunks

        # Quantized spans: {start_chunk, end_chunk, quant_data}
        # quant_data is the packed dict from compress_kv_cache (with "info").
        self.quantized_spans: list[dict] = []
        self.profile_enabled = False
        self._dequantize_calls = 0
        self._dequantize_cpu_ms = 0.0
        self._dequantize_cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def kv_cache_size(self) -> int:
        """Total token capacity."""
        return self.max_num_chunks * self.frame_seq_length

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _token_to_chunk(self, token_index: int) -> int:
        assert token_index % self.frame_seq_length == 0, (
            f"Token index {token_index} not aligned to "
            f"frame_seq_length {self.frame_seq_length}"
        )
        return token_index // self.frame_seq_length

    def _alloc_chunk(self, ci: int):
        if self.chunks[ci] is None:
            if self.layout == "BHSD":
                shape = [self.batch_size, self.num_heads, self.frame_seq_length, self.head_dim]
            else:
                shape = [self.batch_size, self.frame_seq_length, self.num_heads, self.head_dim]
            self.chunks[ci] = torch.empty(shape, dtype=self.dtype, device=self.device)

    def _find_span(self, ci: int) -> dict:
        for span in self.quantized_spans:
            if span["start_chunk"] <= ci < span["end_chunk"]:
                return span
        raise ValueError(f"No quantized span covers chunk {ci}")

    def _remove_overlapping_spans(self, start_chunk: int, end_chunk: int):
        self.quantized_spans = [
            s for s in self.quantized_spans
            if s["end_chunk"] <= start_chunk or s["start_chunk"] >= end_chunk
        ]

    def _decode_span(self, span: dict) -> torch.Tensor:
        start_event = None
        end_event = None
        cpu_start = None
        if self.profile_enabled:
            self._dequantize_calls += 1
            if self.device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                cpu_start = time.perf_counter()
        decoded = uncompress_single_cache(span["quant_data"])
        if start_event is not None and end_event is not None:
            end_event.record()
            self._dequantize_cuda_events.append((start_event, end_event))
        elif cpu_start is not None:
            self._dequantize_cpu_ms += (time.perf_counter() - cpu_start) * 1000.0
        if self.layout == "BSHD":
            decoded = decoded.permute(0, 2, 1, 3).contiguous()
        return decoded

    # ------------------------------------------------------------------
    # Write (BF16)
    # ------------------------------------------------------------------

    def write(self, start_index: int, end_index: int, data: torch.Tensor):
        """
        Write BF16 data into the cache.

        Args:
            start_index: Token position (frame-aligned).
            end_index: End token position (frame-aligned).
            data: Tensor in the cache's configured layout.
                  BHSD → ``[B, H, num_tokens, D]``;
                  BSHD → ``[B, num_tokens, H, D]``.
        """
        num_tokens = data.shape[self.seq_dim]
        assert end_index - start_index == num_tokens, (
            f"end_index - start_index ({end_index - start_index}) != "
            f"data tokens ({num_tokens})"
        )
        sc = self._token_to_chunk(start_index)
        ec = self._token_to_chunk(end_index)
        assert ec <= self.max_num_chunks, (
            f"Write exceeds capacity: chunk {ec} > max {self.max_num_chunks}"
        )

        for i, ci in enumerate(range(sc, ec)):
            s = i * self.frame_seq_length
            e = s + self.frame_seq_length

            if self.chunk_state[ci] == ChunkState.QUANTIZED:
                self._remove_overlapping_spans(ci, ci + 1)

            self._alloc_chunk(ci)
            if self.seq_dim == 2:  # BHSD
                self.chunks[ci].copy_(data[:, :, s:e, :])
            else:  # BSHD
                self.chunks[ci].copy_(data[:, s:e, :, :])
            self.chunk_state[ci] = ChunkState.BF16

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, start_index: int, end_index: int) -> torch.Tensor:
        """
        Return full-precision tensor for ``[start_index, end_index)``
        in the cache's configured layout.  Decompresses quantized spans
        on the fly.
        """
        sc = self._token_to_chunk(start_index)
        ec = self._token_to_chunk(end_index)
        parts: list[torch.Tensor] = []
        ci = sc

        while ci < ec:
            state = self.chunk_state[ci]

            if state == ChunkState.BF16:
                parts.append(self.chunks[ci])
                ci += 1

            elif state == ChunkState.QUANTIZED:
                span = self._find_span(ci)
                dec = self._decode_span(span)

                span_start = span["start_chunk"]
                span_end = span["end_chunk"]
                tok_off = (ci - span_start) * self.frame_seq_length
                tok_len = (min(span_end, ec) - ci) * self.frame_seq_length
                if self.seq_dim == 2:  # BHSD
                    parts.append(dec[:, :, tok_off:tok_off + tok_len, :])
                else:  # BSHD
                    parts.append(dec[:, tok_off:tok_off + tok_len, :, :])
                ci = min(span_end, ec)

            else:
                raise ValueError(f"Chunk {ci} is {ChunkState.EMPTY.name}, cannot read")

        return torch.cat(parts, dim=self.seq_dim)

    # ------------------------------------------------------------------
    # Store quantized data (output of compress_kv_cache, [B, H, S, D])
    # ------------------------------------------------------------------

    def store_quantized(self, start_index: int, end_index: int, quant_data):
        """
        Store quantized result for one tensor (K or V).

        * **Tensor** (fake quant): written in the cache's native layout.
        * **Dict** (real quant): stored as a single quantized span.
          Must already contain ``"info"`` with ``output_dtype`` /
          ``quant_config`` (via ``_pack_info_into_kv_cache``).
        """
        sc = self._token_to_chunk(start_index)
        ec = self._token_to_chunk(end_index)

        if isinstance(quant_data, torch.Tensor):
            self.write(start_index, end_index, quant_data)
        else:
            self._remove_overlapping_spans(sc, ec)
            for ci in range(sc, ec):
                self.chunks[ci] = None
                self.chunk_state[ci] = ChunkState.QUANTIZED
            self.quantized_spans.append({
                "start_chunk": sc,
                "end_chunk": ec,
                "quant_data": quant_data,
            })

    def evict_prefix(self, num_tokens: int, *, sink_tokens: int = 0) -> None:
        """Remove a frame-aligned range after the sink and compact live chunks.

        Quantized spans wholly outside the removed range retain their packed
        state. A span cut by the eviction boundary is decoded once and its
        surviving chunks become BF16, because its original residual chain no
        longer has a valid anchor after slicing.
        """
        if num_tokens == 0:
            return
        if num_tokens < 0 or sink_tokens < 0:
            raise ValueError("eviction and sink sizes must be non-negative")
        start_chunk = self._token_to_chunk(sink_tokens)
        removed_chunks = self._token_to_chunk(num_tokens)
        end_chunk = start_chunk + removed_chunks
        if end_chunk > self.max_num_chunks:
            raise ValueError(
                f"eviction [{start_chunk}, {end_chunk}) exceeds {self.max_num_chunks} chunks"
            )

        overlapping = [
            span for span in self.quantized_spans
            if span["start_chunk"] < end_chunk and span["end_chunk"] > start_chunk
        ]
        overlapping_ids = {id(span) for span in overlapping}
        decoded_spans = {id(span): self._decode_span(span) for span in overlapping}
        new_chunks: list[Optional[torch.Tensor]] = [None] * self.max_num_chunks
        new_states = [ChunkState.EMPTY] * self.max_num_chunks
        new_spans: list[dict] = []

        def compact_index(old_index: int) -> int | None:
            if old_index < start_chunk:
                return old_index
            if old_index >= end_chunk:
                return old_index - removed_chunks
            return None

        for old_index, state in enumerate(self.chunk_state):
            if state != ChunkState.BF16:
                continue
            new_index = compact_index(old_index)
            if new_index is not None:
                new_chunks[new_index] = self.chunks[old_index]
                new_states[new_index] = ChunkState.BF16

        for span in self.quantized_spans:
            overlaps = id(span) in overlapping_ids
            if not overlaps:
                new_start = compact_index(span["start_chunk"])
                new_last = compact_index(span["end_chunk"] - 1)
                if new_start is None or new_last is None:
                    raise RuntimeError("non-overlapping quantized span mapped into eviction range")
                shifted = {
                    "start_chunk": new_start,
                    "end_chunk": new_last + 1,
                    "quant_data": span["quant_data"],
                }
                new_spans.append(shifted)
                for chunk_index in range(shifted["start_chunk"], shifted["end_chunk"]):
                    new_states[chunk_index] = ChunkState.QUANTIZED
                continue

            decoded = decoded_spans[id(span)]
            for old_index in range(span["start_chunk"], span["end_chunk"]):
                new_index = compact_index(old_index)
                if new_index is None:
                    continue
                offset = (old_index - span["start_chunk"]) * self.frame_seq_length
                if self.seq_dim == 2:
                    chunk = decoded[:, :, offset:offset + self.frame_seq_length, :]
                else:
                    chunk = decoded[:, offset:offset + self.frame_seq_length, :, :]
                new_chunks[new_index] = chunk.clone().contiguous()
                new_states[new_index] = ChunkState.BF16

        self.chunks = new_chunks
        self.chunk_state = new_states
        self.quantized_spans = new_spans

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self):
        self.chunks = [None] * self.max_num_chunks
        self.chunk_state = [ChunkState.EMPTY] * self.max_num_chunks
        self.quantized_spans.clear()
        self.reset_profile()

    def enable_profiling(self, enabled: bool = True):
        self.profile_enabled = bool(enabled)
        self.reset_profile()

    def reset_profile(self):
        self._dequantize_calls = 0
        self._dequantize_cpu_ms = 0.0
        self._dequantize_cuda_events.clear()

    def profile_summary(self, *, synchronize: bool = False) -> dict[str, float | int]:
        if synchronize and self._dequantize_cuda_events:
            torch.cuda.synchronize(self.device)
        cuda_ms = sum(
            float(start.elapsed_time(end))
            for start, end in self._dequantize_cuda_events
        )
        return {
            "dequantize_calls": int(self._dequantize_calls),
            "dequantize_ms": float(self._dequantize_cpu_ms + cuda_ms),
        }

    # ------------------------------------------------------------------
    # Offload / onload
    # ------------------------------------------------------------------

    def offload(self):
        for i in range(self.max_num_chunks):
            if self.chunks[i] is not None:
                self.chunks[i] = self.chunks[i].to("cpu")
        for span in self.quantized_spans:
            span["quant_data"] = _move_item(span["quant_data"], "cpu")

    def onload(self, device: torch.device):
        for i in range(self.max_num_chunks):
            if self.chunks[i] is not None:
                self.chunks[i] = self.chunks[i].to(device)
        for span in self.quantized_spans:
            span["quant_data"] = _move_item(span["quant_data"], device)


# ======================================================================
# Layer-dict helpers  (operate on the {"k":…, "v":…, …} dict)
# ======================================================================

def offload_kv_cache_layer(layer: dict):
    """Offload an entire layer dict (with ChunkedKVCache values) to CPU."""
    for key, val in layer.items():
        if isinstance(val, ChunkedKVCache):
            val.offload()
        elif isinstance(val, torch.Tensor):
            layer[key] = val.to("cpu")


def onload_kv_cache_layer(layer: dict, device: torch.device):
    """Onload an entire layer dict to *device*."""
    for key, val in layer.items():
        if isinstance(val, ChunkedKVCache):
            val.onload(device)
        elif isinstance(val, torch.Tensor):
            layer[key] = val.to(device)


# ======================================================================
# Internal helper
# ======================================================================

def _move_item(item, device):
    """Recursively move a tensor / dict-of-tensors to *device*."""
    if isinstance(item, torch.Tensor):
        return item.to(device)
    if isinstance(item, dict):
        for key, val in item.items():
            item[key] = _move_item(val, device)
        return item
    if isinstance(item, list):
        return [_move_item(val, device) for val in item]
    if isinstance(item, tuple):
        return tuple(_move_item(val, device) for val in item)
    return item
