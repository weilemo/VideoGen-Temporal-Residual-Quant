"""LongCat-Video 13.6B KV-cache adapter.

LongCat stores each DiT layer cache as a ``(key, value)`` pair in BHSD layout.
The official attention path concatenates that prefix cache with current K/V on
the sequence dimension. This adapter packs the persistent prefix after it is
built and reconstructs one layer immediately before the official model reads
it, without changing attention or checkpoint code.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Mapping

import torch

from trq.compress import compress_kv_cache, get_quantize_fn
from trq.packed_naive import packed_naive_dequantize_tensor
from trq.real.trq import trq_dequantize_tensor
from trq.sim.quant.quantize_config import QuantizeConfig


MODE_TO_QUANT_TYPE = {
    "trq_int4": "trq-int4",
    "trq-int4": "trq-int4",
    "trq_int2": "trq-int2",
    "trq-int2": "trq-int2",
    "naive_int4": "packed-naive-int4",
    "naive-int4": "packed-naive-int4",
    "packed-naive-int4": "packed-naive-int4",
    "naive_int2": "packed-naive-int2",
    "naive-int2": "packed-naive-int2",
    "packed-naive-int2": "packed-naive-int2",
}


def longcat_quant_config(
    mode: str,
    *,
    group_size: int = 64,
    predictor_stride: int = 1560,
    anchor_bits: int = 4,
) -> QuantizeConfig | None:
    """Build the five experiment modes used by the LongCat baseline."""
    normalized = mode.strip().lower()
    if normalized in {"none", "bf16"}:
        return None
    try:
        quant_type = MODE_TO_QUANT_TYPE[normalized]
    except KeyError as exc:
        supported = "bf16, trq_int4, trq_int2, naive_int4, naive_int2"
        raise ValueError(f"unsupported LongCat quantization mode {mode!r}; choose {supported}") from exc
    return QuantizeConfig(
        quant_type=quant_type,
        quant_block_size=group_size,
        trq_group_size=group_size,
        trq_anchor_bits=anchor_bits,
        trq_predictor_stride=predictor_stride,
        trq_predictor_mode="identity",
    )


def _tree_to(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _tree_to(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to(item, device) for item in value)
    return value


def _state_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_state_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_state_nbytes(item) for item in value)
    return 0


@dataclass(frozen=True)
class EncodedKVPair:
    key: dict
    value: dict
    key_dtype: torch.dtype
    value_dtype: torch.dtype


class LongCatKVCacheCodec:
    """Encode and decode one LongCat BHSD layer cache."""

    def __init__(self, config: QuantizeConfig):
        if config.quant_type not in set(MODE_TO_QUANT_TYPE.values()):
            raise ValueError(f"LongCat requires a real packed cache mode, got {config.quant_type!r}")
        self.config = config
        self.quantize_fn = get_quantize_fn(config.quant_type, config)

    def encode_pair(self, pair: tuple[torch.Tensor, torch.Tensor], layer_idx: int) -> EncodedKVPair:
        key, value = pair
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                f"LongCat KV cache must use BHSD tensors, got K={tuple(key.shape)}, V={tuple(value.shape)}"
            )
        if key.shape[:3] != value.shape[:3]:
            raise ValueError(f"LongCat K/V prefix dimensions differ: {tuple(key.shape)} vs {tuple(value.shape)}")
        packed_key, packed_value = compress_kv_cache(
            key,
            value,
            self.config.quant_type,
            self.config,
            self.quantize_fn,
            layer_idx=layer_idx,
        )
        return EncodedKVPair(packed_key, packed_value, key.dtype, value.dtype)

    def decode_pair(
        self,
        encoded: EncodedKVPair,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_state = encoded.key if device is None else _tree_to(encoded.key, device)
        value_state = encoded.value if device is None else _tree_to(encoded.value, device)
        return (
            self._decode_tensor(key_state, encoded.key_dtype),
            self._decode_tensor(value_state, encoded.value_dtype),
        )

    @staticmethod
    def _decode_tensor(state: dict, output_dtype: torch.dtype) -> torch.Tensor:
        cache_format = state.get("format")
        if cache_format in {"trq", "hrq"}:
            return trq_dequantize_tensor(state, output_dtype=output_dtype)
        if cache_format == "packed-naive":
            return packed_naive_dequantize_tensor(state, output_dtype=output_dtype)
        raise ValueError(f"unsupported encoded LongCat cache format: {cache_format!r}")

    @staticmethod
    def nbytes(encoded: EncodedKVPair) -> int:
        return _state_nbytes(encoded.key) + _state_nbytes(encoded.value)


class QuantizedLongCatCache(dict):
    """Dict-compatible cache that lazily reconstructs a layer on lookup."""

    def __init__(self, native_cache: Mapping, codec: LongCatKVCacheCodec, *, device=None):
        self.codec = codec
        self.decode_device = device
        self.native_nbytes = _state_nbytes(native_cache)
        packed = {
            layer_idx: codec.encode_pair(pair, int(layer_idx))
            for layer_idx, pair in native_cache.items()
        }
        super().__init__(packed)

    def __getitem__(self, key):
        return self.codec.decode_pair(dict.__getitem__(self, key), device=self.decode_device)

    def get(self, key, default=None):
        if not dict.__contains__(self, key):
            return default
        return self[key]

    def packed_items(self):
        """Expose encoded states for memory accounting without decoding."""
        return dict.items(self)

    def packed_nbytes(self) -> int:
        return sum(self.codec.nbytes(encoded) for _, encoded in self.packed_items())


def _infer_pipeline_device(pipeline) -> torch.device | None:
    module = getattr(pipeline, "dit", None)
    if module is None:
        return None
    try:
        return next(module.parameters()).device
    except (StopIteration, AttributeError):
        return None


def install_longcat_kv_quant(
    pipeline,
    mode: str,
    *,
    group_size: int = 64,
    predictor_stride: int = 1560,
    anchor_bits: int = 4,
    device: torch.device | str | None = None,
):
    """Install quantization around LongCat's official cache-builder method.

    The patch is instance-local and leaves the imported LongCat package
    unchanged. BF16 is a deliberate no-op.
    """
    config = longcat_quant_config(
        mode,
        group_size=group_size,
        predictor_stride=predictor_stride,
        anchor_bits=anchor_bits,
    )
    if config is None:
        return pipeline
    if getattr(pipeline, "_trq_longcat_codec", None) is not None:
        raise RuntimeError("LongCat KV quantization is already installed on this pipeline")
    if not hasattr(pipeline, "_get_kv_cache_dict"):
        raise AttributeError("LongCat pipeline has no _get_kv_cache_dict method")

    codec = LongCatKVCacheCodec(config)
    original_builder = pipeline._get_kv_cache_dict
    decode_device = device if device is not None else _infer_pipeline_device(pipeline)

    def quantized_builder(self, *args, **kwargs):
        result = original_builder(*args, **kwargs)
        native_cache = result if result is not None else getattr(self, "kv_cache_dict", None)
        if native_cache is None:
            raise RuntimeError("LongCat cache builder produced no cache dictionary")
        if isinstance(native_cache, QuantizedLongCatCache):
            return native_cache if result is not None else None
        packed_cache = QuantizedLongCatCache(native_cache, codec, device=decode_device)
        self.kv_cache_dict = packed_cache
        return packed_cache if result is not None else None

    pipeline._get_kv_cache_dict = MethodType(quantized_builder, pipeline)
    pipeline._trq_longcat_codec = codec
    pipeline._trq_longcat_mode = mode
    return pipeline
