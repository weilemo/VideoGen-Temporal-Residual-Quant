import pytest
import torch

from trq.backends.causal_forcing import (
    CausalChunkedKVCache,
    create_cache,
    packed_cache_bytes,
    quantize_cache_pair,
    reset_cache_pair,
)


def _cache(quant_type: str):
    return create_cache(
        batch_size=1,
        frame_seq_length=4,
        num_heads=2,
        head_dim=8,
        max_frames=6,
        dtype=torch.float32,
        device=torch.device("cpu"),
        quant_type=quant_type,
    )


def test_bf16_mode_keeps_native_tensor_cache():
    cache = _cache("none")
    assert isinstance(cache, torch.Tensor)
    assert cache.shape == (1, 24, 2, 8)


@pytest.mark.parametrize(
    "quant_type",
    ["trq-int4", "trq-int2", "packed-naive-int4", "packed-naive-int2"],
)
def test_packed_cache_round_trip_and_accounting(quant_type):
    key_cache = _cache(quant_type)
    value_cache = _cache(quant_type)
    assert isinstance(key_cache, CausalChunkedKVCache)

    key = torch.linspace(-1, 1, 128).reshape(1, 8, 2, 8)
    value = torch.flip(key, dims=(1,))
    key_cache[:, 0:8] = key
    value_cache[:, 0:8] = value
    layer = {"k": key_cache, "v": value_cache}

    quantize_cache_pair(
        layer,
        0,
        8,
        meta={
            "kv_quant_type": quant_type,
            "kv_quant_block_size": 4,
            "trq_anchor_bits": 4,
            "trq_predictor_stride": 4,
            "trq_predictor_mode": "identity",
        },
        layer_idx=0,
    )

    decoded_key = key_cache[:, 0:8]
    decoded_value = value_cache[:, 0:8]
    assert decoded_key.shape == key.shape
    assert decoded_value.shape == value.shape
    assert torch.isfinite(decoded_key).all()
    assert torch.isfinite(decoded_value).all()
    native, packed = packed_cache_bytes([layer])
    assert native == key.numel() * key.element_size() * 2
    assert 0 < packed < native

    reset_cache_pair(layer)
    assert not key_cache.quantized_spans
    assert not value_cache.quantized_spans
