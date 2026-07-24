import pytest
import torch

from trq.backends.causal_forcing import (
    CausalChunkedKVCache,
    create_cache,
    packed_cache_bytes,
    quantize_cache_pair,
    reset_cache_pair,
    write_cache,
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


@pytest.mark.parametrize("quant_type", ["none", "trq-int4"])
@pytest.mark.parametrize("max_frames", [21, 42, 84])
def test_cache_capacity_tracks_requested_frames(quant_type, max_frames):
    cache = create_cache(
        batch_size=1,
        frame_seq_length=4,
        num_heads=2,
        head_dim=8,
        max_frames=max_frames,
        dtype=torch.float32,
        device=torch.device("cpu"),
        quant_type=quant_type,
    )
    assert cache.shape[1] == max_frames * 4


@pytest.mark.parametrize("quant_type", ["none", "trq-int4"])
def test_cache_write_reports_capacity_and_token_mismatch(quant_type):
    cache = _cache(quant_type)
    value = torch.zeros((1, 8, 2, 8))
    with pytest.raises(
        ValueError,
        match=r"start=20, end=28, capacity=24, value_tokens=8",
    ):
        write_cache(cache, 20, 28, value)


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
