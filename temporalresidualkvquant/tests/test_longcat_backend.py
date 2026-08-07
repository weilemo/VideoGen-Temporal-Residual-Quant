import pytest
import torch

from trq.backends.longcat_video import (
    LongCatKVCacheCodec,
    QuantizedLongCatCache,
    install_longcat_kv_quant,
    longcat_quant_config,
)


@pytest.mark.parametrize("mode", ["trq_int4", "trq_int2", "naive_int4", "naive_int2"])
def test_longcat_codec_round_trip_bhsd(mode):
    torch.manual_seed(7)
    pair = (
        torch.randn(1, 2, 16, 64, dtype=torch.bfloat16),
        torch.randn(1, 2, 16, 64, dtype=torch.bfloat16),
    )
    config = longcat_quant_config(mode, group_size=16, predictor_stride=4)
    codec = LongCatKVCacheCodec(config)
    encoded = codec.encode_pair(pair, layer_idx=3)
    decoded = codec.decode_pair(encoded)

    assert decoded[0].shape == pair[0].shape
    assert decoded[1].shape == pair[1].shape
    assert decoded[0].dtype == torch.bfloat16
    assert decoded[1].dtype == torch.bfloat16
    assert torch.isfinite(decoded[0]).all()
    assert torch.isfinite(decoded[1]).all()
    assert codec.nbytes(encoded) < sum(t.numel() * t.element_size() for t in pair)


def test_quantized_longcat_cache_decodes_on_lookup():
    config = longcat_quant_config("naive_int4", group_size=16)
    codec = LongCatKVCacheCodec(config)
    pair = (torch.randn(1, 1, 8, 64), torch.randn(1, 1, 8, 64))
    cache = QuantizedLongCatCache({0: pair}, codec)

    assert cache.get(99) is None
    key, value = cache[0]
    assert key.shape == pair[0].shape
    assert value.shape == pair[1].shape
    assert cache.packed_nbytes() > 0


def test_install_hook_wraps_official_cache_builder_contract():
    pair = (torch.randn(1, 1, 8, 64), torch.randn(1, 1, 8, 64))

    class DummyPipeline:
        def _get_kv_cache_dict(self):
            self.kv_cache_dict = {0: pair}
            return self.kv_cache_dict

    pipeline = install_longcat_kv_quant(
        DummyPipeline(), "naive_int2", group_size=16, device="cpu"
    )
    cache = pipeline._get_kv_cache_dict()

    assert isinstance(cache, QuantizedLongCatCache)
    assert isinstance(pipeline.kv_cache_dict, QuantizedLongCatCache)
    assert cache[0][0].shape == pair[0].shape


def test_bf16_install_is_noop():
    pipeline = object()
    assert install_longcat_kv_quant(pipeline, "bf16") is pipeline
