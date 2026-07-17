from unittest import TestCase

import torch

from trq.kv_cache import ChunkedKVCache
from trq.real.trq import trq_quantize_tensor
from trq.runtime_metrics import (
    nested_tensor_nbytes,
    summarize_cache_profile,
    summarize_kv_cache,
)


class RuntimeMetricsTests(TestCase):
    def test_raw_cache_summary_counts_filled_values_and_bytes(self):
        k_cache = self._cache()
        v_cache = self._cache()
        values = torch.ones(1, 6, 2, 4, dtype=torch.bfloat16)
        k_cache.write(0, 6, values)
        v_cache.write(0, 6, values)

        summary = summarize_kv_cache([{"k": k_cache, "v": v_cache}])

        expected_values = 2 * values.numel()
        expected_bytes = expected_values * values.element_size()
        self.assertEqual(summary["total"]["represented_values"], expected_values)
        self.assertEqual(summary["total"]["physical_bytes"], expected_bytes)
        self.assertEqual(summary["total"]["bf16_equivalent_bytes"], expected_bytes)
        self.assertEqual(summary["total"]["effective_bits_per_value"], 16.0)
        self.assertEqual(summary["total"]["compression_ratio"], 1.0)

    def test_quantized_cache_summary_uses_packed_state_payload(self):
        k_cache = self._cache()
        v_cache = self._cache()
        source = torch.randn(1, 2, 6, 4)
        k_state = trq_quantize_tensor(
            source, num_bits=2, block_size=4, predictor_stride=3
        )
        v_state = trq_quantize_tensor(
            source, num_bits=4, block_size=4, predictor_stride=3
        )
        k_cache.store_quantized(0, 6, k_state)
        v_cache.store_quantized(0, 6, v_state)

        summary = summarize_kv_cache([{"k": k_cache, "v": v_cache}])

        self.assertEqual(summary["k"]["physical_bytes"], nested_tensor_nbytes(k_state))
        self.assertEqual(summary["v"]["physical_bytes"], nested_tensor_nbytes(v_state))
        self.assertEqual(summary["total"]["represented_values"], 2 * source.numel())
        self.assertGreater(summary["total"]["compression_ratio"], 1.0)

    def test_profile_counts_quantized_span_decodes_on_cpu(self):
        cache = self._cache()
        source = torch.randn(1, 2, 6, 4)
        state = trq_quantize_tensor(
            source, num_bits=2, block_size=4, predictor_stride=3
        )
        cache.store_quantized(0, 6, state)
        cache.enable_profiling()

        decoded = cache.read(0, 6)
        profile = summarize_cache_profile([{"k": cache}], synchronize=False)

        self.assertEqual(decoded.shape, (1, 6, 2, 4))
        self.assertEqual(profile["dequantize_calls"], 1)
        self.assertGreater(profile["dequantize_ms"], 0.0)

    def test_eviction_decodes_only_a_cut_quantized_span(self):
        cache = ChunkedKVCache(
            batch_size=1,
            frame_seq_length=1,
            num_heads=1,
            head_dim=4,
            max_num_chunks=6,
            dtype=torch.float32,
            device=torch.device("cpu"),
            layout="BSHD",
        )
        quant_source = torch.randn(1, 1, 3, 4)
        quant_state = trq_quantize_tensor(
            quant_source, num_bits=4, block_size=4, predictor_stride=1
        )
        cache.store_quantized(0, 3, quant_state)
        raw_tail = torch.randn(1, 3, 1, 4)
        cache.write(3, 6, raw_tail)
        expected = cache.read(1, 6).clone()

        cache.evict_prefix(1)

        self.assertTrue(torch.equal(cache.read(0, 5), expected))
        self.assertEqual(len(cache.quantized_spans), 0)

    def test_eviction_shifts_an_intact_quantized_span_without_decoding(self):
        cache = ChunkedKVCache(
            batch_size=1,
            frame_seq_length=1,
            num_heads=1,
            head_dim=4,
            max_num_chunks=5,
            dtype=torch.float32,
            device=torch.device("cpu"),
            layout="BSHD",
        )
        cache.write(0, 1, torch.randn(1, 1, 1, 4))
        quant_source = torch.randn(1, 1, 3, 4)
        quant_state = trq_quantize_tensor(
            quant_source, num_bits=4, block_size=4, predictor_stride=1
        )
        cache.store_quantized(1, 4, quant_state)
        expected = cache.read(1, 4).clone()
        original_state = cache.quantized_spans[0]["quant_data"]

        cache.evict_prefix(1)

        self.assertTrue(torch.equal(cache.read(0, 3), expected))
        self.assertEqual(len(cache.quantized_spans), 1)
        self.assertIs(cache.quantized_spans[0]["quant_data"], original_state)
        self.assertEqual(cache.quantized_spans[0]["start_chunk"], 0)
        self.assertEqual(cache.quantized_spans[0]["end_chunk"], 3)

    @staticmethod
    def _cache() -> ChunkedKVCache:
        return ChunkedKVCache(
            batch_size=1,
            frame_seq_length=3,
            num_heads=2,
            head_dim=4,
            max_num_chunks=2,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            layout="BSHD",
        )
