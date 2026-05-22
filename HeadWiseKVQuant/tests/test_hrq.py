from types import SimpleNamespace
from unittest import TestCase

import torch

from hwq.compress import compress_kv_cache, get_quantize_fn
from hwq.headwise import RandomHeadPolicy, compress_headwise_kv_cache
from hwq.real.hrq import hrq_dequantize_tensor, hrq_quantize_tensor
from hwq.uncompress import uncompress_single_cache


class HRQTests(TestCase):
    def test_hrq_roundtrip_shape_dtype_device_and_format(self):
        x = torch.randn(1, 2, 5, 7, dtype=torch.float32)
        packed = hrq_quantize_tensor(x, num_bits=2, block_size=4, anchor_bits=4, predictor_stride=2)
        out = hrq_dequantize_tensor(packed, output_dtype=torch.float32)

        self.assertEqual(packed["format"], "hrq")
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.device, x.device)
        self.assertLess(torch.norm(x - out).item() / torch.norm(x).item(), 0.8)

    def test_hrq_short_span_only_anchor(self):
        x = torch.randn(1, 1, 3, 5)
        packed = hrq_quantize_tensor(x, num_bits=2, block_size=4, anchor_bits=4, predictor_stride=8)
        out = hrq_dequantize_tensor(packed, output_dtype=torch.float32)

        self.assertIsNone(packed["residual_quant"])
        self.assertEqual(packed["unit_lengths"], [3])
        self.assertEqual(out.shape, x.shape)

    def test_hrq_non_multiple_stride(self):
        x = torch.randn(1, 1, 7, 9)
        packed = hrq_quantize_tensor(x, num_bits=4, block_size=4, anchor_bits=4, predictor_stride=3)
        out = hrq_dequantize_tensor(packed, output_dtype=torch.float32)

        self.assertEqual(packed["unit_lengths"], [3, 3, 1])
        self.assertEqual(out.shape, x.shape)

    def test_compress_kv_cache_supports_hrq(self):
        cfg = SimpleNamespace(
            quant_type="hrq-int2", quant_block_size=4, hrq_group_size=4,
            hrq_anchor_bits=4, hrq_predictor_stride=2, hrq_predictor_mode="identity",
            hrq_scale_precision="bf16", hrq_residual_quant_mode="asym_zero_point",
        )
        x = torch.randn(1, 2, 5, 7)
        fn = get_quantize_fn(cfg.quant_type, cfg)
        k_cache, v_cache = compress_kv_cache(x, x, cfg.quant_type, cfg, fn)

        self.assertEqual(k_cache["format"], "hrq")
        k_cache["info"] = {"output_dtype": torch.float32, "quant_config": cfg}
        self.assertEqual(uncompress_single_cache(k_cache).shape, x.shape)

    def test_headwise_hrq_mixed_groups_restore_shape(self):
        cfg = SimpleNamespace(
            quant_type="hrq-int2", quant_block_size=4, hrq_group_size=4,
            hrq_anchor_bits=4, hrq_predictor_stride=2, hrq_predictor_mode="identity",
            hrq_scale_precision="bf16", hrq_residual_quant_mode="asym_zero_point",
        )
        policy = RandomHeadPolicy(
            num_heads=4, num_high_precision_heads=1,
            high_precision_quant_type="hrq-int4", low_precision_quant_type="hrq-int2", seed=0,
        )
        k = torch.randn(1, 4, 5, 7)
        v = torch.randn(1, 4, 5, 7)
        k_cache, v_cache = compress_headwise_kv_cache(k, v, cfg, policy)
        k_out = uncompress_single_cache(k_cache)
        v_out = uncompress_single_cache(v_cache)

        self.assertEqual(k_out.shape, k.shape)
        self.assertEqual(v_out.shape, v.shape)
        self.assertEqual(k_cache["info"]["num_heads"], 4)

