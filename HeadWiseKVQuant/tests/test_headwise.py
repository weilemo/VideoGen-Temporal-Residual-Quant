import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from trq.compress import compress_kv_cache, get_quantize_fn
from trq.headwise import RandomHeadPolicy, TopKHeadPolicy, compress_headwise_kv_cache, load_topk_head_policy
from trq.uncompress import uncompress_single_cache


class HeadwiseTests(TestCase):
    def test_random_policy_is_deterministic(self):
        policy_a = RandomHeadPolicy(
            num_heads=12,
            num_high_precision_heads=4,
            high_precision_quant_type="triton-nstages-kmeans-int4",
            low_precision_quant_type="triton-nstages-kmeans-int2",
            seed=7,
        )
        policy_b = RandomHeadPolicy(
            num_heads=12,
            num_high_precision_heads=4,
            high_precision_quant_type="triton-nstages-kmeans-int4",
            low_precision_quant_type="triton-nstages-kmeans-int2",
            seed=7,
        )

        self.assertEqual(policy_a.groups(), policy_b.groups())

    def test_mixed_cache_reconstructs_head_order(self):
        cache = {
            "groups": [
                {
                    "head_ids": [1, 3],
                    "quant_config": {"quant_type": "triton-nstages-kmeans-int4", "quant_block_size": 64},
                    "payload": torch.full((1, 2, 5, 7), 4.0),
                },
                {
                    "head_ids": [0, 2],
                    "quant_config": {"quant_type": "triton-nstages-kmeans-int2", "quant_block_size": 64},
                    "payload": torch.full((1, 2, 5, 7), 2.0),
                },
            ],
            "info": {"output_dtype": torch.float32, "num_heads": 4},
        }

        out = uncompress_single_cache(cache)

        self.assertEqual(out.shape, (1, 4, 5, 7))
        self.assertTrue(torch.all(out[:, [1, 3]] == 4.0))
        self.assertTrue(torch.all(out[:, [0, 2]] == 2.0))

    def test_headwise_compress_uses_expected_groups(self):
        seen = []

        def fake_get_quantize_fn(quant_type, quant_config):
            return lambda x: x

        def fake_compress_kv_cache(k, v, quant_type, quant_config, quantize_fn, **kwargs):
            seen.append((quant_type, k.shape[1]))
            return k, v

        policy = RandomHeadPolicy(
            num_heads=4,
            num_high_precision_heads=1,
            high_precision_quant_type="triton-nstages-kmeans-int4",
            low_precision_quant_type="triton-nstages-kmeans-int2",
            seed=0,
        )
        quant_config = SimpleNamespace(
            quant_type="triton-nstages-kmeans-int2",
            quant_block_size=64,
            cache_num_k_centroids=256,
            cache_num_v_centroids=256,
            kmeans_max_iters=2,
            num_prq_stages=1,
        )
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)

        with patch("trq.headwise.get_quantize_fn", fake_get_quantize_fn), patch(
            "trq.headwise.compress_kv_cache", fake_compress_kv_cache
        ):
            k_cache, v_cache = compress_headwise_kv_cache(k, v, quant_config, policy)

        self.assertEqual(
            [item[0] for item in seen],
            ["triton-nstages-kmeans-int4", "triton-nstages-kmeans-int2"],
        )
        self.assertEqual(sorted(item[1] for item in seen), [1, 3])
        self.assertEqual(k_cache["info"]["num_heads"], 4)
        self.assertEqual(v_cache["info"]["headwise_mode"], "random")

    def test_topk_policy_uses_layer_specific_heads(self):
        policy = TopKHeadPolicy(
            num_heads=4,
            high_heads_by_layer={0: (1,), 1: (2,)},
            high_precision_quant_type="packed-naive-int4",
            low_precision_quant_type="packed-naive-int2",
        )

        self.assertEqual(policy.groups(layer_idx=0)[0].head_ids, (1,))
        self.assertEqual(policy.groups(layer_idx=1)[0].head_ids, (2,))

    def test_topk_policy_loaded_from_scores_json(self):
        payload = {
            "num_heads": 4,
            "scores_by_layer": {
                "0": [0.1, 0.9, 0.2, 0.3],
                "1": [0.4, 0.5, 0.8, 0.7],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            policy = load_topk_head_policy(
                str(path),
                num_heads=4,
                num_high_precision_heads=1,
                high_precision_quant_type="packed-naive-int4",
                low_precision_quant_type="packed-naive-int2",
            )

        self.assertEqual(policy.groups(layer_idx=0)[0].head_ids, (1,))
        self.assertEqual(policy.groups(layer_idx=1)[0].head_ids, (2,))

    def test_topk_headwise_compress_uses_layer_idx(self):
        seen = []

        def fake_get_quantize_fn(quant_type, quant_config):
            return lambda x: x

        def fake_compress_kv_cache(k, v, quant_type, quant_config, quantize_fn, **kwargs):
            seen.append((quant_type, k.shape[1], k[:, :, 0, 0].flatten().tolist()))
            return k, v

        policy = TopKHeadPolicy(
            num_heads=4,
            high_heads_by_layer={3: (2,)},
            high_precision_quant_type="packed-naive-int4",
            low_precision_quant_type="packed-naive-int2",
        )
        quant_config = SimpleNamespace(
            quant_type="packed-naive-int2",
            quant_block_size=64,
            cache_num_k_centroids=256,
            cache_num_v_centroids=256,
            kmeans_max_iters=2,
            num_prq_stages=1,
        )
        k = torch.arange(1 * 4 * 2 * 1, dtype=torch.float32).reshape(1, 4, 2, 1)
        v = k.clone()

        with patch("trq.headwise.get_quantize_fn", fake_get_quantize_fn), patch(
            "trq.headwise.compress_kv_cache", fake_compress_kv_cache
        ):
            k_cache, _ = compress_headwise_kv_cache(k, v, quant_config, policy, layer_idx=3)

        self.assertEqual([item[0] for item in seen], ["packed-naive-int4", "packed-naive-int2"])
        self.assertEqual([item[1] for item in seen], [1, 3])
        self.assertEqual(k_cache["info"]["headwise_mode"], "topk")
        self.assertEqual(k_cache["info"]["layer_idx"], 3)

    def test_packed_naive_roundtrip_returns_packed_dict(self):
        quant_config = SimpleNamespace(
            quant_type="packed-naive-int2",
            quant_block_size=4,
            cache_num_k_centroids=256,
            cache_num_v_centroids=256,
            kmeans_max_iters=2,
            num_prq_stages=1,
        )
        k = torch.linspace(-1.0, 1.0, steps=30, dtype=torch.float32).reshape(1, 2, 3, 5)
        v = torch.flip(k, dims=[-1])
        quantize_fn = get_quantize_fn(quant_config.quant_type, quant_config)

        k_cache, v_cache = compress_kv_cache(k, v, quant_config.quant_type, quant_config, quantize_fn)
        self.assertIsInstance(k_cache, dict)
        self.assertEqual(k_cache["format"], "packed-naive")
        self.assertEqual(k_cache["num_bits"], 2)
        self.assertEqual(k_cache["packed_codes"].dtype, torch.uint8)
        self.assertLess(k_cache["packed_codes"].numel(), k.numel())

        k_cache["info"] = {"output_dtype": torch.float32, "quant_config": quant_config}
        v_cache["info"] = {"output_dtype": torch.float32, "quant_config": quant_config}
        k_out = uncompress_single_cache(k_cache)
        v_out = uncompress_single_cache(v_cache)

        self.assertEqual(k_out.shape, k.shape)
        self.assertEqual(v_out.shape, v.shape)
        self.assertLess(torch.max(torch.abs(k - k_out)).item(), 0.35)
        self.assertLess(torch.max(torch.abs(v - v_out)).item(), 0.35)

    def test_packed_naive_supports_int4_and_int8(self):
        x = torch.randn(1, 1, 2, 9)
        for quant_type in ("packed-naive-int4", "packed-naive-int8"):
            quant_config = SimpleNamespace(
                quant_type=quant_type,
                quant_block_size=8,
                cache_num_k_centroids=256,
                cache_num_v_centroids=256,
                kmeans_max_iters=2,
                num_prq_stages=1,
            )
            quantize_fn = get_quantize_fn(quant_config.quant_type, quant_config)
            k_cache, _ = compress_kv_cache(x, x, quant_config.quant_type, quant_config, quantize_fn)
            k_cache["info"] = {"output_dtype": torch.float32, "quant_config": quant_config}

            out = uncompress_single_cache(k_cache)

            self.assertEqual(out.shape, x.shape)
            self.assertEqual(k_cache["format"], "packed-naive")
