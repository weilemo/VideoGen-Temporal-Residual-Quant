from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase

import numpy as np
import torch

from trq.compress import compress_kv_cache, get_quantize_fn
from trq.headwise import RandomHeadPolicy, compress_headwise_kv_cache
from trq.kv_cache import ChunkedKVCache
from trq.uncompress import uncompress_single_cache


class TRQIntegrationTests(TestCase):
    @staticmethod
    def _config(quant_type="trq-int2"):
        return SimpleNamespace(
            quant_type=quant_type,
            quant_block_size=4,
            trq_group_size=4,
            trq_anchor_bits=4,
            trq_predictor_stride=2,
            trq_predictor_mode="identity",
            trq_scale_precision="bf16",
            trq_residual_quant_mode="asym_zero_point",
            trq_k_bits=0,
            trq_v_bits=0,
        )

    def test_compress_supports_new_and_legacy_quant_type_aliases(self):
        x = torch.randn(1, 2, 5, 7)
        for quant_type in ("trq-int2", "hrq-int2", "s2pp-int2"):
            with self.subTest(quant_type=quant_type):
                cfg = self._config(quant_type)
                fn = get_quantize_fn(cfg.quant_type, cfg)
                k_cache, _ = compress_kv_cache(x, x, cfg.quant_type, cfg, fn)
                self.assertEqual(k_cache["format"], "trq")
                k_cache["info"] = {"output_dtype": torch.float32, "quant_config": cfg}
                decoded = uncompress_single_cache(k_cache)
                self.assertEqual(decoded.shape, x.shape)
                self.assertEqual(decoded.dtype, torch.float32)

    def test_qvg_s2pp_config_and_npz_affine_path_are_accepted(self):
        x = torch.randn(1, 2, 5, 8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "affine.npz"
            np.savez(path, alpha=np.full((8,), 0.9, np.float32), beta=np.full((8,), 0.1, np.float32))
            cfg = SimpleNamespace(
                quant_type="s2pp-int4",
                quant_block_size=4,
                s2pp_group_size=4,
                s2pp_anchor_bits=8,
                s2pp_predictor_stride=2,
                s2pp_predictor_mode="auto",
                s2pp_v_predictor_mode="auto",
                s2pp_affine_path=str(path),
                s2pp_v_affine_path=str(path),
                s2pp_scale_precision="bf16",
                s2pp_residual_quant_mode="asym_zero_point",
                s2pp_k_bits=0,
                s2pp_v_bits=0,
            )
            fn = get_quantize_fn(cfg.quant_type, cfg)
            k_cache, v_cache = compress_kv_cache(x, x, cfg.quant_type, cfg, fn)

        self.assertEqual(k_cache["predictor"]["kind"], "affine_channel")
        self.assertEqual(v_cache["predictor"]["kind"], "affine_channel")
        self.assertEqual(k_cache["anchor_bits"], 8)

    def test_k_and_v_residual_bits_can_differ(self):
        cfg = self._config()
        cfg.trq_k_bits = 4
        cfg.trq_v_bits = 2
        x = torch.randn(1, 2, 5, 8)
        fn = get_quantize_fn(cfg.quant_type, cfg)
        k_cache, v_cache = compress_kv_cache(x, x, cfg.quant_type, cfg, fn)
        self.assertEqual(k_cache["num_bits"], 4)
        self.assertEqual(v_cache["num_bits"], 2)

    def test_repository_affine_parameters_produce_portable_states(self):
        cfg = self._config()
        cfg.trq_predictor_mode = "affine_channel"
        cfg.trq_predictor_params_path = str(
            Path(__file__).parents[1] / "assets/trq_predictors/affine_channel_self_forcing_dmd.pt"
        )
        x = torch.randn(1, 12, 4, 128)
        fn = get_quantize_fn(cfg.quant_type, cfg)
        k_cache, v_cache = compress_kv_cache(
            x, x, cfg.quant_type, cfg, fn, layer_idx=0
        )

        self.assertIn("alpha", k_cache["predictor"]["params"])
        self.assertIn("alpha", v_cache["predictor"]["params"])
        self.assertEqual(uncompress_single_cache(k_cache).shape, x.shape)
        self.assertEqual(uncompress_single_cache(v_cache).shape, x.shape)

    def test_trq_state_survives_chunk_cache_offload_onload(self):
        cfg = self._config()
        x = torch.randn(1, 2, 6, 8)
        fn = get_quantize_fn(cfg.quant_type, cfg)
        k_cache, _ = compress_kv_cache(x, x, cfg.quant_type, cfg, fn)
        cache = ChunkedKVCache(
            batch_size=1,
            frame_seq_length=2,
            num_heads=2,
            head_dim=8,
            max_num_chunks=3,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        cache.store_quantized(0, 6, k_cache)
        cache.offload()
        cache.onload(torch.device("cpu"))
        self.assertEqual(cache.read(0, 6).shape, x.shape)

    def test_headwise_mixed_groups_restore_shape(self):
        cfg = self._config()
        policy = RandomHeadPolicy(
            num_heads=4,
            num_high_precision_heads=1,
            high_precision_quant_type="trq-int4",
            low_precision_quant_type="trq-int2",
            seed=0,
        )
        k = torch.randn(1, 4, 5, 7)
        v = torch.randn(1, 4, 5, 7)
        k_cache, v_cache = compress_headwise_kv_cache(k, v, cfg, policy)
        self.assertEqual(uncompress_single_cache(k_cache).shape, k.shape)
        self.assertEqual(uncompress_single_cache(v_cache).shape, v.shape)
        self.assertEqual(k_cache["info"]["num_heads"], 4)
