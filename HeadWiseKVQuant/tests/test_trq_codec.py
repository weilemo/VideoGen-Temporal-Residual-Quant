from copy import deepcopy
from unittest import TestCase

import torch

from trq.real.hrq import hrq_dequantize_tensor
from trq.real.trq import trq_dequantize_tensor, trq_quantize_tensor, trq_state_nbytes


class TRQCodecTests(TestCase):
    def test_roundtrip_contract_and_encoder_decoder_parity(self):
        x = torch.randn(1, 2, 5, 7, dtype=torch.float32)
        packed, encoder_recon = trq_quantize_tensor(
            x, num_bits=2, block_size=3, anchor_bits=4,
            predictor_stride=2, return_reconstruction=True,
        )
        decoded = trq_dequantize_tensor(packed, output_dtype=torch.float32)

        self.assertEqual(packed["format"], "trq")
        self.assertEqual(packed["version"], 1)
        self.assertEqual(packed["dependency"], "previous_reconstruction")
        self.assertEqual(decoded.shape, x.shape)
        self.assertTrue(torch.equal(decoded, encoder_recon))
        self.assertLess(torch.norm(x - decoded).item() / torch.norm(x).item(), 0.8)

    def test_short_span_and_partial_final_unit(self):
        short = torch.randn(1, 1, 3, 5)
        short_state = trq_quantize_tensor(
            short, num_bits=2, block_size=4, anchor_bits=4, predictor_stride=8
        )
        self.assertIsNone(short_state["residual_quant"])
        self.assertEqual(short_state["unit_lengths"], [3])
        self.assertEqual(trq_dequantize_tensor(short_state, output_dtype=torch.float32).shape, short.shape)

        partial = torch.randn(1, 1, 7, 9)
        partial_state = trq_quantize_tensor(
            partial, num_bits=4, block_size=4, anchor_bits=4, predictor_stride=3
        )
        self.assertEqual(partial_state["unit_lengths"], [3, 3, 1])
        self.assertEqual(trq_dequantize_tensor(partial_state, output_dtype=torch.float32).shape, partial.shape)

    def test_affine_is_self_contained_for_all_supported_shapes(self):
        x = torch.randn(1, 2, 6, 8)
        variants = [
            ({"alpha": torch.full((8,), 0.9), "beta": torch.full((8,), 0.1)}, None),
            ({"alpha": torch.full((2, 8), 0.9), "beta": torch.full((2, 8), 0.1)}, None),
            ({"alpha": torch.full((3, 2, 8), 0.9), "beta": torch.full((3, 2, 8), 0.1)}, 1),
        ]

        for params, layer_idx in variants:
            with self.subTest(shape=tuple(params["alpha"].shape)):
                state, encoder_recon = trq_quantize_tensor(
                    x, num_bits=4, block_size=4, predictor_stride=2,
                    predictor_mode="affine_channel", predictor_params=params,
                    layer_idx=layer_idx, return_reconstruction=True,
                )
                self.assertIn("alpha", state["predictor"]["params"])
                decoded = trq_dequantize_tensor(state, output_dtype=torch.float32)
                self.assertTrue(torch.equal(decoded, encoder_recon))

    def test_affine_missing_params_fails_closed(self):
        x = torch.randn(1, 2, 4, 8)
        with self.assertRaisesRegex(ValueError, "requires alpha and beta"):
            trq_quantize_tensor(
                x, num_bits=2, block_size=4, predictor_stride=2,
                predictor_mode="affine_channel",
            )

        state = trq_quantize_tensor(
            x, num_bits=2, block_size=4, predictor_stride=2,
            predictor_mode="affine_channel",
            predictor_params={"alpha": torch.ones(2, 8), "beta": torch.zeros(2, 8)},
        )
        broken = deepcopy(state)
        del broken["predictor"]["params"]["alpha"]
        with self.assertRaisesRegex(ValueError, "missing alpha or beta"):
            trq_dequantize_tensor(broken, output_dtype=torch.float32)

    def test_rope_is_explicitly_experimental(self):
        with self.assertRaisesRegex(NotImplementedError, "RoPE prediction is experimental"):
            trq_quantize_tensor(
                torch.randn(1, 1, 4, 8), num_bits=2, block_size=4,
                predictor_stride=2, predictor_mode="rope_affine",
            )

    def test_state_byte_counter_counts_physical_storage(self):
        x = torch.randn(1, 2, 33, 9)
        state = trq_quantize_tensor(x, num_bits=2, block_size=3, predictor_stride=3)
        self.assertGreater(trq_state_nbytes(state), 0)
        self.assertLess(trq_state_nbytes(state), x.numel() * x.element_size())

    def test_legacy_hrq_identity_state_remains_decodable(self):
        x = torch.randn(1, 1, 5, 8)
        state = trq_quantize_tensor(x, num_bits=2, block_size=4, predictor_stride=2)
        legacy = dict(state)
        legacy["format"] = "hrq"
        legacy.pop("version")
        legacy.pop("predictor")
        decoded = hrq_dequantize_tensor(legacy, output_dtype=torch.float32)
        self.assertEqual(decoded.shape, x.shape)

    def test_legacy_s2pp_affine_state_remains_decodable(self):
        x = torch.randn(1, 2, 5, 8)
        params = {"alpha": torch.full((8,), 0.9), "beta": torch.full((8,), 0.1)}
        state = trq_quantize_tensor(
            x, num_bits=4, block_size=4, predictor_stride=2,
            predictor_mode="affine_channel", predictor_params=params,
        )
        legacy = dict(state)
        legacy.pop("format")
        legacy.pop("version")
        legacy.pop("layout")
        legacy.pop("predictor")
        legacy["method"] = "s2pp"
        legacy["predictor_kind"] = "affine"
        legacy["alpha"] = params["alpha"]
        legacy["beta"] = params["beta"]
        decoded = trq_dequantize_tensor(legacy, output_dtype=torch.float32)
        self.assertEqual(decoded.shape, x.shape)
