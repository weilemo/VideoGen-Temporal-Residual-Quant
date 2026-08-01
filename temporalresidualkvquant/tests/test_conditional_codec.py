from unittest import TestCase

import torch

from trq.analysis.conditional_codec import (
    _update_quantizer_diagnostics,
    reconstruct_closed_loop_innovations,
    simulate_conditional_codec,
)
from trq.analysis.conditional_innovation import CrossKVModel


class ConditionalCodecTests(TestCase):
    def test_s2pp_ranges_do_not_require_preclamp_overflow(self):
        tensor = torch.tensor(
            [[[[-1.0, -0.91, -0.2, 0.13], [0.0, 0.07, 0.5, 1.05]]]],
            dtype=torch.float32,
        )
        for symmetric in (True, False):
            with self.subTest(symmetric=symmetric):
                diagnostics = {}
                _update_quantizer_diagnostics(
                    diagnostics,
                    tensor,
                    bits=4,
                    block_size=4,
                    scale_precision=torch.bfloat16,
                    symmetric=symmetric,
                )
                self.assertEqual(diagnostics["overflow"], 0)

    def test_closed_loop_codec_reports_fixed_bit_baselines(self):
        generator = torch.Generator().manual_seed(17)
        key = torch.randn(1, 2, 30, 8, generator=generator)
        innovations = []
        previous = torch.zeros(1, 2, 6, 8)
        for _ in range(5):
            previous = 0.8 * previous + 0.08 * torch.randn(
                1, 2, 6, 8, generator=generator
            )
            innovations.append(previous)
        value = key + torch.cat(innovations, dim=2)
        model = CrossKVModel(
            weight=torch.eye(8).expand(2, 8, 8).clone(),
            bias=torch.zeros(2, 8),
        )
        donor_key = torch.randn(1, 2, 30, 8, generator=generator)
        donor_value = donor_key + 0.2 * torch.randn(
            1, 2, 30, 8, generator=generator
        )
        donor = reconstruct_closed_loop_innovations(
            donor_key,
            donor_value,
            model,
            torch.full((2, 8), 0.8),
            unit_size=6,
            key_bits=2,
            value_bits=2,
            anchor_bits=4,
            block_size=4,
            scale_precision=torch.float32,
        )

        result = simulate_conditional_codec(
            key,
            value,
            model,
            torch.full((2, 8), 0.8),
            unit_size=6,
            key_bits=2,
            value_bits=2,
            anchor_bits=4,
            block_size=4,
            reset_spans=(2, 4),
            scale_precision=torch.float32,
            shuffled_innovations=donor,
        )

        self.assertEqual(result["complete_units"], 5)
        self.assertEqual(result["ignored_tokens"], 0)
        self.assertIn("closed_hybrid", result["methods"])
        self.assertIn("closed_hybrid_reset_2", result["methods"])
        self.assertTrue(result["methods"]["closed_hybrid"]["finite"])
        self.assertGreater(result["methods"]["closed_hybrid"]["physical_bytes"], 0)
        self.assertEqual(len(result["unit_rows"]), 5 * 8)

    def test_codec_ignores_only_the_final_partial_unit(self):
        key = torch.zeros(1, 1, 10, 4)
        value = torch.zeros_like(key)
        model = CrossKVModel(weight=torch.eye(4)[None], bias=torch.zeros(1, 4))
        donor = [torch.zeros(1, 1, 4, 4)]
        result = simulate_conditional_codec(
            key,
            value,
            model,
            torch.zeros(1, 4),
            unit_size=4,
            block_size=4,
            reset_spans=(2,),
            scale_precision=torch.float32,
            shuffled_innovations=donor,
        )
        self.assertEqual(result["complete_units"], 2)
        self.assertEqual(result["ignored_tokens"], 2)

    def test_codec_rejects_gamma_geometry_mismatch(self):
        tensor = torch.zeros(1, 2, 8, 4)
        model = CrossKVModel(
            weight=torch.eye(4).expand(2, 4, 4).clone(),
            bias=torch.zeros(2, 4),
        )
        with self.assertRaisesRegex(ValueError, "gamma"):
            simulate_conditional_codec(
                tensor,
                tensor,
                model,
                torch.zeros(1, 4),
                unit_size=4,
                block_size=4,
                reset_spans=(2,),
                shuffled_innovations=[torch.zeros(1, 2, 4, 4)],
            )

    def test_codec_requires_prompt_disjoint_reconstructed_donor(self):
        tensor = torch.zeros(1, 1, 8, 4)
        model = CrossKVModel(weight=torch.eye(4)[None], bias=torch.zeros(1, 4))
        with self.assertRaisesRegex(ValueError, "prompt-disjoint donor"):
            simulate_conditional_codec(
                tensor,
                tensor,
                model,
                torch.zeros(1, 4),
                unit_size=4,
                block_size=4,
                reset_spans=(2,),
            )
