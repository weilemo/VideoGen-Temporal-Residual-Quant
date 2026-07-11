import math
from unittest import TestCase

import torch

from trq.analysis.trq_diagnostics import (
    DistributionAccumulator,
    aggregate_error_records,
    trace_temporal_residual_quantization,
)


class TRQDiagnosticsTests(TestCase):
    def test_distribution_accumulator_tracks_exact_moments(self):
        accumulator = DistributionAccumulator(sample_capacity=4, seed=3)
        accumulator.update(torch.tensor([-2.0, -1.0, 0.0]))
        accumulator.update(torch.tensor([1.0, 2.0]))
        summary = accumulator.summary()

        self.assertEqual(summary["count"], 5)
        self.assertAlmostEqual(summary["mean"], 0.0)
        self.assertAlmostEqual(summary["rms"], math.sqrt(2.0))
        self.assertLessEqual(accumulator.samples.size, 4)

    def test_correlated_kv_has_concentrated_chained_residuals(self):
        generator = torch.Generator().manual_seed(7)
        chunks = [torch.randn(1, 2, 8, 7, generator=generator)]
        for _ in range(7):
            noise = 0.05 * torch.randn(1, 2, 8, 7, generator=generator)
            chunks.append(0.98 * chunks[-1] + noise)
        tensor = torch.cat(chunks, dim=2)
        raw = DistributionAccumulator(sample_capacity=20_000, seed=1)
        residual = DistributionAccumulator(sample_capacity=20_000, seed=2)

        def collect(kind, values):
            if kind == "raw":
                raw.update(values)
            elif kind == "residual":
                residual.update(values)

        records = trace_temporal_residual_quantization(
            tensor,
            num_bits=2,
            block_size=4,
            anchor_bits=4,
            predictor_stride=8,
            codec_dtype=torch.float32,
            scale_precision=torch.float32,
            value_callback=collect,
        )
        raw_summary = raw.summary()
        residual_summary = residual.summary()
        errors = aggregate_error_records(records[1:])

        self.assertLess(residual_summary["std"], 0.25 * raw_summary["std"])
        self.assertLess(residual_summary["p99_abs"], 0.35 * raw_summary["p99_abs"])
        self.assertGreater(errors["raw_over_residual_error_gain"], 1.0)
        self.assertEqual(errors["codec_parity_max_abs"], 0.0)

    def test_chain_trace_has_expected_rollout_units(self):
        base = torch.linspace(-1.0, 1.0, 8).reshape(1, 1, 1, 8)
        chunks = [base.repeat(1, 1, 4, 1) + step * 0.02 for step in range(6)]
        records = trace_temporal_residual_quantization(
            torch.cat(chunks, dim=2),
            num_bits=2,
            block_size=4,
            anchor_bits=4,
            predictor_stride=4,
            codec_dtype=torch.float32,
            scale_precision=torch.float32,
        )

        self.assertEqual([record["step"] for record in records], list(range(6)))
        self.assertEqual([record["start_token"] for record in records], [0, 4, 8, 12, 16, 20])
        self.assertTrue(all(record["codec_parity_max_abs"] == 0.0 for record in records))
        self.assertTrue(all(math.isfinite(record["chain_rel_l2"]) for record in records))

    def test_reset_interval_creates_independent_spans_and_excludes_anchors(self):
        tensor = torch.randn(1, 1, 24, 8, generator=torch.Generator().manual_seed(11))
        observed_values = {"raw": 0, "residual": 0}

        def collect(kind, values):
            if kind in observed_values:
                observed_values[kind] += values.numel()

        records = trace_temporal_residual_quantization(
            tensor,
            num_bits=2,
            block_size=4,
            anchor_bits=4,
            predictor_stride=4,
            reset_interval_units=2,
            codec_dtype=torch.float32,
            scale_precision=torch.float32,
            value_callback=collect,
        )

        self.assertEqual([record["span_id"] for record in records], [0, 0, 1, 1, 2, 2])
        self.assertEqual([record["chain_position"] for record in records], [0, 1, 0, 1, 0, 1])
        self.assertEqual([record["rollout_unit"] for record in records], list(range(6)))
        self.assertEqual(observed_values["raw"], 3 * 1 * 1 * 4 * 8)
        self.assertEqual(observed_values["residual"], observed_values["raw"])
        self.assertTrue(all(record["decoder_parity_max_abs"] == 0.0 for record in records))

    def test_explicit_runtime_spans_can_exclude_unquantized_tail(self):
        tensor = torch.randn(1, 1, 32, 8, generator=torch.Generator().manual_seed(13))
        records = trace_temporal_residual_quantization(
            tensor,
            num_bits=2,
            block_size=4,
            anchor_bits=4,
            predictor_stride=4,
            span_unit_lengths=[3, 4],
            codec_dtype=torch.float32,
            scale_precision=torch.float32,
        )

        self.assertEqual([record["span_id"] for record in records], [0, 0, 0, 1, 1, 1, 1])
        self.assertEqual([record["chain_position"] for record in records], [0, 1, 2, 0, 1, 2, 3])
        self.assertEqual(records[-1]["end_token"], 28)

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            trace_temporal_residual_quantization(
                tensor,
                predictor_stride=4,
                reset_interval_units=2,
                span_unit_lengths=[3],
            )
