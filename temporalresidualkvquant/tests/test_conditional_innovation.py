from pathlib import Path
from unittest import TestCase

import torch

from trq.analysis.conditional_innovation import (
    CrossKVAccumulator,
    GammaAccumulator,
    bootstrap_median_ci,
    distribution_stats,
    evaluate_layer,
    innovation_units,
)


class ConditionalInnovationTests(TestCase):
    def test_e2_launcher_initializes_mode_before_derived_output(self):
        trq_root = Path(__file__).resolve().parents[1]
        launcher = (
            trq_root / "scripts" / "analysis" / "run_e2_subgates_then_e3.sh"
        ).read_text(encoding="utf-8")
        function_body = launcher.split("run_attention_probe() {", 1)[1].split(
            "\n}", 1
        )[0]
        lines = [line.strip() for line in function_body.splitlines() if line.strip()]

        mode_index = lines.index('local mode="$1"')
        output_index = lines.index('local output="${attention_root}/${mode}"')
        self.assertLess(mode_index, output_index)

    def test_frozen_cross_then_gamma_recovers_synthetic_structure(self):
        unit_size = 8
        cross_accumulator = CrossKVAccumulator(num_heads=2, head_dim=3)
        calibration = [self._sequence(seed, unit_size) for seed in range(8)]
        for key, value in calibration:
            cross_accumulator.update(key, value)
        model = cross_accumulator.fit(ridge=1e-6)

        gamma_accumulator = GammaAccumulator(num_heads=2, head_dim=3)
        for key, value in calibration:
            gamma_accumulator.update(innovation_units(key, value, model, unit_size))
        gamma = gamma_accumulator.fit(ridge=1e-6, rho=0.95)

        key, value = self._sequence(101, unit_size)
        donor_key, donor_value = self._sequence(202, unit_size)
        donor = innovation_units(donor_key, donor_value, model, unit_size)
        rows = evaluate_layer(
            key,
            value,
            model,
            unit_size=unit_size,
            gamma=gamma,
            shuffled_innovations=donor,
            shuffle_seed=7,
        )

        self.assertLess(max(float(row["cross_over_temporal"]) for row in rows), 0.10)
        self.assertLess(max(float(row["oracle_over_cross"]) for row in rows), 0.25)
        self.assertGreater(min(float(row["shuffled_over_cross"]) for row in rows), 0.80)
        self.assertTrue(torch.allclose(gamma, torch.full_like(gamma, 0.75), atol=0.12))

    def test_cross_model_rejects_geometry_mismatch(self):
        accumulator = CrossKVAccumulator(num_heads=1, head_dim=2)
        key = torch.randn(1, 1, 4, 2)
        accumulator.update(key, key)
        model = accumulator.fit(ridge=0.0)

        with self.assertRaisesRegex(ValueError, "does not match predictor"):
            model.predict(torch.randn(1, 2, 4, 2))

    def test_bootstrap_is_prompt_level_and_deterministic(self):
        first = bootstrap_median_ci([0.7, 0.8, 0.9, 1.0], resamples=200, seed=11)
        second = bootstrap_median_ci([0.7, 0.8, 0.9, 1.0], resamples=200, seed=11)

        self.assertEqual(first, second)
        self.assertEqual(first["prompts"], 4)
        self.assertAlmostEqual(float(first["median"]), 0.85)

    def test_distribution_quantile_is_bounded_and_deterministic(self):
        generator = torch.Generator().manual_seed(23)
        chunks = [torch.randn(1, 2, 4096, 3, generator=generator) for _ in range(4)]

        first = distribution_stats(
            chunks,
            1,
            quantile=0.99,
            max_quantile_samples=257,
            seed=19,
        )
        second = distribution_stats(
            chunks,
            1,
            quantile=0.99,
            max_quantile_samples=257,
            seed=19,
        )
        exact = torch.cat([chunk[:, 1].reshape(-1) for chunk in chunks])

        self.assertEqual(first, second)
        self.assertEqual(first["count"], exact.numel())
        self.assertEqual(first["quantile_samples"], 257)
        self.assertAlmostEqual(float(first["std"]), float(exact.std(unbiased=False)), places=5)

    def test_distribution_quantile_rejects_invalid_budget(self):
        with self.assertRaisesRegex(ValueError, "max_quantile_samples"):
            distribution_stats(
                [torch.ones(1, 1, 4, 2)],
                0,
                quantile=0.99,
                max_quantile_samples=0,
            )

    @staticmethod
    def _sequence(seed: int, unit_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        heads, dim, units = 2, 3, 6
        key = torch.randn(1, heads, unit_size * units, dim, generator=generator)
        weight = torch.tensor(
            [
                [[1.4, 0.1, -0.2], [0.0, 1.1, 0.3], [0.2, -0.1, 0.9]],
                [[0.8, -0.3, 0.0], [0.2, 1.3, 0.1], [-0.1, 0.2, 1.1]],
            ]
        )
        bias = torch.tensor([[0.2, -0.1, 0.3], [-0.2, 0.1, 0.0]])
        base = torch.einsum("bhsd,hde->bhse", key, weight) + bias[None, :, None, :]
        innovation_units = []
        previous = 0.12 * torch.randn(1, heads, unit_size, dim, generator=generator)
        for _ in range(units):
            noise = 0.01 * torch.randn(1, heads, unit_size, dim, generator=generator)
            current = 0.75 * previous + noise
            innovation_units.append(current)
            previous = current
        value = base + torch.cat(innovation_units, dim=2)
        return key, value
