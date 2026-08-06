import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from trq.analysis.conditional_increment import (
    aggregate_prompt_rows,
    conditional_feature_batches,
    evaluate_conditional_record,
    fit_conditional_models,
)


class ConditionalIncrementTests(TestCase):
    def test_aligned_current_key_has_incremental_predictive_gain(self):
        unit_size = 7
        calibration = [
            (0, *self._sequence(seed, unit_size)[:2], self._sequence(seed + 100, unit_size)[0], unit_size)
            for seed in range(8)
        ]
        models = fit_conditional_models(calibration, ridge=1e-5)

        rows = []
        for prompt_index, seed in enumerate(range(20, 26)):
            key, value = self._sequence(seed, unit_size)
            donor_key, _ = self._sequence(seed + 100, unit_size)
            for row in evaluate_conditional_record(
                key, value, donor_key, models[0], unit_size=unit_size
            ):
                rows.append({"prompt_id": f"p{prompt_index}", "layer": 0, **row})
        prompt_rows = aggregate_prompt_rows(rows)

        self.assertGreater(min(float(row["partial_r2_joint"]) for row in prompt_rows), 0.20)
        for control in ("wrong_space", "wrong_time", "wrong_prompt"):
            self.assertGreater(
                min(float(row[f"joint_minus_{control}"]) for row in prompt_rows),
                0.05,
            )

    def test_wrong_space_is_shifted_only_within_each_unit(self):
        unit_size = 4
        key = torch.arange(24, dtype=torch.float32).reshape(1, 1, 8, 3)
        value = key.clone()
        features, target, pairs = conditional_feature_batches(
            key, value, key + 1000, unit_size=unit_size
        )

        self.assertEqual(pairs, 1)
        self.assertEqual(target.shape, (1, 1, 4, 3))
        current_key = features["joint"][..., 3:]
        wrong_key = features["wrong_space"][..., 3:]
        self.assertTrue(torch.equal(wrong_key[:, :, 1:], current_key[:, :, :-1]))
        self.assertTrue(torch.equal(wrong_key[:, :, :1], current_key[:, :, -1:]))

    def test_rejects_single_token_units(self):
        tensor = torch.zeros(1, 1, 4, 2)
        with self.assertRaisesRegex(ValueError, "greater than one"):
            conditional_feature_batches(tensor, tensor, tensor, unit_size=1)

    def test_cli_writes_prompt_disjoint_report(self):
        trq_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = root / "calibration"
            validation = root / "validation"
            output = root / "output"
            calibration.mkdir()
            validation.mkdir()
            for split, seeds in ((calibration, (1, 2)), (validation, (11, 12))):
                for sample_index, seed in enumerate(seeds):
                    key, value = self._sequence(seed, unit_size=7)
                    torch.save(
                        {
                            "format": "hwq_kv_tensors",
                            "metadata": {
                                "sample_id": f"kv_cache_frames180_{sample_index:04d}",
                                "text_prompts": [f"{split.name} prompt {seed}"],
                                "frame_seq_length": 7,
                            },
                            "layers": {0: {"k": key, "v": value}},
                        },
                        split / f"prompt_{seed}.pt",
                    )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(trq_root / "src")
            result = subprocess.run(
                [
                    sys.executable,
                    str(trq_root / "scripts" / "analysis" / "analyze_conditional_increment.py"),
                    "--calibration-dumps",
                    str(calibration / "*.pt"),
                    "--validation-dumps",
                    str(validation / "*.pt"),
                    "--output-dir",
                    str(output),
                    "--layers",
                    "0",
                    "--unit-size",
                    "7",
                    "--bootstrap-resamples",
                    "100",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            summary = json.loads((output / "summary.json").read_text())
            self.assertIn(summary["status"], ("PASS_STRUCTURE_GATE", "FAIL_STRUCTURE_GATE"))
            self.assertEqual(len((output / "prompt_rows.csv").read_text().splitlines()), 3)

    def test_cli_rejects_same_prompt_text_with_different_sample_ids(self):
        trq_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = root / "calibration"
            validation = root / "validation"
            output = root / "output"
            calibration.mkdir()
            validation.mkdir()
            split_specs = (
                (calibration, ((1, "shared prompt"), (2, "calibration only"))),
                (validation, ((11, "shared prompt"), (12, "validation only"))),
            )
            for split, samples in split_specs:
                for sample_index, (seed, prompt) in enumerate(samples):
                    key, value = self._sequence(seed, unit_size=7)
                    torch.save(
                        {
                            "format": "hwq_kv_tensors",
                            "metadata": {
                                "sample_id": f"{split.name}_{sample_index:04d}",
                                "text_prompts": [prompt],
                                "frame_seq_length": 7,
                            },
                            "layers": {0: {"k": key, "v": value}},
                        },
                        split / f"prompt_{seed}.pt",
                    )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(trq_root / "src")
            result = subprocess.run(
                [
                    sys.executable,
                    str(trq_root / "scripts" / "analysis" / "analyze_conditional_increment.py"),
                    "--calibration-dumps",
                    str(calibration / "*.pt"),
                    "--validation-dumps",
                    str(validation / "*.pt"),
                    "--output-dir",
                    str(output),
                    "--layers",
                    "0",
                    "--unit-size",
                    "7",
                    "--bootstrap-resamples",
                    "10",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("prompt identities overlap", result.stderr)

    @staticmethod
    def _sequence(seed: int, unit_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        heads, dim, units = 2, 4, 6
        key_units = torch.randn(
            1, heads, units, unit_size, dim, generator=generator
        )
        values = [0.05 * torch.randn(1, heads, unit_size, dim, generator=generator)]
        weight = torch.tensor(
            [
                [[0.7, 0.1, 0.0, -0.1], [0.0, 0.6, 0.2, 0.0], [0.1, 0.0, 0.8, 0.1], [0.0, -0.2, 0.0, 0.7]],
                [[0.6, 0.0, 0.1, 0.0], [-0.1, 0.7, 0.0, 0.2], [0.0, 0.1, 0.6, 0.0], [0.2, 0.0, -0.1, 0.8]],
            ]
        )
        for unit in range(1, units):
            synchronous = torch.einsum(
                "bhsd,hde->bhse", key_units[:, :, unit], weight
            )
            noise = 0.01 * torch.randn(
                1, heads, unit_size, dim, generator=generator
            )
            values.append(0.35 * values[-1] + synchronous + noise)
        return key_units.reshape(1, heads, units * unit_size, dim), torch.cat(values, dim=2)
