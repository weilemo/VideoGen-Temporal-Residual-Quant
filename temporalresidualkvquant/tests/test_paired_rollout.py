from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from trq.analysis.paired_rollout import analyze_paired_rollouts, write_analysis


class PairedRolloutTests(TestCase):
    def test_constant_trq_offset_has_no_progressive_drift(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            directories = self._directories(root)
            for prompt_index in range(2):
                reference = torch.ones(10, 2, 2) * (prompt_index + 1)
                self._save(directories["a"], prompt_index, reference)
                self._save(directories["b"], prompt_index, reference)
                self._save(directories["q"], prompt_index, reference + 0.01)

            positions, pairs, summary = analyze_paired_rollouts(
                directories["a"], directories["b"], directories["q"],
                bootstrap_resamples=100,
            )
            write_analysis(root / "analysis", positions, pairs, summary)

            self.assertEqual(len(positions), 20)
            self.assertEqual(len(pairs), 2)
            self.assertTrue(summary["passes_latent_drift_gate"])
            self.assertTrue((root / "analysis" / "summary.json").exists())

    def test_increasing_error_fails_drift_gate(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            directories = self._directories(root)
            for prompt_index in range(2):
                reference = torch.ones(10, 2, 2)
                drift = torch.linspace(0.0, 0.2, 10).reshape(10, 1, 1)
                self._save(directories["a"], prompt_index, reference)
                self._save(directories["b"], prompt_index, reference)
                self._save(directories["q"], prompt_index, reference + drift)

            _, _, summary = analyze_paired_rollouts(
                directories["a"], directories["b"], directories["q"],
                bootstrap_resamples=100,
            )

            self.assertFalse(summary["passes_latent_drift_gate"])

    def test_absolute_metrics_and_metadata_aligned_boundary_jump(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            directories = self._directories(root)
            reference = torch.ones(12, 2, 2)
            candidate = reference.clone()
            candidate[6:].add_(0.2)
            self._save(directories["a"], 0, reference)
            self._save(directories["b"], 0, reference)
            self._save(
                directories["q"],
                0,
                candidate,
                quantization_events=[{"boundary_frame": 6}],
            )

            positions, pairs, summary = analyze_paired_rollouts(
                directories["a"],
                directories["b"],
                directories["q"],
                bootstrap_resamples=20,
                boundary_before=2,
                boundary_after=2,
            )
            write_analysis(root / "analysis", positions, pairs, summary)

            self.assertEqual(pairs[0]["first_quant_frame"], 6)
            self.assertGreater(pairs[0]["boundary_jump"], 0.0)
            self.assertIn("absolute_metrics", summary)
            self.assertTrue((root / "analysis" / "online_boundary_jump.csv").exists())
            self.assertTrue((root / "analysis" / "online_latent_absolute_curve.png").exists())
            self.assertTrue((root / "analysis" / "online_prompt_time_heatmap.png").exists())

    def test_key_mismatch_fails_closed(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            directories = self._directories(root)
            tensor = torch.ones(4, 2)
            self._save(directories["a"], 0, tensor)
            self._save(directories["b"], 0, tensor)
            self._save(directories["q"], 1, tensor)

            with self.assertRaisesRegex(ValueError, "keys do not match"):
                analyze_paired_rollouts(directories["a"], directories["b"], directories["q"])

    def test_runtime_metadata_mismatch_fails_closed(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            directories = self._directories(root)
            tensor = torch.ones(4, 2)
            self._save(directories["a"], 0, tensor, num_output_frames=180)
            self._save(directories["b"], 0, tensor, num_output_frames=180)
            self._save(directories["q"], 0, tensor, num_output_frames=183)

            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                analyze_paired_rollouts(directories["a"], directories["b"], directories["q"])

    @staticmethod
    def _directories(root: Path) -> dict[str, Path]:
        result = {name: root / name for name in ("a", "b", "q")}
        for path in result.values():
            path.mkdir()
        return result

    @staticmethod
    def _save(
        directory: Path,
        prompt_index: int,
        latents: torch.Tensor,
        *,
        num_output_frames: int | None = None,
        quantization_events: list[dict] | None = None,
    ) -> None:
        metadata = {
            "prompt_index": prompt_index,
            "seed": 0,
            "sample_index": 0,
        }
        if num_output_frames is not None:
            metadata["num_output_frames"] = num_output_frames
        if quantization_events is not None:
            metadata["quantization_events"] = quantization_events
        torch.save({
            "metadata": metadata,
            "latents": latents,
        }, directory / f"{prompt_index}.pt")
