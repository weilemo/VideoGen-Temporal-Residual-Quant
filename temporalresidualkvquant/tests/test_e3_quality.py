import tempfile
from pathlib import Path
from unittest import TestCase

import torch

from trq.analysis.e3_quality import analyze_e3_direct, write_e3_direct


class E3QualityTests(TestCase):
    def test_direct_analysis_matches_rollouts_and_preserves_evidence_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = {
                "prompt_index": 0,
                "seed": 0,
                "sample_index": 0,
                "prompt": "test prompt",
                "num_output_frames": 4,
                "local_attn_size": 4,
                "effective_seed": 0,
            }
            reference = torch.ones(4, 2, 2)
            self._save(root / "bf16", reference, metadata)
            self._save(root / "cross", reference + 0.2, metadata)
            self._save(root / "hybrid", reference + 0.1, metadata)

            positions, pairs, summary = analyze_e3_direct(
                root / "bf16", root / "cross", root / "hybrid"
            )
            self.assertEqual(len(positions), 8)
            self.assertEqual(len(pairs), 2)
            self.assertEqual(summary["keys"], 1)
            self.assertEqual(summary["status"], "MEASURED_REQUIRES_MANUAL_REVIEW")
            self.assertLess(
                summary["hybrid_over_cross_mean_relative_l2"]["pair_median"], 1.0
            )

            write_e3_direct(root / "output", positions, pairs, summary)
            self.assertTrue((root / "output" / "summary.json").is_file())
            self.assertTrue((root / "output" / "position_rows.csv").is_file())

    @staticmethod
    def _save(directory: Path, latents: torch.Tensor, metadata: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"latents": latents, "metadata": metadata}, directory / "rollout.pt")
