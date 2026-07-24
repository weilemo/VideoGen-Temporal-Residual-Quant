import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from trq.analysis.paired_quality import (
    DIMENSIONS,
    VBenchRun,
    analyze_paired_quality,
    write_quality_analysis,
)


class PairedQualityTests(TestCase):
    def test_missing_qualitative_tags_keeps_gate_incomplete(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._write_runs(root, candidate_delta=0.0)

            rows, correlations, summary = analyze_paired_quality(runs)
            write_quality_analysis(root / "analysis", rows, correlations, summary)

            self.assertEqual(summary["configs"]["k4v4"]["status"], "INCOMPLETE")
            self.assertTrue((root / "analysis" / "paired_vbench.csv").exists())

    def test_complete_tags_pass_without_numeric_threshold(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._write_runs(root, candidate_delta=-0.2)
            tags = root / "tags.csv"
            self._write_tags(tags, catastrophe=False)

            _, _, summary = analyze_paired_quality(runs, failure_tags_path=tags)

            self.assertEqual(summary["configs"]["k4v4"]["status"], "PASS")
            self.assertFalse(summary["quality_gate_policy"]["vbench_numeric_gate_enabled"])
            self.assertIsNone(summary["configs"]["k4v4"]["passes_final_delta"])

    def test_two_trq_only_catastrophes_fail(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._write_runs(root, candidate_delta=0.0)
            tags = root / "tags.csv"
            self._write_tags(tags, catastrophe=True)

            _, _, summary = analyze_paired_quality(runs, failure_tags_path=tags)

            self.assertEqual(summary["configs"]["k4v4"]["status"], "FAIL")

    def _write_runs(self, root: Path, candidate_delta: float):
        runs = []
        for config, delta in (("bf16_a", 0.0), ("bf16_b", 0.0), ("k4v4", candidate_delta)):
            directory = root / config
            directory.mkdir()
            label = f"{config}_s0"
            for dimension in DIMENSIONS:
                details = []
                for prompt in range(2):
                    details.append({
                        "video_path": f"/tmp/split_clip/{prompt}-0_ema/{prompt}-0_ema_000.mp4",
                        "video_results": 0.9 + delta,
                    })
                (directory / f"{label}_{dimension}_eval_results.json").write_text(
                    json.dumps({dimension: [0.9 + delta, details]}),
                    encoding="utf-8",
                )
            runs.append(VBenchRun(config, 0, label, directory))
        return runs

    @staticmethod
    def _write_tags(path: Path, catastrophe: bool):
        rows = [
            {
                "config": "k4v4",
                "prompt_index": prompt,
                "seed": 0,
                "sample_index": 0,
                "catastrophe": str(catastrophe).lower(),
                "bf16_present": "false",
                "identity_switch": "false",
                "background_jump": "false",
                "texture_repetition": "false",
                "motion_freeze": "false",
                "color_drift": "false",
                "black_or_nan": "false",
            }
            for prompt in range(2)
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
