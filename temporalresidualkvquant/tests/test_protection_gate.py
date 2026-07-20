import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from trq.analysis.protection_gate import analyze_protection_candidates


class ProtectionGateTests(TestCase):
    def test_candidate_passes_quality_jump_and_bytes_gate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.csv"
            candidate = root / "candidate.csv"
            self._write_csv(baseline, 0.2)
            self._write_csv(candidate, 0.08)
            quality = root / "quality.json"
            quality.write_text(json.dumps({"configs": {"p1": {"status": "PASS"}}}))
            runtime = root / "runtime" / "rollout_metrics" / "runtime"
            runtime.mkdir(parents=True)
            (runtime / "0.json").write_text(json.dumps({
                "pipeline": {"kv_cache": {"total": {"saving_fraction": 0.72}}}
            }))

            rows, summary = analyze_protection_candidates(
                quality_gate_path=quality,
                baseline_latent_csv=baseline,
                candidate_latent_csvs={"p1": candidate},
                runtime_roots={"p1": root},
            )

            self.assertEqual(rows[0]["status"], "PASS")
            self.assertEqual(summary["passing_configs"], ["p1"])

    @staticmethod
    def _write_csv(path: Path, jump: float):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["boundary_jump"])
            writer.writeheader()
            writer.writerow({"boundary_jump": jump})
