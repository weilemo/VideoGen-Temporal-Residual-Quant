import csv
import json
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
import torch

from trq.analysis.cross_kv_smoke import analyze_cross_kv_smoke, write_cross_kv_smoke


class CrossKVSmokeTests(TestCase):
    def test_complete_paired_evidence_authorizes_only_the_mb32_followup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quality_runs = self._build_evidence(root, manual_complete=True)

            tables, summary = analyze_cross_kv_smoke(
                bf16_root=root / "bf16",
                temporal_root=root / "temporal",
                cross_root=root / "cross",
                temporal_trace_root=root / "temporal",
                cross_trace_root=root / "cross",
                runtime_roots={
                    "bf16": root / "bf16",
                    "temporal": root / "temporal",
                    "cross": root / "cross",
                },
                quality_runs=quality_runs,
                failure_tags=root / "failure_tags.csv",
                cross_params=root / "cross.npz",
            )

            self.assertEqual(summary["status"], "PASS_SMOKE_MB32_AUTHORIZED")
            self.assertTrue(summary["mb32_authorized"])
            self.assertEqual(summary["gates"]["latent_trajectory"]["status"], "PASS")
            self.assertEqual(summary["gates"]["attention_output"]["status"], "PASS")
            self.assertEqual(summary["gates"]["actual_bytes"]["status"], "PASS")
            self.assertEqual(len(tables["latent_pair_rows"]), 16)

            write_cross_kv_smoke(root / "output", tables, summary)
            self.assertTrue((root / "output" / "summary.json").is_file())
            self.assertIn("MB32 authorized: **True**", (root / "output" / "decision.md").read_text())

    def test_blank_manual_review_never_auto_authorizes_mb32(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quality_runs = self._build_evidence(root, manual_complete=False)

            _, summary = analyze_cross_kv_smoke(
                bf16_root=root / "bf16",
                temporal_root=root / "temporal",
                cross_root=root / "cross",
                temporal_trace_root=root / "temporal",
                cross_trace_root=root / "cross",
                runtime_roots={
                    "bf16": root / "bf16",
                    "temporal": root / "temporal",
                    "cross": root / "cross",
                },
                quality_runs=quality_runs,
                failure_tags=root / "failure_tags.csv",
                cross_params=root / "cross.npz",
            )

            self.assertEqual(summary["status"], "PASS_AUTOMATIC_MANUAL_REVIEW_PENDING")
            self.assertFalse(summary["mb32_authorized"])

    def _build_evidence(self, root: Path, *, manual_complete: bool):
        methods = {"bf16": 0.0, "temporal": 0.2, "cross": 0.1}
        quality_runs = []
        tag_rows = []
        for prompt in range(4):
            for seed in range(2):
                metadata = {
                    "prompt_index": prompt,
                    "seed": seed,
                    "sample_index": 0,
                    "prompt": f"prompt {prompt}",
                    "num_output_frames": 4,
                    "local_attn_size": 4,
                    "effective_seed": seed,
                }
                reference = torch.ones(4, 2, 2)
                for method, delta in methods.items():
                    run = root / method / f"p{prompt}" / f"s{seed}"
                    latent_dir = run / "rollout_metrics" / "latents"
                    latent_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"latents": reference + delta, "metadata": metadata},
                        latent_dir / "rollout.pt",
                    )
                    self._runtime(run, metadata, cache_bytes=10_000)
                for method, factor in (("temporal", 1.0), ("cross", 0.5)):
                    run = root / method / f"p{prompt}" / f"s{seed}"
                    self._trace(run, factor)
                    self._snapshot(run, prompt, seed)
                    quality = root / "quality" / method / f"p{prompt}_s{seed}.json"
                    quality.parent.mkdir(parents=True, exist_ok=True)
                    quality.write_text(
                        json.dumps(
                            {
                                "per_video": [
                                    {
                                        "idx": prompt,
                                        "sample_idx": 0,
                                        "psnr": 30.0 if method == "cross" else 28.0,
                                        "ssim": 0.9 if method == "cross" else 0.8,
                                        "lpips": 0.1 if method == "cross" else 0.2,
                                    }
                                ]
                            }
                        ),
                        encoding="utf-8",
                    )
                    quality_runs.append((method, seed, quality))
                    row = {
                        "config": method,
                        "prompt_index": prompt,
                        "seed": seed,
                        "sample_index": 0,
                        "notes": "",
                    }
                    for field in (
                        "identity_switch",
                        "background_jump",
                        "texture_repetition",
                        "motion_freeze",
                        "color_drift",
                        "black_or_nan",
                        "catastrophe",
                        "bf16_present",
                    ):
                        row[field] = "0" if manual_complete else ""
                    tag_rows.append(row)

        with (root / "failure_tags.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(tag_rows[0]))
            writer.writeheader()
            writer.writerows(tag_rows)
        np.savez(
            root / "cross.npz",
            cross_weight=np.ones((2, 2), dtype=np.float32),
            cross_bias=np.ones((2,), dtype=np.float32),
        )
        return quality_runs

    @staticmethod
    def _trace(run: Path, factor: float) -> None:
        trace = run / "attention_trace" / "boundary0024_layer08.json"
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_text(
            json.dumps(
                {
                    "boundary_frame": 24,
                    "layer_idx": 8,
                    "metrics": {
                        "v_cache_read_rel_l2": 0.2 * factor,
                        "attention_output_rel_l2": 0.2 * factor,
                        "attention_logits_rel_l2": 0.2 * factor,
                        "softmax_kl": 0.2 * factor,
                        "attention_top1_agreement": 0.8 + 0.1 * (1.0 - factor),
                        "attention_topk_overlap": 0.8 + 0.1 * (1.0 - factor),
                    },
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _snapshot(run: Path, prompt: int, seed: int) -> None:
        path = run / "parity_snapshots" / "first_event_rank0_layer08.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        key = torch.full((1, 1, 2, 2), float(prompt + seed), dtype=torch.bfloat16)
        torch.save(
            {
                "metadata": {"layer_idx": 8},
                "layers": {8: {"k": key, "trq_decoded_k": key.clone()}},
            },
            path,
        )

    @staticmethod
    def _runtime(run: Path, metadata: dict, *, cache_bytes: int) -> None:
        path = run / "rollout_metrics" / "runtime" / "runtime.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "metadata": metadata,
                    "wall_time_e2e_ms": 100.0,
                    "pipeline": {
                        "stages_ms": {"diffusion": 90.0},
                        "kv_cache": {"total": {"physical_bytes": cache_bytes}},
                        "memory": {
                            "max_allocated_bytes": 20_000,
                            "max_reserved_bytes": 25_000,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
