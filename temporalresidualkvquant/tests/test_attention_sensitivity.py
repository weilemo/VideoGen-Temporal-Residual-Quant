from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from trq.analysis.attention_sensitivity import (
    sampled_attention_metrics,
    write_attention_trace,
)


class AttentionSensitivityTests(TestCase):
    def test_identical_inputs_have_zero_drift_and_full_agreement(self):
        generator = torch.Generator().manual_seed(7)
        q = torch.randn(1, 6, 2, 4, generator=generator)
        k = torch.randn(1, 10, 2, 4, generator=generator)
        v = torch.randn(1, 10, 2, 4, generator=generator)

        metrics = sampled_attention_metrics(q, k, v, k.clone(), v.clone(), topk=3)

        self.assertEqual(metrics["k_cache_read_rel_l2"], 0.0)
        self.assertEqual(metrics["attention_output_rel_l2"], 0.0)
        self.assertAlmostEqual(metrics["attention_top1_agreement"], 1.0)
        self.assertAlmostEqual(metrics["attention_topk_overlap"], 1.0)

    def test_k_perturbation_changes_routing_metrics(self):
        generator = torch.Generator().manual_seed(9)
        q = torch.randn(1, 5, 2, 4, generator=generator)
        k = torch.randn(1, 8, 2, 4, generator=generator)
        v = torch.randn(1, 8, 2, 4, generator=generator)
        candidate_k = k.clone()
        candidate_k[:, 0].add_(10.0)

        metrics = sampled_attention_metrics(q, k, v, candidate_k, v, topk=2)

        self.assertGreater(metrics["attention_logits_rel_l2"], 0.0)
        self.assertGreater(metrics["softmax_kl"], 0.0)

    def test_trace_writer_uses_boundary_and_layer_identity(self):
        with TemporaryDirectory() as directory:
            path = write_attention_trace(
                directory,
                layer_idx=8,
                boundary_frame=24,
                metrics={"softmax_kl": 0.1},
            )

            self.assertEqual(path.name, "boundary0024_layer08.json")
            self.assertTrue(path.exists())
