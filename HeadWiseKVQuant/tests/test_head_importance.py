import json
import tempfile
from pathlib import Path
from unittest import TestCase

from trq.head_importance import (
    build_topk_policy_from_focused_forcing,
    load_focused_forcing_head_losses,
    select_top_heads_by_layer,
)


class HeadImportanceTests(TestCase):
    def test_load_focused_forcing_head_losses_aggregates_repeated_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "0.json").write_text(json.dumps({"0": 0.1, "1": 0.2}), encoding="utf-8")
            (root / "1.json").write_text(json.dumps({"0": 0.3, "2": 0.5}), encoding="utf-8")

            losses = load_focused_forcing_head_losses(root)

        self.assertEqual(losses[0], [0.1, 0.3])
        self.assertEqual(losses[1], [0.2])
        self.assertEqual(losses[2], [0.5])

    def test_select_top_heads_by_layer_uses_higher_scores_by_default(self):
        selected = select_top_heads_by_layer(
            {0: [0.1, 0.9, 0.2, 0.3], 1: [0.4, 0.5, 0.8, 0.7]},
            top_k=1,
        )

        self.assertEqual(selected, {0: [1], 1: [2]})

    def test_build_topk_policy_from_focused_forcing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scores = [0.1, 0.9, 0.2, 0.3, 0.4, 0.5, 0.8, 0.7]
            (root / "chunk.json").write_text(
                json.dumps({str(i): value for i, value in enumerate(scores)}),
                encoding="utf-8",
            )

            policy = build_topk_policy_from_focused_forcing(
                root,
                num_layers=2,
                num_heads=4,
                top_k=1,
            )

        self.assertEqual(policy["format"], "headwise-topk-policy-v1")
        self.assertEqual(policy["top_heads_by_layer"], {"0": [1], "1": [2]})
        self.assertEqual(policy["global_scores"]["1"], 0.9)
