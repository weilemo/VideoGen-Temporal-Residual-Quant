import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "eval" / "eval_ref_metrics.py"
SPEC = importlib.util.spec_from_file_location("eval_ref_metrics", SCRIPT)
eval_ref_metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(eval_ref_metrics)


class EvalRefMetricsTest(unittest.TestCase):
    def test_video_psnr_is_finite_when_only_the_prefix_is_identical(self):
        reference = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        candidate = reference.copy()
        candidate[1] = 255

        with self.subTest("direct video-level aggregation"):
            self.assertTrue(math.isfinite(eval_ref_metrics.psnr(reference, candidate)))

    def test_directory_evaluation_uses_video_level_psnr(self):
        reference = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        candidate = reference.copy()
        candidate[1] = 255

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_dir = root / "ref"
            cmp_dir = root / "cmp"
            ref_dir.mkdir()
            cmp_dir.mkdir()
            (ref_dir / "0-0_ema.mp4").touch()
            (cmp_dir / "0-0_ema.mp4").touch()

            with (
                patch.object(eval_ref_metrics, "lpips", None),
                patch.object(
                    eval_ref_metrics,
                    "read_video",
                    side_effect=(reference, candidate),
                ),
            ):
                summary = eval_ref_metrics.evaluate_directories(
                    ref_dir,
                    cmp_dir,
                    start_frame=1,
                    match_by_index=True,
                    strict_shape=True,
                )

        self.assertTrue(math.isfinite(summary["mean_psnr"]))
        self.assertEqual(summary["start_frame"], 1)
        self.assertEqual(
            summary["psnr_aggregation"],
            "RGB global MSE per video, then mean PSNR across videos",
        )


if __name__ == "__main__":
    unittest.main()
