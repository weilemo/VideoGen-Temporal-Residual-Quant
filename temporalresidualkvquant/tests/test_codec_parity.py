from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from trq.analysis.codec_parity import (
    compare_identity_codecs,
    state_metadata,
    tensor_error_metrics,
)
from trq.analysis.kv_dump import iter_kv_dump_layers
from trq.analysis.online_snapshot import parse_capture_layers, save_online_snapshot
from trq.real.trq import trq_dequantize_tensor, trq_quantize_tensor


class _StudentIdentityStub:
    """Expose the S2++ API while reusing TRQ to test comparison plumbing."""

    @staticmethod
    def s2pp_quantize_tensor(tensor, **kwargs):
        state = trq_quantize_tensor(
            tensor,
            num_bits=kwargs["num_bits"],
            block_size=kwargs["block_size"],
            anchor_bits=kwargs["anchor_bits"],
            predictor_stride=kwargs["predictor_stride"],
            predictor_mode="identity",
            layer_idx=kwargs.get("layer_idx"),
            scale_precision=kwargs["scale_precision"],
            residual_quant_mode=kwargs["residual_quant_mode"],
        )
        legacy = dict(state)
        for key in ("format", "version", "layout", "predictor", "dependency"):
            legacy.pop(key, None)
        legacy.update(
            {
                "method": "s2pp",
                "mode": "identity",
                "predictor_mode": "identity",
                "predictor_kind": "identity",
            }
        )
        return legacy

    @staticmethod
    def s2pp_dequantize_tensor(state, output_dtype):
        return trq_dequantize_tensor(state, output_dtype=output_dtype)


class CodecParityTests(TestCase):
    def test_tensor_error_metrics_are_zero_for_identical_tensors(self):
        tensor = torch.randn(1, 2, 4, 8, generator=torch.Generator().manual_seed(2))
        metrics = tensor_error_metrics(tensor, tensor.clone())
        self.assertEqual(metrics["rel_l2"], 0.0)
        self.assertEqual(metrics["max_abs"], 0.0)
        self.assertAlmostEqual(metrics["cosine"], 1.0, places=6)

    def test_comparison_reports_exact_stub_parity(self):
        tensor = torch.randn(1, 2, 18, 8, generator=torch.Generator().manual_seed(3))
        summary, unit_rows = compare_identity_codecs(
            tensor,
            student_module=_StudentIdentityStub,
            num_bits=2,
            anchor_bits=4,
            block_size=4,
            predictor_stride=6,
            scale_precision=torch.float32,
            codec_dtype=torch.float32,
        )
        self.assertTrue(summary["state_unit_lengths_match"])
        self.assertEqual(summary["trq_vs_student_torch"]["max_abs"], 0.0)
        self.assertEqual(summary["trq_encoder_decoder_parity"]["max_abs"], 0.0)
        self.assertEqual(len(unit_rows), 3)

    def test_online_snapshot_is_readable_as_raw_dump(self):
        raw_k = torch.randn(1, 2, 8, 8, generator=torch.Generator().manual_seed(4))
        raw_v = torch.randn(1, 2, 8, 8, generator=torch.Generator().manual_seed(5))
        k_state, decoded_k = trq_quantize_tensor(
            raw_k,
            num_bits=2,
            anchor_bits=4,
            block_size=4,
            predictor_stride=4,
            scale_precision=torch.float32,
            return_reconstruction=True,
        )
        v_state, decoded_v = trq_quantize_tensor(
            raw_v,
            num_bits=2,
            anchor_bits=4,
            block_size=4,
            predictor_stride=4,
            scale_precision=torch.float32,
            return_reconstruction=True,
        )
        with TemporaryDirectory() as directory:
            path = save_online_snapshot(
                directory,
                layer_idx=2,
                raw_k=raw_k,
                raw_v=raw_v,
                decoded_k=decoded_k,
                decoded_v=decoded_v,
                encoded_k=k_state,
                encoded_v=v_state,
                tokens_start=0,
                tokens_end=8,
                max_tokens=8,
                quant_config={"quant_type": "trq-int2"},
            )
            records = list(iter_kv_dump_layers(path, layers="2"))
            payload = torch.load(Path(path), map_location="cpu", weights_only=False)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][0], 2)
        self.assertEqual(tuple(records[0][1].shape), tuple(raw_k.shape))
        self.assertIn("trq_decoded_k", payload["layers"][2])
        self.assertEqual(payload["metadata"]["tokens_end"], 8)
        self.assertEqual(state_metadata(k_state)["predictor_kind"], "identity")

    def test_capture_layer_parser(self):
        self.assertEqual(parse_capture_layers("0,2-3", 5), {0, 2, 3})
        self.assertEqual(parse_capture_layers("all", 3), {0, 1, 2})
        with self.assertRaisesRegex(ValueError, "outside"):
            parse_capture_layers("4", 4)
