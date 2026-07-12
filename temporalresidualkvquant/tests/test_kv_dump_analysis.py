from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from trq.analysis.kv_dump import (
    iter_kv_dump_layers,
    iter_kv_dumps,
    parse_layer_spec,
    resolve_dump_paths,
)
from trq.kv_cache import ChunkedKVCache


class KVDumpAnalysisTests(TestCase):
    def test_layer_spec_supports_all_indices_and_inclusive_ranges(self):
        self.assertEqual(parse_layer_spec("all", 5), [0, 1, 2, 3, 4])
        self.assertEqual(parse_layer_spec("0,2-4,2", 6), [0, 2, 3, 4])
        self.assertEqual(parse_layer_spec("3", [1, 3, 8]), [3])

        with self.assertRaisesRegex(ValueError, "ascending"):
            parse_layer_spec("4-2", 5)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            parse_layer_spec("5", 5)

    def test_lightweight_dump_yields_selected_bhsd_float32_layers(self):
        layer_zero = torch.arange(24, dtype=torch.bfloat16).reshape(1, 2, 3, 4)
        layer_two = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4)
        payload = {
            "format": "hwq_kv_tensors",
            "layers": {
                0: {"k": layer_zero, "v": layer_zero + 1},
                "2": {"k": layer_two, "v": layer_two + 2},
            },
            "metadata": {"prompt_id": "fixture-0"},
        }

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixture.pt"
            torch.save(payload, path)
            records = list(iter_kv_dump_layers(path, layers="2,0"))

        self.assertEqual([record[0] for record in records], [2, 0])
        self.assertEqual(records[0][1].dtype, torch.float32)
        self.assertEqual(records[0][1].shape, (1, 1, 4, 4))
        self.assertTrue(records[0][1].is_contiguous())
        self.assertEqual(records[0][3]["layout"], "BHSD")
        self.assertEqual(records[0][3]["dump_metadata"]["prompt_id"], "fixture-0")

    def test_layer_shards_accept_one_global_layer_filter(self):
        payload = {
            "format": "hwq_kv_tensors",
            "layers": {5: {"k": torch.ones(1, 1, 2, 4), "v": torch.ones(1, 1, 2, 4)}},
        }
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample_layer05.pt"
            torch.save(payload, path)
            selected = list(iter_kv_dump_layers(path, layers="0,5,11"))
            skipped = list(iter_kv_dump_layers(path, layers="0,11"))

        self.assertEqual([record[0] for record in selected], [5])
        self.assertEqual(skipped, [])

    def test_comma_and_glob_paths_are_expanded_and_deduplicated(self):
        payload = {
            "format": "hwq_kv_tensors",
            "layers": {0: {"k": torch.ones(1, 1, 1, 1), "v": torch.ones(1, 1, 1, 1)}},
        }
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "a.pt"
            second = root / "b.pt"
            torch.save(payload, first)
            torch.save(payload, second)

            resolved = resolve_dump_paths(f"{root}/*.pt,{first}")
            records = list(iter_kv_dumps(f"{first},{second}", layers="all"))

        self.assertEqual(resolved, [first.resolve(), second.resolve()])
        self.assertEqual(len(records), 2)

    def test_chunked_dump_requires_contiguous_raw_bf16_and_returns_bhsd(self):
        cache_k = self._raw_cache(layout="BSHD")
        cache_v = self._raw_cache(layout="BSHD")
        k_source = torch.arange(24, dtype=torch.bfloat16).reshape(1, 3, 2, 4)
        v_source = k_source + 1
        cache_k.write(0, 3, k_source)
        cache_v.write(0, 3, v_source)
        payload = {
            "kv_cache": [{
                "k": cache_k,
                "v": cache_v,
                "local_end_index": torch.tensor([3]),
                "global_end_index": torch.tensor([3]),
            }]
        }

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chunked.pt"
            torch.save(payload, path)
            [(layer_idx, k, v, metadata)] = list(iter_kv_dump_layers(path))

        self.assertEqual(layer_idx, 0)
        self.assertEqual(k.shape, (1, 2, 3, 4))
        self.assertEqual(k.dtype, torch.float32)
        self.assertTrue(torch.equal(k, k_source.permute(0, 2, 1, 3).float()))
        self.assertTrue(torch.equal(v, v_source.permute(0, 2, 1, 3).float()))
        self.assertEqual(metadata["filled_chunks"], 3)
        self.assertEqual(metadata["source_layout"], "BSHD")

    def test_chunked_dump_rejects_gaps_and_quantized_spans(self):
        cache_k = self._raw_cache(layout="BHSD")
        cache_v = self._raw_cache(layout="BHSD")
        chunk = torch.ones(1, 2, 1, 4, dtype=torch.bfloat16)
        cache_k.write(1, 2, chunk)
        cache_v.write(1, 2, chunk)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gap.pt"
            torch.save({"kv_cache": [{"k": cache_k, "v": cache_v}]}, path)
            with self.assertRaisesRegex(ValueError, "contiguous prefix"):
                list(iter_kv_dump_layers(path))

            cache_k.quantized_spans.append({"start_chunk": 0, "end_chunk": 1, "quant_data": {}})
            quantized_path = Path(tmpdir) / "quantized.pt"
            torch.save({"kv_cache": [{"k": cache_k, "v": cache_v}]}, quantized_path)
            with self.assertRaisesRegex(ValueError, "quantized_spans"):
                list(iter_kv_dump_layers(quantized_path))

    @staticmethod
    def _raw_cache(*, layout: str) -> ChunkedKVCache:
        return ChunkedKVCache(
            batch_size=1,
            frame_seq_length=1,
            num_heads=2,
            head_dim=4,
            max_num_chunks=3,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            layout=layout,
        )
