import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_b1_review", "prepare_b1_review.py")
decode = load_module("decode_b1_review", "decode_b1_review.py")


def movie_manifest(tmp_path: Path, baseline: str, prompts: int = 2) -> Path:
    records = []
    for prompt_index in range(prompts):
        for variant in ("bf16", *prepare.CANDIDATES):
            source = tmp_path / "sources" / baseline / variant / f"{prompt_index}-0.mp4"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"{baseline}-{variant}-{prompt_index}".encode())
            records.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": f"prompt {prompt_index}",
                    "variant": variant,
                    "source_path": str(source),
                }
            )
    path = tmp_path / f"{baseline}.json"
    path.write_text(json.dumps({"records": records}), encoding="utf-8")
    return path


def hy_root(tmp_path: Path) -> Path:
    root = tmp_path / "hy"
    for action in prepare.ACTION_ORDER:
        for variant in ("bf16", *prepare.CANDIDATES):
            source = root / "scene_06" / action / variant / "0.mp4"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"{action}-{variant}".encode())
    return root


def build_package(tmp_path: Path) -> tuple[Path, dict, dict]:
    output = tmp_path / "review"
    args = argparse.Namespace(
        causal_manifest=movie_manifest(tmp_path, "causal"),
        longcat_manifest=movie_manifest(tmp_path, "longcat"),
        hy_root=hy_root(tmp_path),
        output_root=output,
        expected_prompts=2,
        expected_hy_scenes=1,
        seed=29,
    )
    prepare.write_package(args)
    public = json.loads((output / "public" / "review_manifest.json").read_text())
    private = json.loads((output / "private_mapping.json").read_text())
    return output, public, private


def test_review_package_is_complete_and_public_manifest_is_anonymous(tmp_path):
    output, public, private = build_package(tmp_path)

    assert public["task_count"] == 2 * 2 * 4 + 1 * 4
    assert sum(task["kind"] == "catastrophe" for task in public["tasks"]) == 16
    assert sum(task["kind"] == "action" for task in public["tasks"]) == 4
    public_text = json.dumps(public).lower()
    for secret in ("bf16", "trq_int4", "trq_int2", "naive_int4", "naive_int2", "source_path"):
        assert secret not in public_text
    assert any(
        task.get("conditioning_frames") == 13
        for task in public["tasks"]
        if task["baseline"] == "LongCat Video"
    )
    assert set(private["tasks"]) == {task["id"] for task in public["tasks"]}
    assert (output / "public" / "index.html").is_file()
    assert all(path.is_symlink() for path in (output / "public" / "media").rglob("*.mp4"))


def test_review_package_refuses_to_overwrite_existing_package(tmp_path):
    output, _, _ = build_package(tmp_path)
    args = argparse.Namespace(
        causal_manifest=tmp_path / "causal.json",
        longcat_manifest=tmp_path / "longcat.json",
        hy_root=tmp_path / "hy",
        output_root=output,
        expected_prompts=2,
        expected_hy_scenes=1,
        seed=29,
    )
    with pytest.raises(FileExistsError):
        prepare.write_package(args)


def test_decode_joins_variant_only_after_review(tmp_path):
    _, public, private = build_package(tmp_path)
    task = next(task for task in public["tasks"] if task["kind"] == "catastrophe")
    review = {
        "manifest_sha256": public["manifest_sha256"],
        "answers": {
            task["id"]: {
                "complete": True,
                "sides": {
                    "A": {"severity": "0", "catastrophe": "no", "onset": "none"},
                    "B": {"severity": "2", "catastrophe": "yes", "onset": "middle"},
                },
            }
        },
    }

    rows = decode.flatten_export(review, private)

    assert len(rows) == 2
    assert {row["variant"] for row in rows} == {"bf16", private["tasks"][task["id"]]["sides"]["A" if private["tasks"][task["id"]]["sides"]["A"]["variant"] != "bf16" else "B"]["variant"]}
    assert all(row["complete"] for row in rows)


def test_decode_rejects_a_different_manifest(tmp_path):
    _, _, private = build_package(tmp_path)
    with pytest.raises(ValueError, match="different manifests"):
        decode.flatten_export({"manifest_sha256": "wrong", "answers": {}}, private)
