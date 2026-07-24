import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_hy_official_scenes", ROOT / "prepare_hy_official_scenes.py"
)
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare)

CAUSAL_SPEC = importlib.util.spec_from_file_location(
    "prepare_causal_prompts", ROOT / "prepare_causal_prompts.py"
)
causal_prepare = importlib.util.module_from_spec(CAUSAL_SPEC)
sys.modules[CAUSAL_SPEC.name] = causal_prepare
assert CAUSAL_SPEC.loader is not None
CAUSAL_SPEC.loader.exec_module(causal_prepare)


def test_official_hy_cases_split_into_fixed_dev_and_holdout(tmp_path):
    csv_path = tmp_path / "test_case.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "caption"])
        writer.writeheader()
        for index in range(1, 11):
            writer.writerow(
                {
                    "image_name": f"./assets/img/{index}.png",
                    "caption": f"scene {index}",
                }
            )

    cases = prepare.load_cases(csv_path)
    dev = prepare.build_manifest(cases[:5], tmp_path / "img")
    holdout = prepare.build_manifest(cases[5:], tmp_path / "img")

    assert [row["id"] for row in dev["scenes"]] == [
        f"official_{index:02d}" for index in range(1, 6)
    ]
    assert [row["id"] for row in holdout["scenes"]] == [
        f"official_{index:02d}" for index in range(6, 11)
    ]


def test_materialize_can_use_official_local_checkout(tmp_path):
    source = tmp_path / "official" / "img" / "case.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"official-image")
    destination = tmp_path / "dataset" / "img" / "case.png"

    prepare.materialize(source, "https://invalid.example/case.png", destination)

    assert destination.read_bytes() == b"official-image"


def test_two_gpu_dry_run_uses_gpu_2_and_4_without_execution(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "run_two_gpu_matrix.sh")],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "RUN_ID": "test-run",
            "HOME": str(tmp_path),
        },
    )

    assert "GPU 2" in result.stdout
    assert "GPU 4" in result.stdout
    assert "prompts 0-4" in result.stdout
    assert "prompts 5-9" in result.stdout
    assert "continue after a sibling failure" in result.stdout
    assert "No GPU commands executed" in result.stdout


def test_prompt_slice_selects_disjoint_longcat_shards(tmp_path):
    command = f"""
      source {ROOT / 'common.sh'}
      a=$(prepare_prompt_slice 5 0 test_a)
      b=$(prepare_prompt_slice 5 5 test_b)
      printf '%s|%s\n' "$(head -n 1 "$a")" "$(head -n 1 "$b")"
    """
    result = subprocess.run(
        ["bash", "-c", command], check=True, capture_output=True, text=True
    )
    left, right = result.stdout.strip().split("|", maxsplit=1)
    assert left != right


def test_causal_resume_selects_only_missing_or_invalid_outputs(tmp_path, monkeypatch):
    prompts = ["scene zero", "scene one", "scene two"]
    output = tmp_path / "videos"
    output.mkdir()
    (output / causal_prepare.output_name(prompts[0])).write_bytes(b"complete")
    (output / causal_prepare.output_name(prompts[1])).write_bytes(b"broken")
    monkeypatch.setattr(
        causal_prepare,
        "is_decodable",
        lambda path: path.name == causal_prepare.output_name(prompts[0]),
    )

    missing, reused = causal_prepare.select_missing(prompts, 0, 3, output)

    assert reused == 1
    assert missing == prompts[1:]
