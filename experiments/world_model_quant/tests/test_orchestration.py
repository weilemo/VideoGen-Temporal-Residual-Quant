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

MANIFEST_SPEC = importlib.util.spec_from_file_location(
    "prepare_moviegen_manifest", ROOT / "prepare_moviegen_manifest.py"
)
moviegen_manifest = importlib.util.module_from_spec(MANIFEST_SPEC)
sys.modules[MANIFEST_SPEC.name] = moviegen_manifest
assert MANIFEST_SPEC.loader is not None
MANIFEST_SPEC.loader.exec_module(moviegen_manifest)

PROMPT_SPEC = importlib.util.spec_from_file_location(
    "validate_moviegen_prompts", ROOT / "validate_moviegen_prompts.py"
)
prompt_validator = importlib.util.module_from_spec(PROMPT_SPEC)
sys.modules[PROMPT_SPEC.name] = prompt_validator
assert PROMPT_SPEC.loader is not None
PROMPT_SPEC.loader.exec_module(prompt_validator)

PAIRED_SPEC = importlib.util.spec_from_file_location(
    "run_forcing_paired_metrics",
    ROOT.parent / "paired_quality" / "run_forcing_paired_metrics.py",
)
paired_metrics = importlib.util.module_from_spec(PAIRED_SPEC)
sys.modules[PAIRED_SPEC.name] = paired_metrics
assert PAIRED_SPEC.loader is not None
PAIRED_SPEC.loader.exec_module(paired_metrics)


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


def test_runtime_prefix_binds_python_and_torchrun(tmp_path):
    prefix = tmp_path / "videoquant"
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("python", "torchrun"):
        executable = bin_dir / name
        executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)

    command = f"""
      source {ROOT / 'common.sh'}
      VIDEOQUANT_ENV_PREFIX={prefix}
      configure_videoquant_runtime
      printf '%s|%s|%s\n' "$CONDA_PREFIX" "$PYTHON_BIN" "$(command -v torchrun)"
    """
    result = subprocess.run(
        ["bash", "-c", command], check=True, capture_output=True, text=True
    )

    conda_prefix, python_bin, torchrun_bin = result.stdout.strip().split("|")
    assert conda_prefix == str(prefix)
    assert python_bin == str(bin_dir / "python")
    assert torchrun_bin == str(bin_dir / "torchrun")


def test_explicit_missing_runtime_prefix_fails(tmp_path):
    missing = tmp_path / "missing-videoquant"
    command = f"""
      source {ROOT / 'common.sh'}
      VIDEOQUANT_ENV_PREFIX={missing}
      configure_videoquant_runtime
    """

    result = subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True
    )

    assert result.returncode == 2
    assert "VIDEOQUANT_ENV_PREFIX does not exist" in result.stderr


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


def test_prompt_source_override_supports_moviegen32(tmp_path):
    prompts = tmp_path / "moviegen32.txt"
    prompts.write_text("".join(f"prompt {index}\n" for index in range(32)))
    command = f"""
      source {ROOT / 'common.sh'}
      PROMPTS_SOURCE={prompts}
      export PROMPTS_SOURCE
      selected=$(prepare_prompt_slice 22 10 test_moviegen32)
      printf '%s|%s|%s\n' "$(wc -l < "$selected")" "$(head -n 1 "$selected")" "$(tail -n 1 "$selected")"
    """
    result = subprocess.run(
        ["bash", "-c", command], check=True, capture_output=True, text=True
    )

    assert result.stdout.strip() == "22|prompt 10|prompt 31"


def test_expansion_b1_dry_run_uses_gpu_6_and_7(tmp_path):
    prompts = tmp_path / "moviegen32.txt"
    prompts.write_text("".join(f"prompt {index}\n" for index in range(32)))
    holdout = tmp_path / "hy_scenes_holdout.json"
    holdout.write_text('{"scenes": [{"id": "official_06"}]}')
    result = subprocess.run(
        ["bash", str(ROOT / "run_expansion_b1_serial.sh")],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "HOME": str(tmp_path),
            "PROMPTS_SOURCE": str(prompts),
            "HY_SCENES": str(holdout),
        },
    )

    assert "GPU 6" in result.stdout
    assert "GPU 7" in result.stdout
    assert "prompts 10-31" in result.stdout
    assert "Causal" in result.stdout
    assert "LongCat" in result.stdout
    assert "HY official cases 6-10" in result.stdout
    assert "No GPU commands executed" in result.stdout


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


def test_moviegen_manifest_merges_split_causal_roots(tmp_path):
    prompts = [f"scene {index}" for index in range(4)]
    roots = [("legacy", tmp_path / "legacy"), ("b1", tmp_path / "b1")]
    for index, prompt in enumerate(prompts):
        root = roots[0][1] if index < 2 else roots[1][1]
        for variant in moviegen_manifest.VARIANTS:
            directory = root / variant
            directory.mkdir(parents=True, exist_ok=True)
            (directory / moviegen_manifest.source_name("causal_forcing", prompt, index)).write_bytes(
                f"{variant}-{index}".encode()
            )

    records = moviegen_manifest.build_records(
        baseline="causal_forcing",
        roots=roots,
        prompts=prompts,
        ffprobe_bin="ffprobe",
        check_decode=False,
    )
    output = tmp_path / "normalized"
    moviegen_manifest.materialize(records, output)

    assert len(records) == 4 * 5
    assert (output / "bf16" / "0-0.mp4").is_symlink()
    assert (output / "trq_int2" / "3-0.mp4").is_symlink()
    assert {record["source_label"] for record in records[:10]} == {"legacy"}
    assert {record["source_label"] for record in records[10:]} == {"b1"}


def test_moviegen_manifest_accepts_normalized_b1_and_native_b2_causal_names(tmp_path):
    prompts = ["legacy prompt", "new prompt"]
    roots = [("b1", tmp_path / "b1"), ("b2", tmp_path / "b2")]
    for variant in moviegen_manifest.VARIANTS:
        b1_dir = roots[0][1] / variant
        b2_dir = roots[1][1] / variant
        b1_dir.mkdir(parents=True, exist_ok=True)
        b2_dir.mkdir(parents=True, exist_ok=True)
        (b1_dir / "0-0.mp4").write_bytes(f"b1-{variant}".encode())
        (b2_dir / moviegen_manifest.source_name("causal_forcing", prompts[1], 1)).write_bytes(
            f"b2-{variant}".encode()
        )

    records = moviegen_manifest.build_records(
        baseline="causal_forcing",
        roots=roots,
        prompts=prompts,
        ffprobe_bin="ffprobe",
        check_decode=False,
    )

    assert len(records) == 10
    assert {record["source_label"] for record in records[:5]} == {"b1"}
    assert {record["source_label"] for record in records[5:]} == {"b2"}


def test_moviegen_manifest_rejects_ambiguous_duplicate(tmp_path):
    prompt = "duplicate scene"
    roots = [("legacy", tmp_path / "legacy"), ("b1", tmp_path / "b1")]
    filename = moviegen_manifest.source_name("causal_forcing", prompt, 0)
    for _, root in roots:
        directory = root / "bf16"
        directory.mkdir(parents=True)
        (directory / filename).write_bytes(b"video")

    try:
        moviegen_manifest.find_unique_source(
            roots=roots, directory="bf16", filename=filename
        )
    except ValueError as exc:
        assert "ambiguous duplicate" in str(exc)
    else:
        raise AssertionError("duplicate result roots must fail closed")


def test_moviegen128_prompt_superset_preserves_b1_prefix():
    prompts = ROOT.parent.parent / "temporalresidualkvquant/assets/moviegenbench_resume_128.txt"
    prefix = ROOT.parent.parent / "temporalresidualkvquant/assets/moviegenbench_resume_32.txt"

    result = prompt_validator.validate(
        prompts,
        expected=128,
        prefix_path=prefix,
        prefix_count=32,
    )

    assert result["count"] == 128
    assert result["prefix_count"] == 32
    assert result["causal_filename_collisions"] == 0
    assert len(result["sha256"]) == 64


def test_moviegen_prompt_validator_rejects_causal_filename_collision(tmp_path):
    prompts = tmp_path / "prompts.txt"
    shared = "x" * 100
    prompts.write_text(f"{shared} first\n{shared} second\n", encoding="utf-8")

    try:
        prompt_validator.validate(prompts, expected=2)
    except ValueError as exc:
        assert "filename collision" in str(exc)
    else:
        raise AssertionError("Causal filename collisions must fail closed")


def test_expansion_b2_dry_run_uses_two_disjoint_shards(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "run_expansion_b2_moviegen128.sh")],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "HOME": str(tmp_path),
            "GPU_A": "2",
            "GPU_B": "4",
            "RUN_ID": "b2-test",
        },
    )

    assert "Phase A Causal: GPU 2 indices 32-79" in result.stdout
    assert "GPU 4 indices 80-127" in result.stdout
    assert "Phase B LongCat" in result.stdout
    assert "Existing decodable outputs" in result.stdout
    assert "No GPU commands executed" in result.stdout


def test_b2_evaluation_dry_run_requires_128_prompts_and_two_gpus(tmp_path):
    roots = [tmp_path / name for name in ("causal_b1", "causal_b2", "longcat_b1", "longcat_b2")]
    for root in roots:
        root.mkdir()
    result = subprocess.run(
        ["bash", str(ROOT / "run_b2_moviegen128_evaluation.sh")],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "HOME": str(tmp_path),
            "CAUSAL_GPU": "2",
            "LONGCAT_GPU": "4",
            "CAUSAL_B1_ROOT": str(roots[0]),
            "CAUSAL_B2_ROOT": str(roots[1]),
            "LONGCAT_B1_ROOT": str(roots[2]),
            "LONGCAT_B2_ROOT": str(roots[3]),
        },
    )

    assert "GPU 2: Causal MovieGen128" in result.stdout
    assert "GPU 4: LongCat MovieGen128" in result.stdout
    assert "EXPECTED_PROMPTS=128" in result.stdout
    assert "No manifests, metrics, or GPU jobs were started" in result.stdout


def test_paired_bootstrap_reports_positive_trq_advantage():
    def summary(values):
        return {
            "per_video": [
                {
                    "idx": index,
                    "sample_idx": 0,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips,
                }
                for index, (psnr, ssim, lpips) in enumerate(values)
            ]
        }

    trq = summary([(12.0, 0.6, 0.2), (11.0, 0.5, 0.3)])
    naive = summary([(10.0, 0.4, 0.4), (10.0, 0.4, 0.5)])
    result = paired_metrics.paired_bootstrap(trq, naive, resamples=100, seed=7)

    assert result["paired_video_count"] == 2
    assert all(
        metric["trq_advantage_mean"] > 0 for metric in result["metrics"].values()
    )
    assert all(metric["paired_win_rate"] == 1.0 for metric in result["metrics"].values())


def test_b1_evaluation_dry_run_assigns_gpu_6_and_7(tmp_path):
    prompts = tmp_path / "moviegen32.txt"
    prompts.write_text("".join(f"prompt {index}\n" for index in range(32)))
    roots = [tmp_path / name for name in ("causal_old", "causal_b1", "longcat_old", "longcat_b1")]
    for root in roots:
        root.mkdir()
    result = subprocess.run(
        ["bash", str(ROOT / "run_b1_evaluation.sh")],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "PROMPTS_SOURCE": str(prompts),
            "CAUSAL_LEGACY_ROOT": str(roots[0]),
            "CAUSAL_B1_ROOT": str(roots[1]),
            "LONGCAT_LEGACY_ROOT": str(roots[2]),
            "LONGCAT_B1_ROOT": str(roots[3]),
            "HY_ROOT": str(tmp_path / "hy"),
            "HOME": str(tmp_path),
        },
    )

    assert "GPU 6" in result.stdout
    assert "GPU 7" in result.stdout
    assert "No manifests, metrics, or GPU jobs were started" in result.stdout
