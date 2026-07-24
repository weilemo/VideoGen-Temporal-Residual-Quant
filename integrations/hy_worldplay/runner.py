#!/usr/bin/env python3
"""Build reproducible HY-WorldPlay quantization commands and manifests."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


@dataclass(frozen=True)
class Variant:
    quant_type: str


VARIANTS = {
    "bf16": Variant("none"),
    "trq_int4": Variant("trq-int4"),
    "trq_int2": Variant("trq-int2"),
    "naive_int4": Variant("packed-naive-int4"),
    "naive_int2": Variant("packed-naive-int2"),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def source_root(config: dict[str, Any]) -> Path:
    override = os.environ.get("HY_WORLDPLAY_SOURCE")
    if override:
        return Path(override).expanduser().resolve()
    configured = Path(str(config["source_root"]))
    if configured.is_absolute():
        return configured
    return (REPO_ROOT / configured).resolve()


def resolve_from(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def validate_config(config: dict[str, Any]) -> None:
    memory = int(config["memory_frames"])
    context = int(config["temporal_context_size"])
    predicted = int(config["pred_latent_size"])
    if not 0 < context < memory:
        raise ValueError("temporal_context_size must be between 0 and memory_frames")
    if memory - context != predicted:
        raise ValueError(
            "memory_frames - temporal_context_size must equal pred_latent_size"
        )
    if int(config["num_chunks"]) <= 0:
        raise ValueError("num_chunks must be positive")
    if int(config["seed"]) != 0:
        raise ValueError(
            "HY-WorldPlay upstream currently hard-codes seed 0; other seeds are unsupported"
        )


def build_command(
    config: dict[str, Any],
    actions: dict[str, Any],
    variant_name: str,
    action_name: str,
) -> tuple[list[str], dict[str, Any]]:
    validate_config(config)
    if variant_name not in VARIANTS:
        raise ValueError(f"unknown variant: {variant_name}")
    prompt_arg = str(config["prompt"])
    if resolve_from(REPO_ROOT, prompt_arg).exists():
        prompt_arg = str(resolve_from(REPO_ROOT, prompt_arg))
    action_map = actions.get("actions", {})
    if action_name not in action_map:
        raise ValueError(f"unknown action: {action_name}")

    upstream = source_root(config)
    variant = VARIANTS[variant_name]
    output = resolve_from(
        REPO_ROOT,
        str(Path(config["output_root"]) / config["experiment_id"] / action_name / variant_name),
    )
    quant = config["quantization"]
    command = [
        "torchrun",
        "--nproc_per_node=1",
        "--standalone",
        str(upstream / "experiments/HY-WorldPlay/wan/generate.py"),
        "--input",
        prompt_arg,
        "--image_path",
        str(resolve_from(upstream, str(config["image_path"]))),
        "--num_chunk",
        str(config["num_chunks"]),
        "--pose",
        str(action_map[action_name]),
        "--ar_model_path",
        str(resolve_from(upstream, str(config["ar_model_path"]))),
        "--ckpt_path",
        str(resolve_from(upstream, str(config["ckpt_path"]))),
        "--offload_text_encoder",
        "--out",
        str(output),
        "--memory_frames",
        str(config["memory_frames"]),
        "--temporal_context_size",
        str(config["temporal_context_size"]),
        "--pred_latent_size",
        str(config["pred_latent_size"]),
        "--quant_type",
        variant.quant_type,
        "--quant_block_size",
        str(quant["block_size"]),
        "--cache_num_k_centroids",
        str(quant["cache_num_k_centroids"]),
        "--cache_num_v_centroids",
        str(quant["cache_num_v_centroids"]),
        "--trq_anchor_bits",
        str(quant.get("trq_anchor_bits", 4)),
        "--trq_predictor_stride",
        str(quant.get("trq_predictor_stride", 880)),
        "--trq_predictor_mode",
        str(quant.get("trq_predictor_mode", "identity")),
        "--trq_group_size",
        str(quant.get("trq_group_size", quant["block_size"])),
        "--trq_k_bits",
        "0",
        "--trq_v_bits",
        "0",
        "--kmeans_max_iters",
        str(quant["kmeans_max_iters"]),
        "--num_prq_stages",
        str(quant["num_prq_stages"]),
    ]
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "variant": variant_name,
        "quant_type": variant.quant_type,
        "action_id": action_name,
        "pose": action_map[action_name],
        "seed": config["seed"],
        "source_root": str(upstream),
        "output_dir": str(output),
        "controlled_variables": {
            "prompt": config["prompt"],
            "image_path": str(resolve_from(upstream, str(config["image_path"]))),
            "num_chunks": config["num_chunks"],
            "memory_frames": config["memory_frames"],
            "temporal_context_size": config["temporal_context_size"],
            "pred_latent_size": config["pred_latent_size"],
        },
        "quantization": quant,
        "command": command,
    }
    return command, manifest


def validate_runtime(config: dict[str, Any]) -> None:
    upstream = source_root(config)
    required = [
        upstream / "experiments/HY-WorldPlay/wan/generate.py",
        resolve_from(upstream, str(config["image_path"])),
        resolve_from(upstream, str(config["ar_model_path"])),
        resolve_from(upstream, str(config["ckpt_path"])),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"missing HY-WorldPlay runtime assets:\n  - {formatted}"
        )

def run_one(
    config: dict[str, Any],
    actions: dict[str, Any],
    variant: str,
    action: str,
    dry_run: bool,
) -> None:
    command, manifest = build_command(config, actions, variant, action)
    print(f"[{action}/{variant}] {shlex.join(command)}")
    if dry_run:
        return
    validate_runtime(config)
    output = Path(manifest["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    completed = [path for path in output.glob("*.mp4") if is_decodable_video(path)]
    if completed:
        print(f"[skip] decodable output already exists: {completed[0]}")
        return
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(REPO_ROOT / "temporalresidualkvquant/src"),
        str(source_root(config)),
        str(source_root(config) / "experiments/HY-WorldPlay"),
    ])
    subprocess.run(command, cwd=source_root(config), env=env, check=True)


def is_decodable_video(path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and "video" in result.stdout.splitlines()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--actions", type=Path, default=HERE / "actions.json")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="bf16")
    parser.add_argument("--action", default="canonical")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="run every variant for every action in actions.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    actions = load_json(args.actions)
    if args.matrix:
        for action in actions["actions"]:
            for variant in VARIANTS:
                run_one(config, actions, variant, action, args.dry_run)
        return
    run_one(config, actions, args.variant, args.action, args.dry_run)


if __name__ == "__main__":
    main()
