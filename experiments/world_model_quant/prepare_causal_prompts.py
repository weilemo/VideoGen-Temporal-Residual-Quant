#!/usr/bin/env python3
"""Build a Causal Forcing prompt shard containing only missing videos."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def output_name(prompt: str) -> str:
    return f"{prompt[:100]}.mp4"


def is_decodable(path: Path) -> bool:
    if not path.is_file():
        return False
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


def select_missing(
    prompts: list[str], start: int, count: int, output_dir: Path
) -> tuple[list[str], int]:
    selected = prompts[start : start + count]
    if len(selected) != count:
        raise ValueError(f"prompt slice {start}:{count} exceeds {len(prompts)} prompts")
    missing = [prompt for prompt in selected if not is_decodable(output_dir / output_name(prompt))]
    return missing, len(selected) - len(missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
    missing, reused = select_missing(prompts, args.start, args.count, args.output_dir)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text("".join(f"{prompt}\n" for prompt in missing), encoding="utf-8")
    print(args.destination)
    print(f"reused={reused} missing={len(missing)}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
