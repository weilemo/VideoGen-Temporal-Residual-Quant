#!/usr/bin/env python3
"""Validate a canonical MovieGen prompt superset before GPU execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(
    prompts_path: Path,
    *,
    expected: int,
    prefix_path: Path | None = None,
    prefix_count: int = 0,
) -> dict:
    prompts = load_prompts(prompts_path)
    if len(prompts) != expected:
        raise ValueError(f"expected exactly {expected} prompts, found {len(prompts)}")
    if len(set(prompts)) != len(prompts):
        raise ValueError("prompt set contains duplicate full prompts")

    filenames: dict[str, int] = {}
    for index, prompt in enumerate(prompts):
        filename = f"{prompt[:100]}.mp4"
        if filename in filenames:
            raise ValueError(
                f"Causal output filename collision between indices "
                f"{filenames[filename]} and {index}: {filename}"
            )
        filenames[filename] = index

    if prefix_path is not None:
        prefix = load_prompts(prefix_path)
        if len(prefix) < prefix_count:
            raise ValueError(
                f"prefix source has {len(prefix)} prompts, expected at least {prefix_count}"
            )
        if prompts[:prefix_count] != prefix[:prefix_count]:
            raise ValueError(
                f"first {prefix_count} prompts do not exactly match {prefix_path}"
            )

    digest = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    return {
        "prompts": str(prompts_path.resolve()),
        "count": len(prompts),
        "sha256": digest,
        "prefix_count": prefix_count,
        "causal_filename_collisions": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=128)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--prefix-count", type=int, default=0)
    args = parser.parse_args()
    if args.expected <= 0 or args.prefix_count < 0:
        parser.error("--expected must be positive and --prefix-count non-negative")
    result = validate(
        args.prompts.expanduser().resolve(),
        expected=args.expected,
        prefix_path=args.prefix.expanduser().resolve() if args.prefix else None,
        prefix_count=args.prefix_count,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
