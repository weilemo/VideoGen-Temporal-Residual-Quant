#!/usr/bin/env python3
"""Create the stable MovieGen10 filename layout expected by VBench."""

import argparse
import re
from pathlib import Path


def sort_key(path: Path):
    numbers = re.findall(r"\d+", path.stem)
    return (int(numbers[0]) if numbers else 10**9, path.as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--expected-videos", type=int, default=10)
    args = parser.parse_args()

    files = sorted(args.src.rglob("*.mp4"), key=sort_key)
    if len(files) != args.expected_videos:
        raise SystemExit(
            f"expected exactly {args.expected_videos} videos in {args.src}, "
            f"found {len(files)}"
        )

    args.dst.mkdir(parents=True, exist_ok=True)
    for old in args.dst.glob("*.mp4"):
        old.unlink()
    for index, video in enumerate(files):
        (args.dst / f"{index}-0_ema.mp4").symlink_to(video.resolve())
    print(f"linked {len(files)} videos: {args.src} -> {args.dst}")


if __name__ == "__main__":
    main()
