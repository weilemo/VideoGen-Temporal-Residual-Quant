#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path

import imageio.v2 as imageio


VIDEO_RE = re.compile(r"^(\d+)-(\d+)_(?:ema|regular)\.mp4$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split long videos into latent-aligned evaluation windows")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--latent-frames", type=int, required=True)
    parser.add_argument("--window", type=int, default=180)
    parser.add_argument("--stride", type=int, default=90)
    args = parser.parse_args()
    if args.latent_frames < args.window or args.window <= 0 or args.stride <= 0:
        raise ValueError("invalid latent window configuration")

    source = Path(args.src).expanduser()
    output = Path(args.dst).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(source.glob("*.mp4")):
        match = VIDEO_RE.match(path.name)
        if match is None:
            continue
        reader = imageio.get_reader(path)
        try:
            fps = float(reader.get_meta_data().get("fps", 16.0))
            pixel_frames = int(reader.count_frames())
            windows = []
            window_starts = set(range(0, args.latent_frames - args.window + 1, args.stride))
            window_starts.add(args.latent_frames - args.window)
            for start in sorted(window_starts):
                end = start + args.window
                pixel_start = round(start * pixel_frames / args.latent_frames)
                pixel_end = round(end * pixel_frames / args.latent_frames)
                target = output / f"p{match.group(1)}_n{match.group(2)}_l{start:04d}-{end:04d}.mp4"
                windows.append({
                    "latent_start": start,
                    "latent_end": end,
                    "pixel_start": pixel_start,
                    "pixel_end": max(pixel_start + 1, pixel_end),
                    "path": target,
                    "writer": imageio.get_writer(target, fps=fps, codec="libx264"),
                })
            try:
                for pixel_index, frame in enumerate(reader):
                    for window in windows:
                        if window["pixel_start"] <= pixel_index < window["pixel_end"]:
                            window["writer"].append_data(frame)
            finally:
                for window in windows:
                    window["writer"].close()
            for window in windows:
                rows.append({
                    "prompt_index": int(match.group(1)),
                    "sample_index": int(match.group(2)),
                    "latent_start": window["latent_start"],
                    "latent_end": window["latent_end"],
                    "pixel_start": window["pixel_start"],
                    "pixel_end": window["pixel_end"],
                    "path": str(window["path"]),
                })
        finally:
            reader.close()
    if not rows:
        raise ValueError(f"no indexed videos found in {source}")
    with (output / "window_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} latent-aligned windows in {output}")


if __name__ == "__main__":
    main()
