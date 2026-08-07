#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


VIDEO_RE = re.compile(r"^(\d+)-(\d+)_(?:ema|regular)\.mp4$")


def load_run(config: str, seed: int, directory: Path):
    result = {}
    for path in sorted(directory.glob("*.mp4")):
        match = VIDEO_RE.match(path.name)
        if match:
            result[(config, seed, int(match.group(1)), int(match.group(2)))] = path
    if not result:
        raise ValueError(f"no indexed mp4 files found in {directory}")
    return result


def read_frames(path: Path) -> tuple[list[np.ndarray], float]:
    reader = imageio.get_reader(path)
    try:
        frames = [np.asarray(frame) for frame in reader]
        fps = float(reader.get_meta_data().get("fps", 16.0))
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"video contains no frames: {path}")
    return frames, fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paired videos, contact sheets, and failure-tag template")
    parser.add_argument("--run", action="append", required=True, metavar="CONFIG,SEED,DIR")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    videos = {}
    for value in args.run:
        parts = value.split(",", 2)
        if len(parts) != 3:
            parser.error(f"invalid --run {value!r}; expected CONFIG,SEED,DIR")
        config, seed, directory = parts
        videos.update(load_run(config, int(seed), Path(directory).expanduser()))

    output = Path(args.output_dir).expanduser().resolve()
    contacts = output / "contact_sheets"
    comparisons = output / "side_by_side"
    contacts.mkdir(parents=True, exist_ok=True)
    comparisons.mkdir(parents=True, exist_ok=True)
    tag_rows = []
    candidates = sorted({key[0] for key in videos} - {"bf16_a", "bf16_b"})
    reference_keys = sorted(key for key in videos if key[0] == "bf16_a")
    for _, seed, prompt, sample in reference_keys:
        reference_path = videos[("bf16_a", seed, prompt, sample)]
        reference_frames, fps = read_frames(reference_path)
        indices = [0, len(reference_frames) // 2, len(reference_frames) - 1]
        for config in candidates:
            candidate_key = (config, seed, prompt, sample)
            if candidate_key not in videos:
                raise ValueError(f"missing candidate video {candidate_key}")
            candidate_frames, candidate_fps = read_frames(videos[candidate_key])
            length = min(len(reference_frames), len(candidate_frames))
            if reference_frames[0].shape != candidate_frames[0].shape:
                raise ValueError(f"video shape mismatch for {candidate_key}")
            comparison_path = comparisons / f"{config}_p{prompt}_s{seed}_n{sample}.mp4"
            writer = imageio.get_writer(comparison_path, fps=min(fps, candidate_fps), codec="libx264")
            try:
                for index in range(length):
                    writer.append_data(np.concatenate([reference_frames[index], candidate_frames[index]], axis=1))
            finally:
                writer.close()
            candidate_indices = [min(index, len(candidate_frames) - 1) for index in indices]
            top = np.concatenate([reference_frames[index] for index in indices], axis=1)
            bottom = np.concatenate(
                [candidate_frames[index] for index in candidate_indices], axis=1
            )
            imageio.imwrite(
                contacts / f"{config}_p{prompt}_s{seed}_n{sample}.png",
                np.concatenate([top, bottom], axis=0),
            )
            tag_rows.append({
                "config": config,
                "prompt_index": prompt,
                "seed": seed,
                "sample_index": sample,
                "identity_switch": "",
                "background_jump": "",
                "texture_repetition": "",
                "motion_freeze": "",
                "color_drift": "",
                "black_or_nan": "",
                "catastrophe": "",
                "bf16_present": "",
                "notes": "",
            })

    with (output / "failure_tags.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tag_rows[0]))
        writer.writeheader()
        writer.writerows(tag_rows)
    print(f"Paired qualitative review artifacts written to {output}")


if __name__ == "__main__":
    main()
