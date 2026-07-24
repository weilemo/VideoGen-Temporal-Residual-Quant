#!/usr/bin/env python3
"""Compute optical-flow proxies for HY-WorldPlay action preservation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


VARIANTS = ("bf16", "trq_int4", "trq_int2", "naive_int4", "naive_int2")


def motion_signature(path: Path, sample_every: int) -> dict[str, float]:
    try:
        import cv2
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("install opencv-python and imageio for HY action evaluation") from exc

    frames = []
    for index, frame in enumerate(iio.imiter(path)):
        if index % sample_every == 0:
            frames.append(cv2.resize(frame[..., :3], (256, 144)))
    if len(frames) < 2:
        raise ValueError(f"need at least two sampled frames: {path}")

    horizontal = []
    radial = []
    yy, xx = np.mgrid[0:144, 0:256].astype(np.float32)
    xx = (xx - 127.5) / 127.5
    yy = (yy - 71.5) / 71.5
    radius = np.sqrt(xx * xx + yy * yy) + 1e-6
    for previous, current in zip(frames, frames[1:]):
        a = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        horizontal.append(float(np.median(flow[..., 0])))
        radial_flow = (flow[..., 0] * xx + flow[..., 1] * yy) / radius
        radial.append(float(np.median(radial_flow)))
    return {
        "horizontal_flow": float(np.median(horizontal)),
        "radial_flow": float(np.median(radial)),
    }


def one_video(directory: Path) -> Path:
    videos = sorted(directory.rglob("*.mp4"))
    if len(videos) != 1:
        raise ValueError(f"expected one video below {directory}, found {len(videos)}")
    return videos[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-every", type=int, default=4)
    args = parser.parse_args()

    action_config = json.loads(args.actions.read_text(encoding="utf-8"))
    action_ids = sorted({item[key] for item in action_config["counterfactual_pairs"] for key in ("left", "right")})
    rows = []
    signatures = {}
    scene_dirs = sorted(
        path
        for path in args.root.iterdir()
        if path.is_dir() and all((path / action_id).is_dir() for action_id in action_ids)
    )
    if not scene_dirs:
        raise FileNotFoundError(f"no complete HY scene directories below {args.root}")
    for scene_dir in scene_dirs:
        signatures[scene_dir.name] = {}
        for action_id in action_ids:
            signatures[scene_dir.name][action_id] = {}
            for variant in VARIANTS:
                video = one_video(scene_dir / action_id / variant)
                signature = motion_signature(video, args.sample_every)
                signatures[scene_dir.name][action_id][variant] = signature
                reference = signatures[scene_dir.name][action_id].get("bf16")
                row = {"scene": scene_dir.name, "action": action_id, "variant": variant, **signature}
                if reference is not None:
                    row["horizontal_error_vs_bf16"] = abs(signature["horizontal_flow"] - reference["horizontal_flow"])
                    row["radial_error_vs_bf16"] = abs(signature["radial_flow"] - reference["radial_flow"])
                rows.append(row)

    pairs = []
    for scene, scene_values in signatures.items():
        for pair in action_config["counterfactual_pairs"]:
            left = pair["left"]
            right = pair["right"]
            metric = "horizontal_flow" if pair["id"] == "steering" else "radial_flow"
            bf16_delta = scene_values[left]["bf16"][metric] - scene_values[right]["bf16"][metric]
            for variant in VARIANTS:
                delta = scene_values[left][variant][metric] - scene_values[right][variant][metric]
                pairs.append({
                    "scene": scene,
                    "pair": pair["id"],
                    "variant": variant,
                    "metric": metric,
                    "bf16_separation": bf16_delta,
                    "variant_separation": delta,
                    "sign_preserved": bool(bf16_delta * delta > 0),
                    "separation_ratio": abs(delta) / (abs(bf16_delta) + 1e-8),
                })

    for row in rows:
        for key, value in list(row.items()):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite action metric: {row}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "action_metrics.json").write_text(
        json.dumps({"motion_signatures": rows, "counterfactual_pairs": pairs}, indent=2),
        encoding="utf-8",
    )
    for name, values in (("motion_signatures.csv", rows), ("counterfactual_pairs.csv", pairs)):
        with (args.output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)


if __name__ == "__main__":
    main()
