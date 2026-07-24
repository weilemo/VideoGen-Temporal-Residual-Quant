#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import re

import imageio.v3 as iio
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

try:
    import lpips  # type: ignore
    LPIPS_IMPORT_ERROR = None
except Exception as exc:
    lpips = None
    LPIPS_IMPORT_ERROR = exc


def read_video(path: Path, max_frames=None, start_frame=0):
    frames = []
    for i, frame in enumerate(iio.imiter(path)):
        if i < start_frame:
            continue
        if max_frames is not None and len(frames) >= max_frames:
            break
        frames.append(frame[..., :3])
    if not frames:
        raise ValueError(f"no frames: {path}")
    return np.stack(frames, axis=0)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def evaluate_directories(
    ref_dir,
    cmp_dir,
    *,
    max_frames=None,
    start_frame=0,
    device="cuda",
    match_by_index=False,
    strict_shape=False,
    require_lpips=False,
):
    ref_dir = Path(ref_dir)
    cmp_dir = Path(cmp_dir)
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"reference directory does not exist: {ref_dir}")
    if not cmp_dir.is_dir():
        raise FileNotFoundError(f"comparison directory does not exist: {cmp_dir}")

    torch_device = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    lpips_model = None
    if lpips is not None:
        lpips_model = lpips.LPIPS(net="alex").to(torch_device).eval()
    elif require_lpips:
        raise RuntimeError(
            "LPIPS is required for this evaluation but could not be imported. "
            "Install the analysis extras with "
            "`pip install -e 'temporalresidualkvquant[analysis]'`. "
            f"Import error: {LPIPS_IMPORT_ERROR!r}"
        )

    rows = []
    ref_videos = list(ref_dir.glob("*.mp4"))
    cmp_videos = list(cmp_dir.glob("*.mp4"))
    if not ref_videos:
        raise FileNotFoundError(f"reference directory contains no MP4 videos: {ref_dir}")
    if not cmp_videos:
        raise FileNotFoundError(f"comparison directory contains no MP4 videos: {cmp_dir}")
    if match_by_index:
        ref_map = video_index_map(ref_videos)
        cmp_map = video_index_map(cmp_videos)
        missing = sorted(set(ref_map).difference(cmp_map))
        extra = sorted(set(cmp_map).difference(ref_map))
        if missing or extra:
            raise FileNotFoundError(
                f"video index sets differ; missing comparison indices={missing}, "
                f"extra comparison indices={extra}"
            )
        pairs = [(key, ref_map[key], cmp_map[key]) for key in sorted(ref_map)]
    else:
        ref_map = {path.name: path for path in ref_videos}
        cmp_map = {path.name: path for path in cmp_videos}
        missing = sorted(set(ref_map).difference(cmp_map))
        extra = sorted(set(cmp_map).difference(ref_map))
        if missing or extra:
            raise FileNotFoundError(
                f"video filename sets differ; missing comparison files={missing}, "
                f"extra comparison files={extra}"
            )
        pairs = [((idx, 0), ref_map[name], cmp_map[name]) for idx, name in enumerate(sorted(ref_map))]

    for (idx, sample_idx), ref, cmp in pairs:
        a = read_video(ref, max_frames=max_frames, start_frame=start_frame)
        b = read_video(cmp, max_frames=max_frames, start_frame=start_frame)
        if strict_shape and a.shape != b.shape:
            raise ValueError(
                f"paired video shapes differ for index {(idx, sample_idx)}: "
                f"{ref.name}={a.shape}, {cmp.name}={b.shape}"
            )
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]
        cur_psnr = psnr(a, b)
        cur_ssim = float(np.mean([ssim(a[i], b[i], channel_axis=2, data_range=255) for i in range(n)]))
        cur_lpips = None
        if lpips_model is not None:
            vals = []
            with torch.no_grad():
                for i in range(n):
                    ta = torch.from_numpy(a[i]).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
                    tb = torch.from_numpy(b[i]).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
                    vals.append(float(lpips_model(ta.to(torch_device), tb.to(torch_device)).item()))
            cur_lpips = float(np.mean(vals))
        rows.append({
            "idx": idx,
            "sample_idx": sample_idx,
            "ref_name": ref.name,
            "cmp_name": cmp.name,
            "frames": n,
            "psnr": cur_psnr,
            "ssim": cur_ssim,
            "lpips": cur_lpips,
        })

    return {
        "ref_dir": str(ref_dir),
        "cmp_dir": str(cmp_dir),
        "num_videos": len(rows),
        "max_frames": max_frames,
        "start_frame": start_frame,
        "lpips_available": lpips_model is not None,
        "psnr_aggregation": "RGB global MSE per video, then mean PSNR across videos",
        "mean_psnr": float(np.mean([row["psnr"] for row in rows])),
        "mean_ssim": float(np.mean([row["ssim"] for row in rows])),
        "mean_lpips": (
            None if lpips_model is None else float(np.mean([row["lpips"] for row in rows]))
        ),
        "per_video": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--cmp-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Skip this many leading frames before computing paired metrics",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--match-by-index",
        action="store_true",
        help="Pair names by leading prompt/sample indices, accepting both 0-0 and 0_0 forms",
    )
    ap.add_argument(
        "--strict-shape",
        action="store_true",
        help="Fail when paired videos have different frame counts or frame shapes",
    )
    ap.add_argument(
        "--require-lpips",
        action="store_true",
        help="Fail instead of silently emitting null LPIPS when the lpips package is unavailable",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = evaluate_directories(
        args.ref_dir,
        args.cmp_dir,
        max_frames=args.max_frames or None,
        start_frame=args.start_frame,
        device=args.device,
        match_by_index=args.match_by_index,
        strict_shape=args.strict_shape,
        require_lpips=args.require_lpips,
    )
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_video"}, indent=2))


def video_index_map(paths):
    result = {}
    for path in paths:
        match = re.match(r"^(\d+)[-_](\d+)", path.name)
        if match is None:
            raise ValueError(f"cannot extract prompt/sample indices from video name: {path.name}")
        key = (int(match.group(1)), int(match.group(2)))
        if key in result:
            raise ValueError(f"duplicate video index {key}: {result[key].name}, {path.name}")
        result[key] = path
    return result

if __name__ == "__main__":
    main()
