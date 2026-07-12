#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

try:
    import lpips  # type: ignore
except Exception:
    lpips = None


def read_video(path: Path, max_frames=None):
    frames = []
    for i, frame in enumerate(iio.imiter(path)):
        if max_frames is not None and i >= max_frames:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--cmp-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ref_dir = Path(args.ref_dir)
    cmp_dir = Path(args.cmp_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    max_frames = args.max_frames or None

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    lpips_model = None
    if lpips is not None:
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    rows = []
    for ref in sorted(ref_dir.glob("*.mp4"), key=lambda p: int(p.name.split("-")[0])):
        idx = int(ref.name.split("-")[0])
        cmp = cmp_dir / ref.name
        if not cmp.exists():
            raise FileNotFoundError(f"missing cmp for {ref.name}: {cmp}")
        a = read_video(ref, max_frames=max_frames)
        b = read_video(cmp, max_frames=max_frames)
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]
        cur_psnr = float(np.mean([psnr(a[i], b[i]) for i in range(n)]))
        cur_ssim = float(np.mean([ssim(a[i], b[i], channel_axis=2, data_range=255) for i in range(n)]))
        cur_lpips = None
        if lpips_model is not None:
            vals = []
            with torch.no_grad():
                for i in range(n):
                    ta = torch.from_numpy(a[i]).permute(2,0,1).float().unsqueeze(0) / 127.5 - 1.0
                    tb = torch.from_numpy(b[i]).permute(2,0,1).float().unsqueeze(0) / 127.5 - 1.0
                    vals.append(float(lpips_model(ta.to(device), tb.to(device)).item()))
            cur_lpips = float(np.mean(vals))
        rows.append({"idx": idx, "frames": n, "psnr": cur_psnr, "ssim": cur_ssim, "lpips": cur_lpips})

    summary = {
        "ref_dir": str(ref_dir),
        "cmp_dir": str(cmp_dir),
        "num_videos": len(rows),
        "max_frames": max_frames,
        "lpips_available": lpips_model is not None,
        "mean_psnr": float(np.mean([r["psnr"] for r in rows])),
        "mean_ssim": float(np.mean([r["ssim"] for r in rows])),
        "mean_lpips": None if lpips_model is None else float(np.mean([r["lpips"] for r in rows])),
        "per_video": rows,
    }
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_video"}, indent=2))

if __name__ == "__main__":
    main()
