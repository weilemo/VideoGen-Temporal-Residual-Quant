#!/usr/bin/env python3
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--dst", required=True)
p.add_argument("--copy-split", action="store_true")
args = p.parse_args()

src = Path(args.src)
dst = Path(args.dst)
dst.mkdir(parents=True, exist_ok=True)
for pth in dst.glob("*.mp4"):
    pth.unlink()
files = sorted(src.glob("*.mp4"), key=lambda p: int(p.name.split("-")[0]))
if not files:
    raise SystemExit(f"no mp4 found in {src}")
for pth in files:
    idx = int(pth.name.split("-")[0])
    (dst / f"{idx}-0_ema.mp4").symlink_to(pth.resolve())
print(f"linked {len(files)} videos: {src} -> {dst}")
