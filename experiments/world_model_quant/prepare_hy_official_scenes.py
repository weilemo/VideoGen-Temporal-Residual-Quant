#!/usr/bin/env python3
"""Download the official HY-WorldPlay test cases and build fixed splits."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/Tencent-Hunyuan/"
    "HY-WorldPlay/main/assets"
)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"empty download: {url}")
    destination.write_bytes(payload)


def materialize(source: Path | None, url: str, destination: Path) -> None:
    if source is not None:
        if not source.is_file():
            raise FileNotFoundError(f"missing official HY asset: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return
    download(url, destination)


def load_cases(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10:
        raise ValueError(f"expected 10 official HY cases, found {len(rows)}")
    cases = []
    for index, row in enumerate(rows, start=1):
        image_name = str(row.get("image_name", "")).removeprefix("./assets/img/")
        caption = str(row.get("caption", "")).strip()
        if not image_name or not caption:
            raise ValueError(f"invalid official HY case {index}")
        cases.append({"id": f"official_{index:02d}", "image_name": image_name, "prompt": caption})
    return cases


def build_manifest(cases: list[dict[str, str]], image_dir: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": "Tencent-Hunyuan/HY-WorldPlay assets/test_case.csv",
        "scenes": [
            {
                "id": case["id"],
                "image_path": str((image_dir / case["image_name"]).resolve()),
                "prompt": case["prompt"],
            }
            for case in cases
        ],
    }


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="local checkout of the official HY-WorldPlay assets directory",
    )
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    csv_path = root / "test_case.csv"
    image_dir = root / "img"
    source_dir = args.source_dir.expanduser().resolve() if args.source_dir else None
    materialize(
        source_dir / "test_case.csv" if source_dir else None,
        f"{args.base_url}/test_case.csv",
        csv_path,
    )
    cases = load_cases(csv_path)
    for case in cases:
        name = case["image_name"]
        materialize(
            source_dir / "img" / name if source_dir else None,
            f"{args.base_url}/img/{name}",
            image_dir / name,
        )

    dev = root / "hy_scenes_dev.json"
    holdout = root / "hy_scenes_holdout.json"
    write_manifest(dev, build_manifest(cases[:5], image_dir))
    write_manifest(holdout, build_manifest(cases[5:], image_dir))
    print(dev)
    print(holdout)


if __name__ == "__main__":
    main()
