#!/usr/bin/env python3
"""Build one validated MovieGen32 view from split experiment result roots."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


VARIANTS = ("bf16", "trq_int4", "trq_int2", "naive_int4", "naive_int2")


def parse_source(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("source root must have the form LABEL=PATH")
    return label, Path(path).expanduser().resolve()


def load_prompts(path: Path, expected: int) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) < expected:
        raise ValueError(f"expected at least {expected} prompts in {path}, found {len(prompts)}")
    return prompts[:expected]


def is_decodable(path: Path, ffprobe_bin: str) -> bool:
    result = subprocess.run(
        [
            ffprobe_bin,
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


def source_name(baseline: str, prompt: str, index: int) -> str:
    if baseline == "causal_forcing":
        return f"{prompt[:100]}.mp4"
    return f"{index}-0.mp4"


def find_unique_source(
    *,
    roots: list[tuple[str, Path]],
    directory: str,
    filename: str,
) -> tuple[str, Path]:
    matches: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for label, root in roots:
        candidate = root / directory / filename
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in seen:
                matches.append((label, candidate.resolve()))
                seen.add(resolved)
    if not matches:
        searched = ", ".join(str(root / directory / filename) for _, root in roots)
        raise FileNotFoundError(f"missing result; searched: {searched}")
    if len(matches) > 1:
        found = ", ".join(f"{label}={path}" for label, path in matches)
        raise ValueError(f"ambiguous duplicate result for {directory}/{filename}: {found}")
    return matches[0]


def build_records(
    *,
    baseline: str,
    roots: list[tuple[str, Path]],
    prompts: list[str],
    ffprobe_bin: str,
    check_decode: bool,
) -> list[dict]:
    directories = list(VARIANTS)
    if baseline == "longcat":
        directories.insert(0, "prefix_bf16")
    records = []
    for index, prompt in enumerate(prompts):
        filename = source_name(baseline, prompt, index)
        for directory in directories:
            label, path = find_unique_source(
                roots=roots, directory=directory, filename=filename
            )
            if check_decode and not is_decodable(path, ffprobe_bin):
                raise ValueError(f"not decodable: {path}")
            records.append(
                {
                    "prompt_index": index,
                    "sample_index": 0,
                    "prompt": prompt,
                    "variant": directory,
                    "source_label": label,
                    "source_path": str(path),
                    "normalized_name": f"{index}-0.mp4",
                    "size_bytes": path.stat().st_size,
                    "decodable": check_decode,
                }
            )
    return records


def materialize(records: list[dict], output_root: Path) -> None:
    for record in records:
        destination = output_root / record["variant"] / record["normalized_name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = Path(record["source_path"])
        if destination.is_symlink() and destination.resolve() == source:
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"refusing to replace existing normalized result: {destination}"
            )
        destination.symlink_to(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", required=True, choices=("causal_forcing", "longcat")
    )
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        action="append",
        type=parse_source,
        required=True,
        help="Repeat LABEL=PATH for legacy and B1 result roots",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int, default=32)
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--skip-decode-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.expected_prompts <= 0:
        parser.error("--expected-prompts must be positive")
    roots = args.source_root
    for label, root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"source root does not exist: {label}={root}")
    prompts_path = args.prompts.expanduser().resolve()
    prompts = load_prompts(prompts_path, args.expected_prompts)
    records = build_records(
        baseline=args.baseline,
        roots=roots,
        prompts=prompts,
        ffprobe_bin=args.ffprobe_bin,
        check_decode=not args.skip_decode_check,
    )

    output_root = args.output_root.expanduser().resolve()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline": args.baseline,
        "prompts_path": str(prompts_path),
        "expected_prompts": args.expected_prompts,
        "variants": list(VARIANTS),
        "source_roots": [{"label": label, "path": str(root)} for label, root in roots],
        "decode_check": not args.skip_decode_check,
        "record_count": len(records),
        "records": records,
    }
    if not args.dry_run:
        materialize(records, output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "baseline": args.baseline,
                "expected_prompts": args.expected_prompts,
                "record_count": len(records),
                "output_root": str(output_root),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
