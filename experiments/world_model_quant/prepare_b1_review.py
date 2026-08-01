#!/usr/bin/env python3
"""Build an anonymous, grouped B1 catastrophe/action review package."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path


CANDIDATES = ("trq_int4", "trq_int2", "naive_int4", "naive_int2")
VARIANTS = ("bf16", *CANDIDATES)
ACTION_ORDER = ("turn_left", "turn_right", "forward", "backward")
PANEL_LABELS = ("A", "B", "C", "D", "E")
APP_FILES = ("index.html", "app.js", "styles.css")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def anonymous_id(rng: random.Random, used: set[str], prefix: str) -> str:
    while True:
        value = f"{prefix}-{rng.getrandbits(48):012x}"
        if value not in used:
            used.add(value)
            return value


def safe_link(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"review source does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source:
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace review media: {destination}")
    destination.symlink_to(source)


def index_moviegen_manifest(path: Path, expected_prompts: int) -> dict:
    payload = read_json(path)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"manifest has no records list: {path}")
    indexed: dict[tuple[int, str], dict] = {}
    for record in records:
        key = (int(record["prompt_index"]), str(record["variant"]))
        if key in indexed:
            raise ValueError(f"duplicate MovieGen record {key} in {path}")
        indexed[key] = record
    for prompt_index in range(expected_prompts):
        for variant in VARIANTS:
            if (prompt_index, variant) not in indexed:
                raise ValueError(f"missing MovieGen record {(prompt_index, variant)} in {path}")
    return indexed


def build_movie_tasks(
    *,
    baseline: str,
    manifest_path: Path,
    expected_prompts: int,
    rng: random.Random,
    used: set[str],
    public_root: Path,
) -> tuple[list[dict], dict]:
    indexed = index_moviegen_manifest(manifest_path, expected_prompts)
    tasks: list[dict] = []
    private: dict[str, dict] = {}
    for prompt_index in range(expected_prompts):
        bf16 = indexed[(prompt_index, "bf16")]
        prompt = str(bf16["prompt"])
        task_id = anonymous_id(rng, used, "V")
        variants = list(VARIANTS)
        rng.shuffle(variants)
        media: dict[str, str] = {}
        private_items: dict[str, dict] = {}
        for label, variant in zip(PANEL_LABELS, variants, strict=True):
            record = indexed[(prompt_index, variant)]
            relative = Path("media") / "video" / task_id / f"{label}.mp4"
            safe_link(Path(record["source_path"]), public_root / relative)
            media[label] = relative.as_posix()
            private_items[label] = {
                "variant": variant,
                "source_path": str(Path(record["source_path"]).resolve()),
            }
        task = {
            "id": task_id,
            "kind": "catastrophe",
            "baseline": baseline,
            "prompt_index": prompt_index,
            "prompt": prompt,
            "labels": list(PANEL_LABELS),
            "media": media,
        }
        if baseline == "LongCat Video":
            task["conditioning_frames"] = 13
        tasks.append(task)
        private[task_id] = {
            "kind": "catastrophe",
            "baseline": baseline,
            "prompt_index": prompt_index,
            "items": private_items,
        }
    return tasks, private


def discover_hy_scenes(root: Path) -> list[Path]:
    scenes = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        if all((candidate / action).is_dir() for action in ACTION_ORDER):
            scenes.append(candidate)
    return scenes


def unique_action_video(root: Path, scene: Path, action: str, variant: str) -> Path:
    directory = scene / action / variant
    matches = sorted(directory.glob("*.mp4")) if directory.is_dir() else []
    if len(matches) != 1:
        raise ValueError(
            f"expected one video for {scene.name}/{action}/{variant}, found {len(matches)}"
        )
    return matches[0].resolve()


def build_action_tasks(
    *,
    root: Path,
    expected_scenes: int,
    rng: random.Random,
    used: set[str],
    public_root: Path,
) -> tuple[list[dict], dict]:
    scenes = discover_hy_scenes(root)
    if len(scenes) != expected_scenes:
        raise ValueError(f"expected {expected_scenes} HY scenes in {root}, found {len(scenes)}")
    tasks: list[dict] = []
    private: dict[str, dict] = {}
    for scene_index, scene in enumerate(scenes):
        task_id = anonymous_id(rng, used, "A")
        variants = list(VARIANTS)
        rng.shuffle(variants)
        media: dict[str, dict[str, str]] = {}
        private_items: dict[str, dict] = {}
        for label, variant in zip(PANEL_LABELS, variants, strict=True):
            media[label] = {}
            sources = {}
            for action in ACTION_ORDER:
                source = unique_action_video(root, scene, action, variant)
                relative = Path("media") / "action" / task_id / label / f"{action}.mp4"
                safe_link(source, public_root / relative)
                media[label][action] = relative.as_posix()
                sources[action] = str(source)
            private_items[label] = {"variant": variant, "source_paths": sources}
        tasks.append(
            {
                "id": task_id,
                "kind": "action",
                "baseline": "HY-WorldPlay",
                "scene_index": scene_index,
                "scene_label": f"Scene {scene_index + 1:02d}",
                "labels": list(PANEL_LABELS),
                "actions": list(ACTION_ORDER),
                "media": media,
            }
        )
        private[task_id] = {
            "kind": "action",
            "baseline": "HY-WorldPlay",
            "scene_index": scene_index,
            "scene_source_id": scene.name,
            "items": private_items,
        }
    return tasks, private


def copy_app(public_root: Path) -> None:
    app_root = Path(__file__).with_name("review_app")
    for filename in APP_FILES:
        source = app_root / filename
        if not source.is_file():
            raise FileNotFoundError(f"review app asset is missing: {source}")
        shutil.copy2(source, public_root / filename)


def write_package(args: argparse.Namespace) -> dict:
    output_root = args.output_root.expanduser().resolve()
    public_root = output_root / "public"
    public_manifest_path = public_root / "review_manifest.json"
    private_mapping_path = output_root / "private_mapping.json"
    if public_manifest_path.exists() or private_mapping_path.exists():
        raise FileExistsError(
            f"review package already exists at {output_root}; choose a new output root"
        )
    public_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    used: set[str] = set()
    public_tasks: list[dict] = []
    private_tasks: dict[str, dict] = {}

    for label, manifest in (
        ("Causal Forcing", args.causal_manifest),
        ("LongCat Video", args.longcat_manifest),
    ):
        tasks, private = build_movie_tasks(
            baseline=label,
            manifest_path=manifest.expanduser().resolve(),
            expected_prompts=args.expected_prompts,
            rng=rng,
            used=used,
            public_root=public_root,
        )
        public_tasks.extend(tasks)
        private_tasks.update(private)

    action_tasks, private = build_action_tasks(
        root=args.hy_root.expanduser().resolve(),
        expected_scenes=args.expected_hy_scenes,
        rng=rng,
        used=used,
        public_root=public_root,
    )
    public_tasks.extend(action_tasks)
    private_tasks.update(private)
    rng.shuffle(public_tasks)

    public_manifest = {
        "schema_version": 2,
        "study": "B1 grouped anonymous catastrophe and action review",
        "task_count": len(public_tasks),
        "tasks": public_tasks,
    }
    canonical_public = json.dumps(public_manifest, sort_keys=True, separators=(",", ":"))
    manifest_sha256 = hashlib.sha256(canonical_public.encode()).hexdigest()
    public_manifest["manifest_sha256"] = manifest_sha256
    private_mapping = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "public_manifest_sha256": manifest_sha256,
        "tasks": private_tasks,
    }
    copy_app(public_root)
    public_manifest_path.write_text(
        json.dumps(public_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    private_mapping_path.write_text(
        json.dumps(private_mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": 2,
        "public_root": str(public_root),
        "private_mapping": str(private_mapping_path),
        "manifest_sha256": manifest_sha256,
        "catastrophe_tasks": sum(t["kind"] == "catastrophe" for t in public_tasks),
        "action_tasks": sum(t["kind"] == "action" for t in public_tasks),
        "total_tasks": len(public_tasks),
        "catastrophe_panels": sum(t["kind"] == "catastrophe" for t in public_tasks),
        "action_panels": sum(t["kind"] == "action" for t in public_tasks),
        "total_panels": len(public_tasks),
    }
    (output_root / "package_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-manifest", type=Path, required=True)
    parser.add_argument("--longcat-manifest", type=Path, required=True)
    parser.add_argument("--hy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int, default=32)
    parser.add_argument("--expected-hy-scenes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.expected_prompts <= 0 or args.expected_hy_scenes <= 0:
        parser.error("expected counts must be positive")
    print(json.dumps(write_package(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
