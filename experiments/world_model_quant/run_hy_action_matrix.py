#!/usr/bin/env python3
"""Run a scene/action/precision matrix through the HY-WorldPlay adapter."""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from integrations.hy_worldplay import runner  # noqa: E402


DEFAULT_ACTIONS = ("turn_left", "turn_right", "forward", "backward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=runner.HERE / "config.json")
    parser.add_argument("--actions", type=Path, default=runner.HERE / "actions.json")
    parser.add_argument("--experiment-id", default="action_control")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--minimum-scenes", type=int, default=1)
    parser.add_argument("--action", action="append", choices=sorted(DEFAULT_ACTIONS + ("canonical",)))
    parser.add_argument("--variant", action="append", choices=sorted(runner.VARIANTS))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_scenes(path: Path) -> list[dict[str, str]]:
    payload = runner.load_json(path)
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError(f"scenes must be a non-empty list: {path}")
    normalized = []
    seen = set()
    for row in scenes:
        if not isinstance(row, dict):
            raise ValueError("each scene must be a JSON object")
        scene_id = str(row.get("id", ""))
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", scene_id):
            raise ValueError(f"invalid scene id: {scene_id!r}")
        if scene_id in seen:
            raise ValueError(f"duplicate scene id: {scene_id}")
        if not row.get("image_path") or not row.get("prompt"):
            raise ValueError(f"scene {scene_id} requires image_path and prompt")
        seen.add(scene_id)
        normalized.append({
            "id": scene_id,
            "image_path": str(row["image_path"]),
            "prompt": str(row["prompt"]),
        })
    return normalized


def main() -> None:
    args = parse_args()
    base_config = runner.load_json(args.config)
    actions = runner.load_json(args.actions)
    scenes = load_scenes(args.scenes)
    if len(scenes) < args.minimum_scenes:
        raise ValueError(
            f"full HY experiment requires at least {args.minimum_scenes} scenes; "
            f"manifest has {len(scenes)}"
        )
    if args.limit_scenes:
        scenes = scenes[: args.limit_scenes]
    action_ids = args.action or list(DEFAULT_ACTIONS)
    variants = args.variant or list(runner.VARIANTS)

    for scene in scenes:
        config = copy.deepcopy(base_config)
        config["experiment_id"] = str(Path(args.experiment_id) / scene["id"])
        config["image_path"] = scene["image_path"]
        config["prompt"] = scene["prompt"]
        for action_id in action_ids:
            for variant in variants:
                runner.run_one(config, actions, variant, action_id, args.dry_run)


if __name__ == "__main__":
    main()
