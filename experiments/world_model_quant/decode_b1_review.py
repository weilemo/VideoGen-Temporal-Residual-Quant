#!/usr/bin/env python3
"""Join anonymous B1 review exports with the private method mapping."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def flatten_export(review: dict, mapping: dict) -> list[dict]:
    if review.get("manifest_sha256") != mapping.get("public_manifest_sha256"):
        raise ValueError("review export and private mapping use different manifests")
    rows: list[dict] = []
    for task_id, answer in review.get("answers", {}).items():
        private = mapping.get("tasks", {}).get(task_id)
        if private is None:
            raise ValueError(f"unknown task id in review export: {task_id}")
        for side in ("A", "B"):
            side_answer = answer.get("sides", {}).get(side, {})
            row = {
                "task_id": task_id,
                "kind": private["kind"],
                "baseline": private["baseline"],
                "prompt_or_scene_index": private.get("prompt_index", private.get("scene_index")),
                "side": side,
                "variant": private["sides"][side]["variant"],
                "complete": bool(answer.get("complete")),
            }
            if private["kind"] == "catastrophe":
                row.update(side_answer)
                rows.append(row)
            else:
                pair = side_answer.get("pair", {})
                for action, action_answer in side_answer.get("actions", {}).items():
                    action_row = {**row, **pair, **action_answer, "action": action}
                    rows.append(action_row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--private-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = flatten_export(read_json(args.review), read_json(args.private_mapping))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
