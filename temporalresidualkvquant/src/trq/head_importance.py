from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def iter_focused_forcing_loss_files(input_path: str | Path) -> Iterable[Path]:
    """Yield focused-forcing head-ablation JSON files.

    ``input_path`` can be either a single JSON file or a directory containing
    the ``chunk_id.json`` files emitted by focused-forcing DMD-loss inference.
    """
    path = Path(input_path).expanduser()
    if path.is_file():
        yield path
        return
    yield from sorted(path.rglob("*.json"))


def load_focused_forcing_head_losses(input_path: str | Path) -> dict[int, list[float]]:
    """Load focused-forcing head-ablation DMD losses.

    The expected JSON format is the one produced by the focused-forcing
    ``inference.py`` analysis path:

    ``{"global_head_id": dmd_loss}``

    where ``global_head_id = layer_idx * num_heads + head_idx``.
    Multiple files can contain repeated ids; values are kept as a list so later
    aggregation can average across prompts/chunks/runs.
    """
    values: dict[int, list[float]] = defaultdict(list)
    for path in iter_focused_forcing_loss_files(input_path):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            try:
                global_head_id = int(key)
                score = float(value)
            except (TypeError, ValueError):
                continue
            values[global_head_id].append(score)
    return dict(values)


def mean_head_scores(head_losses: dict[int, list[float]]) -> dict[int, float]:
    scores = {}
    for global_head_id, values in sorted(head_losses.items()):
        if not values:
            continue
        scores[int(global_head_id)] = float(sum(values) / len(values))
    return scores


def reshape_global_scores_by_layer(
    global_scores: dict[int, float],
    *,
    num_layers: int,
    num_heads: int,
    allow_incomplete: bool = False,
) -> dict[int, list[float]]:
    total_heads = num_layers * num_heads
    missing = [idx for idx in range(total_heads) if idx not in global_scores]
    if missing and not allow_incomplete:
        preview = ", ".join(str(idx) for idx in missing[:12])
        raise ValueError(
            f"Missing {len(missing)} / {total_heads} heads. First missing ids: {preview}. "
            "Run a full ablation sweep or pass allow_incomplete=True."
        )

    scores_by_layer = {}
    for layer in range(num_layers):
        layer_scores = []
        complete = True
        for head in range(num_heads):
            global_id = layer * num_heads + head
            if global_id not in global_scores:
                complete = False
                break
            layer_scores.append(float(global_scores[global_id]))
        if complete:
            scores_by_layer[layer] = layer_scores
    return scores_by_layer


def select_top_heads_by_layer(
    scores_by_layer: dict[int, list[float]],
    *,
    top_k: int,
    score_direction: str = "higher",
) -> dict[int, list[int]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if score_direction not in ("higher", "lower"):
        raise ValueError("score_direction must be 'higher' or 'lower'")

    top_heads_by_layer = {}
    for layer, layer_scores in sorted(scores_by_layer.items()):
        if top_k >= len(layer_scores):
            raise ValueError(f"top_k must be smaller than num_heads for layer {layer}")
        order = sorted(range(len(layer_scores)), key=lambda head: layer_scores[head])
        selected = order[-top_k:] if score_direction == "higher" else order[:top_k]
        top_heads_by_layer[int(layer)] = sorted(int(head) for head in selected)
    return top_heads_by_layer


def build_topk_policy_from_focused_forcing(
    input_path: str | Path,
    *,
    num_layers: int = 30,
    num_heads: int = 12,
    top_k: int = 4,
    score_direction: str = "higher",
    allow_incomplete: bool = False,
) -> dict:
    """Build a top-k policy payload from focused-forcing DMD-loss outputs."""
    head_losses = load_focused_forcing_head_losses(input_path)
    if not head_losses:
        raise ValueError(f"No numeric head-loss entries found under {input_path}")

    global_scores = mean_head_scores(head_losses)
    scores_by_layer = reshape_global_scores_by_layer(
        global_scores,
        num_layers=num_layers,
        num_heads=num_heads,
        allow_incomplete=allow_incomplete,
    )
    top_heads_by_layer = select_top_heads_by_layer(
        scores_by_layer,
        top_k=top_k,
        score_direction=score_direction,
    )

    return {
        "format": "headwise-topk-policy-v1",
        "source": str(Path(input_path).expanduser()),
        "num_heads": num_heads,
        "num_layers": num_layers,
        "top_k": top_k,
        "score_direction": score_direction,
        "score_meaning": "Mean DMD loss after masking this global attention head; higher means more important.",
        "global_scores": {str(key): value for key, value in sorted(global_scores.items())},
        "scores_by_layer": {str(key): value for key, value in sorted(scores_by_layer.items())},
        "top_heads_by_layer": {str(key): value for key, value in sorted(top_heads_by_layer.items())},
    }


def write_topk_policy(policy: dict, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(policy, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path
