from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .compress import compress_kv_cache, get_quantize_fn


@dataclass(frozen=True)
class HeadGroup:
    name: str
    head_ids: tuple[int, ...]
    quant_type: str


@dataclass(frozen=True)
class RandomHeadPolicy:
    num_heads: int
    num_high_precision_heads: int
    high_precision_quant_type: str
    low_precision_quant_type: str
    seed: int = 0

    def groups(self) -> tuple[HeadGroup, HeadGroup]:
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.num_high_precision_heads <= 0:
            raise ValueError("num_high_precision_heads must be positive")
        if self.num_high_precision_heads >= self.num_heads:
            raise ValueError("num_high_precision_heads must be smaller than num_heads")
        if self.high_precision_quant_type in ("", "none"):
            raise ValueError("high_precision_quant_type must be a real quantization type")
        if self.low_precision_quant_type in ("", "none"):
            raise ValueError("low_precision_quant_type must be a real quantization type")

        rng = np.random.default_rng(self.seed)
        high_heads = tuple(sorted(rng.choice(self.num_heads, size=self.num_high_precision_heads, replace=False).tolist()))
        high_set = set(high_heads)
        low_heads = tuple(head for head in range(self.num_heads) if head not in high_set)

        return (
            HeadGroup("high", high_heads, self.high_precision_quant_type),
            HeadGroup("low", low_heads, self.low_precision_quant_type),
        )


@dataclass(frozen=True)
class TopKHeadPolicy:
    num_heads: int
    high_heads_by_layer: dict[int, tuple[int, ...]]
    high_precision_quant_type: str
    low_precision_quant_type: str
    source_path: str = ""
    score_direction: str = "higher"

    def groups(self, layer_idx: int | None = None) -> tuple[HeadGroup, HeadGroup]:
        if layer_idx is None:
            raise ValueError("TopKHeadPolicy requires layer_idx")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.high_precision_quant_type in ("", "none"):
            raise ValueError("high_precision_quant_type must be a real quantization type")
        if self.low_precision_quant_type in ("", "none"):
            raise ValueError("low_precision_quant_type must be a real quantization type")

        if layer_idx not in self.high_heads_by_layer:
            available = sorted(self.high_heads_by_layer)
            raise ValueError(f"No top-k head policy for layer {layer_idx}; available layers: {available}")

        high_heads = tuple(sorted(int(head) for head in self.high_heads_by_layer[layer_idx]))
        if not high_heads:
            raise ValueError(f"Layer {layer_idx} has no high-precision heads")
        if any(head < 0 or head >= self.num_heads for head in high_heads):
            raise ValueError(f"Layer {layer_idx} has head ids outside [0, {self.num_heads})")
        if len(set(high_heads)) != len(high_heads):
            raise ValueError(f"Layer {layer_idx} has duplicate high-precision heads: {high_heads}")
        if len(high_heads) >= self.num_heads:
            raise ValueError("number of high-precision heads must be smaller than num_heads")

        high_set = set(high_heads)
        low_heads = tuple(head for head in range(self.num_heads) if head not in high_set)
        return (
            HeadGroup("high", high_heads, self.high_precision_quant_type),
            HeadGroup("low", low_heads, self.low_precision_quant_type),
        )


def _coerce_layer_head_mapping(raw_mapping) -> dict[int, tuple[int, ...]]:
    mapping = {}
    for layer_id, head_ids in raw_mapping.items():
        mapping[int(layer_id)] = tuple(int(head_id) for head_id in head_ids)
    return mapping


def _scores_to_top_heads(scores_by_layer, num_heads: int, num_high_precision_heads: int, score_direction: str):
    if num_high_precision_heads <= 0:
        raise ValueError("num_high_precision_heads must be positive")
    if num_high_precision_heads >= num_heads:
        raise ValueError("num_high_precision_heads must be smaller than num_heads")
    if score_direction not in ("higher", "lower"):
        raise ValueError("score_direction must be 'higher' or 'lower'")

    high_heads_by_layer = {}
    for layer_id, scores in scores_by_layer.items():
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (num_heads,):
            raise ValueError(f"Layer {layer_id} scores must have shape ({num_heads},), got {scores.shape}")
        order = np.argsort(scores)
        if score_direction == "higher":
            selected = order[-num_high_precision_heads:]
        else:
            selected = order[:num_high_precision_heads]
        high_heads_by_layer[int(layer_id)] = tuple(sorted(int(head) for head in selected.tolist()))
    return high_heads_by_layer


def _load_head_policy_json(path: Path, num_heads: int, num_high_precision_heads: int):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    file_num_heads = int(data.get("num_heads", num_heads))
    if file_num_heads != num_heads:
        raise ValueError(f"Policy file num_heads={file_num_heads} does not match runtime num_heads={num_heads}")

    score_direction = data.get("score_direction", "higher")
    if "top_heads_by_layer" in data:
        high_heads_by_layer = _coerce_layer_head_mapping(data["top_heads_by_layer"])
        return high_heads_by_layer, score_direction
    if "high_heads_by_layer" in data:
        high_heads_by_layer = _coerce_layer_head_mapping(data["high_heads_by_layer"])
        return high_heads_by_layer, score_direction

    if "scores_by_layer" in data:
        scores_by_layer = {int(layer): scores for layer, scores in data["scores_by_layer"].items()}
    elif "scores" in data:
        scores = data["scores"]
        if scores and isinstance(scores[0], list):
            scores_by_layer = {layer: layer_scores for layer, layer_scores in enumerate(scores)}
        else:
            flat = np.asarray(scores, dtype=np.float64)
            if flat.size % num_heads != 0:
                raise ValueError(f"Flat scores length {flat.size} is not divisible by num_heads={num_heads}")
            scores_by_layer = {
                layer: flat[layer * num_heads:(layer + 1) * num_heads].tolist()
                for layer in range(flat.size // num_heads)
            }
    elif "global_scores" in data:
        flat_scores = {int(global_id): float(score) for global_id, score in data["global_scores"].items()}
        scores_by_layer = {}
        for global_id, score in flat_scores.items():
            layer = global_id // num_heads
            head = global_id % num_heads
            scores_by_layer.setdefault(layer, [float("nan")] * num_heads)
            scores_by_layer[layer][head] = score
        for layer, scores in scores_by_layer.items():
            if any(np.isnan(score) for score in scores):
                raise ValueError(f"Layer {layer} has incomplete global_scores")
    else:
        raise ValueError(
            "Top-k policy JSON must contain top_heads_by_layer, high_heads_by_layer, "
            "scores_by_layer, scores, or global_scores"
        )

    return _scores_to_top_heads(scores_by_layer, num_heads, num_high_precision_heads, score_direction), score_direction


def _load_head_policy_table(path: Path, num_heads: int, num_high_precision_heads: int, score_direction: str):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.replace(",", " ").split()]
            if parts[0].lower() in ("global_head", "global_head_id", "layer"):
                continue
            if len(parts) == 2:
                global_id, score = int(parts[0]), float(parts[1])
                layer, head = divmod(global_id, num_heads)
            elif len(parts) >= 3:
                layer, head, score = int(parts[0]), int(parts[1]), float(parts[2])
            else:
                raise ValueError(f"Cannot parse importance row: {line}")
            rows.append((layer, head, score))

    scores_by_layer = {}
    for layer, head, score in rows:
        if head < 0 or head >= num_heads:
            raise ValueError(f"Head id {head} outside [0, {num_heads})")
        scores_by_layer.setdefault(layer, [float("nan")] * num_heads)
        scores_by_layer[layer][head] = score

    for layer, scores in scores_by_layer.items():
        if any(np.isnan(score) for score in scores):
            raise ValueError(f"Layer {layer} has incomplete importance table")

    return _scores_to_top_heads(scores_by_layer, num_heads, num_high_precision_heads, score_direction)


def load_topk_head_policy(
    path: str,
    *,
    num_heads: int,
    num_high_precision_heads: int,
    high_precision_quant_type: str,
    low_precision_quant_type: str,
    score_direction: str = "higher",
) -> TopKHeadPolicy:
    policy_path = Path(path).expanduser()
    if not policy_path.exists():
        raise FileNotFoundError(f"Head importance policy file does not exist: {policy_path}")

    if policy_path.suffix.lower() == ".json":
        high_heads_by_layer, loaded_direction = _load_head_policy_json(
            policy_path, num_heads, num_high_precision_heads
        )
        score_direction = loaded_direction or score_direction
    else:
        high_heads_by_layer = _load_head_policy_table(
            policy_path, num_heads, num_high_precision_heads, score_direction
        )

    return TopKHeadPolicy(
        num_heads=num_heads,
        high_heads_by_layer=high_heads_by_layer,
        high_precision_quant_type=high_precision_quant_type,
        low_precision_quant_type=low_precision_quant_type,
        source_path=str(policy_path),
        score_direction=score_direction,
    )


def clone_quant_config(base_config, quant_type: str):
    cfg = vars(base_config).copy() if not isinstance(base_config, dict) else dict(base_config)
    cfg["quant_type"] = quant_type
    return SimpleNamespace(**cfg)


def _pack_single_cache(cache, output_dtype: torch.dtype, quant_config):
    if isinstance(cache, dict):
        cache["info"] = {
            "output_dtype": output_dtype,
            "quant_config": quant_config,
        }
    return cache


def _pack_group_cache(groups: list[dict], output_dtype: torch.dtype, policy, layer_idx: int | None) -> dict:
    info = {
        "output_dtype": output_dtype,
        "num_heads": policy.num_heads,
        "headwise_mode": "topk" if isinstance(policy, TopKHeadPolicy) else "random",
    }
    if isinstance(policy, RandomHeadPolicy):
        info["headwise_seed"] = policy.seed
    if isinstance(policy, TopKHeadPolicy):
        info["layer_idx"] = layer_idx
        info["head_importance_path"] = policy.source_path
        info["score_direction"] = policy.score_direction
    return {
        "groups": groups,
        "info": info,
    }


def compress_headwise_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    base_quant_config,
    policy: RandomHeadPolicy | TopKHeadPolicy,
    layer_idx: int | None = None,
) -> tuple[dict, dict]:
    """Compress K/V tensors with a head-group mixed precision policy.

    Args:
        k: Key tensor in [B, H, S, D].
        v: Value tensor in [B, H, S, D].
        base_quant_config: Quantization config object or dict.
        policy: Random or per-layer top-k head-group policy.
        layer_idx: Required by ``TopKHeadPolicy``.

    Returns:
        Packed K/V cache dictionaries consumable by ``uncompress_kv_cache``.
    """
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("k and v must be [B, H, S, D]")
    if k.shape != v.shape:
        raise ValueError(f"k and v must have the same shape, got {k.shape} vs {v.shape}")
    if k.shape[1] != policy.num_heads:
        raise ValueError(f"policy.num_heads={policy.num_heads} does not match tensor heads={k.shape[1]}")

    k_groups = []
    v_groups = []

    try:
        groups = policy.groups(layer_idx)
    except TypeError:
        groups = policy.groups()

    for group in groups:
        if not group.head_ids:
            continue

        group_config = clone_quant_config(base_quant_config, group.quant_type)
        quantize_fn = get_quantize_fn(group_config.quant_type, group_config)
        head_index = torch.tensor(group.head_ids, device=k.device, dtype=torch.long)
        k_group = k.index_select(1, head_index)
        v_group = v.index_select(1, head_index)

        k_quant, v_quant = compress_kv_cache(
            k_group,
            v_group,
            group_config.quant_type,
            group_config,
            quantize_fn,
            layer_idx=layer_idx,
            head_ids=group.head_ids,
        )

        k_quant = _pack_single_cache(k_quant, k.dtype, group_config)
        v_quant = _pack_single_cache(v_quant, v.dtype, group_config)

        k_groups.append({
            "name": group.name,
            "head_ids": list(group.head_ids),
            "quant_type": group.quant_type,
            "quant_config": vars(group_config),
            "payload": k_quant,
        })
        v_groups.append({
            "name": group.name,
            "head_ids": list(group.head_ids),
            "quant_type": group.quant_type,
            "quant_config": vars(group_config),
            "payload": v_quant,
        })

    return _pack_group_cache(k_groups, k.dtype, policy, layer_idx), _pack_group_cache(v_groups, v.dtype, policy, layer_idx)
