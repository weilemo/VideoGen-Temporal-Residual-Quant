"""Optional first-quantization-event snapshots for cross-repository parity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .codec_parity import state_metadata


def parse_capture_layers(value: str, num_layers: int) -> set[int]:
    value = value.strip().lower()
    if value == "all":
        return set(range(num_layers))
    result: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"capture layer range must be ascending: {token}")
            result.update(range(start, end + 1))
        else:
            result.add(int(token))
    invalid = sorted(index for index in result if not 0 <= index < num_layers)
    if invalid:
        raise ValueError(f"capture layers outside [0, {num_layers - 1}]: {invalid}")
    if not result:
        raise ValueError("TRQ_PARITY_CAPTURE_LAYERS selected no layers")
    return result


def save_online_snapshot(
    output_dir: str | Path,
    *,
    layer_idx: int,
    raw_k: torch.Tensor,
    raw_v: torch.Tensor,
    decoded_k: torch.Tensor,
    decoded_v: torch.Tensor,
    encoded_k: Any,
    encoded_v: Any,
    tokens_start: int,
    tokens_end: int,
    max_tokens: int,
    quant_config: Any,
    rank: int = 0,
) -> Path:
    """Save one layer in the raw-dump schema plus TRQ reconstruction fields."""
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"first_event_rank{rank}_layer{layer_idx:02d}.pt"
    payload = {
        "format": "hwq_kv_tensors",
        "version": 1,
        "metadata": {
            "purpose": "identity_codec_first_event_parity",
            "implementation": "trq",
            "event_index": 0,
            "rank": int(rank),
            "layer_idx": int(layer_idx),
            "tokens_start": int(tokens_start),
            "tokens_end": int(tokens_end),
            "max_tokens": int(max_tokens),
            "quant_config": _config_mapping(quant_config),
            "k_state": state_metadata(encoded_k) if isinstance(encoded_k, dict) else None,
            "v_state": state_metadata(encoded_v) if isinstance(encoded_v, dict) else None,
        },
        "layers": {
            int(layer_idx): {
                "k": _cpu_bf16(raw_k),
                "v": _cpu_bf16(raw_v),
                "trq_decoded_k": _cpu_bf16(decoded_k),
                "trq_decoded_v": _cpu_bf16(decoded_v),
            }
        },
    }
    torch.save(payload, target)
    return target


def _cpu_bf16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()


def _config_mapping(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        items = config.items()
    elif hasattr(config, "items"):
        items = config.items()
    elif hasattr(config, "__dict__"):
        items = vars(config).items()
    else:
        return {"repr": repr(config)}
    return {str(key): _safe_scalar(value) for key, value in items}


def _safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, (list, tuple)):
        return [_safe_scalar(item) for item in value]
    return repr(value)
