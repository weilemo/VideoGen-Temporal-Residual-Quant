"""Validated, layer-at-a-time readers for raw KV-cache experiment dumps."""

from __future__ import annotations

import glob
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from ..kv_cache import ChunkState


LayerRecord = tuple[int, torch.Tensor, torch.Tensor, dict[str, Any]]


def resolve_dump_paths(path_spec: str | Path | Iterable[str | Path]) -> list[Path]:
    """Expand comma-separated paths and glob patterns into unique dump files.

    Glob matches are sorted, while comma-separated groups retain their order.
    Every token must match at least one regular file so a misspelled experiment
    input cannot silently produce an empty report.
    """
    if isinstance(path_spec, (str, Path)):
        raw_specs = [path_spec]
    else:
        raw_specs = list(path_spec)

    tokens: list[str] = []
    for raw_spec in raw_specs:
        tokens.extend(token.strip() for token in str(raw_spec).split(",") if token.strip())
    if not tokens:
        raise ValueError("dump path specification is empty")

    result: list[Path] = []
    seen: set[Path] = set()
    for token in tokens:
        expanded = str(Path(token).expanduser())
        matches = sorted(Path(match) for match in glob.glob(expanded, recursive=True))
        files = [match for match in matches if match.is_file()]
        if not files:
            raise FileNotFoundError(f"dump path pattern matched no files: {token}")
        for path in files:
            normalized = path.resolve()
            if normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
    return result


def parse_layer_spec(
    layer_spec: str | int | Sequence[int] | None,
    num_layers: int | Sequence[int],
) -> list[int]:
    """Parse ``all``, comma-separated indices, and inclusive ranges.

    Examples: ``all``, ``0,3,7`` and ``0-3,8-10``. Repeated indices are
    removed without changing the requested order. ``num_layers`` may be an
    integer count or an explicit sequence of available layer indices.
    """
    if isinstance(num_layers, int):
        if num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {num_layers}")
        available = tuple(range(num_layers))
    else:
        available = tuple(int(index) for index in num_layers)
        if len(set(available)) != len(available):
            raise ValueError("available layer indices contain duplicates")
    available_set = set(available)

    if layer_spec is None or (isinstance(layer_spec, str) and layer_spec.strip().lower() == "all"):
        return list(available)
    if isinstance(layer_spec, int):
        requested = [layer_spec]
    elif isinstance(layer_spec, str):
        requested = []
        spec = layer_spec.strip()
        if not spec:
            raise ValueError("layer specification is empty")
        for token in spec.split(","):
            token = token.strip()
            if not token:
                raise ValueError(f"invalid empty entry in layer specification: {layer_spec!r}")
            range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
            if range_match:
                start, end = (int(value) for value in range_match.groups())
                if end < start:
                    raise ValueError(f"layer range must be ascending: {token}")
                requested.extend(range(start, end + 1))
            elif re.fullmatch(r"\d+", token):
                requested.append(int(token))
            else:
                raise ValueError(f"invalid layer selector: {token!r}")
    else:
        requested = [int(index) for index in layer_spec]
        if not requested:
            raise ValueError("layer specification is empty")

    result: list[int] = []
    seen: set[int] = set()
    for index in requested:
        if index not in available_set:
            available_text = _available_layer_text(available)
            raise ValueError(f"layer {index} is unavailable; available layers: {available_text}")
        if index not in seen:
            seen.add(index)
            result.append(index)
    return result


def iter_kv_dump_layers(
    dump_path: str | Path,
    layers: str | int | Sequence[int] | None = "all",
) -> Iterator[LayerRecord]:
    """Yield validated ``(layer_idx, k, v, metadata)`` records from one dump.

    Returned K/V tensors are contiguous CPU float32 in BHSD layout. Supported
    inputs are the production ``{"kv_cache": ...}`` dump and a compact tensor
    fixture format named ``hwq_kv_tensors``.
    """
    path = Path(dump_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"KV dump does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"KV dump root must be a mapping: {path}")

    if payload.get("format") == "hwq_kv_tensors":
        yield from _iter_tensor_layers(payload, path, layers)
        return
    if "kv_cache" in payload:
        yield from _iter_chunked_layers(payload, path, layers)
        return
    raise ValueError(
        f"unsupported KV dump format in {path}; expected 'kv_cache' or "
        "format='hwq_kv_tensors'"
    )


def iter_kv_dumps(
    path_spec: str | Path | Iterable[str | Path],
    layers: str | int | Sequence[int] | None = "all",
) -> Iterator[LayerRecord]:
    """Resolve one or more dump paths and stream their selected layers."""
    for path in resolve_dump_paths(path_spec):
        yield from iter_kv_dump_layers(path, layers=layers)


def _iter_tensor_layers(
    payload: Mapping[str, Any],
    path: Path,
    layers: str | int | Sequence[int] | None,
) -> Iterator[LayerRecord]:
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, Mapping) or not raw_layers:
        raise ValueError(f"hwq_kv_tensors dump has no layer mapping: {path}")

    layer_map: dict[int, Any] = {}
    for raw_index, layer in raw_layers.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid layer key {raw_index!r} in {path}") from exc
        if index < 0 or index in layer_map:
            raise ValueError(f"invalid or duplicate layer key {raw_index!r} in {path}")
        layer_map[index] = layer

    if layers is None or (isinstance(layers, str) and layers.strip().lower() == "all"):
        selected = sorted(layer_map)
    else:
        requested = _parse_unbounded_layer_spec(layers)
        selected = [index for index in requested if index in layer_map]
    dump_metadata = payload.get("metadata", {})
    if dump_metadata is None:
        dump_metadata = {}
    if not isinstance(dump_metadata, Mapping):
        raise ValueError(f"metadata must be a mapping in {path}")

    for layer_idx in selected:
        layer = layer_map[layer_idx]
        if not isinstance(layer, Mapping):
            raise ValueError(f"layer {layer_idx} must be a mapping in {path}")
        k = _validate_tensor(layer.get("k"), "k", layer_idx, path)
        v = _validate_tensor(layer.get("v"), "v", layer_idx, path)
        _validate_kv_shapes(k, v, layer_idx, path)
        k_out = k.detach().to(device="cpu", dtype=torch.float32).contiguous()
        v_out = v.detach().to(device="cpu", dtype=torch.float32).contiguous()
        metadata = _record_metadata(
            path=path,
            dump_format="hwq_kv_tensors",
            layer_idx=layer_idx,
            k=k_out,
            source_layout="BHSD",
            source_dtype=k.dtype,
            dump_metadata=dump_metadata,
        )
        yield layer_idx, k_out, v_out, metadata


def _iter_chunked_layers(
    payload: Mapping[str, Any],
    path: Path,
    layers: str | int | Sequence[int] | None,
) -> Iterator[LayerRecord]:
    cache_layers = payload.get("kv_cache")
    if not isinstance(cache_layers, (list, tuple)) or not cache_layers:
        raise ValueError(f"kv_cache must be a non-empty list in {path}")
    selected = parse_layer_spec(layers, len(cache_layers))
    dump_metadata = payload.get("metadata", {})
    if dump_metadata is None:
        dump_metadata = {}
    if not isinstance(dump_metadata, Mapping):
        raise ValueError(f"metadata must be a mapping in {path}")

    for layer_idx in selected:
        layer = cache_layers[layer_idx]
        if not isinstance(layer, Mapping) or "k" not in layer or "v" not in layer:
            raise ValueError(f"layer {layer_idx} must contain k and v caches in {path}")
        k, k_info = _read_raw_cache(layer["k"], "k", layer_idx, path)
        v, v_info = _read_raw_cache(layer["v"], "v", layer_idx, path)
        if k_info != v_info:
            raise ValueError(
                f"layer {layer_idx} K/V cache geometry or filled chunks differ in {path}: "
                f"K={k_info}, V={v_info}"
            )
        _validate_kv_shapes(k, v, layer_idx, path)

        local_end = _optional_scalar_int(layer.get("local_end_index"), "local_end_index", layer_idx, path)
        if local_end is not None and local_end != k.shape[2]:
            raise ValueError(
                f"layer {layer_idx} local_end_index={local_end} but raw cache has "
                f"{k.shape[2]} tokens in {path}"
            )
        global_end = _optional_scalar_int(
            layer.get("global_end_index"), "global_end_index", layer_idx, path
        )

        metadata = _record_metadata(
            path=path,
            dump_format="chunked_kv_cache",
            layer_idx=layer_idx,
            k=k,
            source_layout=k_info[0],
            source_dtype=torch.bfloat16,
            dump_metadata=dump_metadata,
        )
        metadata.update(
            {
                "frame_seq_length": k_info[1],
                "filled_chunks": k_info[2],
                "local_end_index": local_end,
                "global_end_index": global_end,
            }
        )
        yield layer_idx, k, v, metadata


def _read_raw_cache(
    cache: Any,
    kind: str,
    layer_idx: int,
    path: Path,
) -> tuple[torch.Tensor, tuple[str, int, int]]:
    required = ("chunks", "chunk_state", "quantized_spans", "layout", "frame_seq_length")
    missing = [name for name in required if not hasattr(cache, name)]
    if missing:
        raise ValueError(
            f"layer {layer_idx} {kind} is not a ChunkedKVCache-like object in {path}; "
            f"missing {missing}"
        )
    if cache.layout not in ("BHSD", "BSHD"):
        raise ValueError(f"layer {layer_idx} {kind} has invalid layout {cache.layout!r} in {path}")
    if cache.quantized_spans:
        raise ValueError(
            f"layer {layer_idx} {kind} contains quantized_spans; experiments require raw BF16 in {path}"
        )
    if len(cache.chunks) != len(cache.chunk_state):
        raise ValueError(f"layer {layer_idx} {kind} chunk arrays have different lengths in {path}")

    states = [int(state) for state in cache.chunk_state]
    filled_indices = [index for index, state in enumerate(states) if state != int(ChunkState.EMPTY)]
    if not filled_indices:
        raise ValueError(f"layer {layer_idx} {kind} has no filled chunks in {path}")
    expected_indices = list(range(len(filled_indices)))
    if filled_indices != expected_indices:
        raise ValueError(
            f"layer {layer_idx} {kind} filled chunks are not a contiguous prefix in {path}: "
            f"{filled_indices}"
        )

    parts: list[torch.Tensor] = []
    for index, (state, chunk) in enumerate(zip(states, cache.chunks)):
        if index < len(filled_indices):
            if state != int(ChunkState.BF16):
                raise ValueError(
                    f"layer {layer_idx} {kind} chunk {index} is not raw BF16 state in {path}"
                )
            tensor = _validate_tensor(chunk, kind, layer_idx, path, chunk_idx=index)
            if tensor.dtype != torch.bfloat16:
                raise ValueError(
                    f"layer {layer_idx} {kind} chunk {index} has dtype {tensor.dtype}; "
                    f"expected torch.bfloat16 in {path}"
                )
            parts.append(tensor)
        elif state != int(ChunkState.EMPTY) or chunk is not None:
            raise ValueError(
                f"layer {layer_idx} {kind} chunk {index} is not an empty suffix entry in {path}"
            )

    sequence_dim = 2 if cache.layout == "BHSD" else 1
    raw = torch.cat(parts, dim=sequence_dim)
    if cache.layout == "BSHD":
        raw = raw.permute(0, 2, 1, 3)
    output = raw.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return output, (cache.layout, int(cache.frame_seq_length), len(filled_indices))


def _validate_tensor(
    tensor: Any,
    kind: str,
    layer_idx: int,
    path: Path,
    *,
    chunk_idx: int | None = None,
) -> torch.Tensor:
    location = f" chunk {chunk_idx}" if chunk_idx is not None else ""
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"layer {layer_idx} {kind}{location} is not a tensor in {path}")
    if tensor.ndim != 4:
        raise ValueError(
            f"layer {layer_idx} {kind}{location} must be rank-4, got {tuple(tensor.shape)} in {path}"
        )
    if not tensor.is_floating_point():
        raise ValueError(
            f"layer {layer_idx} {kind}{location} must be floating point, got {tensor.dtype} in {path}"
        )
    return tensor


def _validate_kv_shapes(k: torch.Tensor, v: torch.Tensor, layer_idx: int, path: Path) -> None:
    if k.shape != v.shape:
        raise ValueError(
            f"layer {layer_idx} K/V shapes differ in {path}: K={tuple(k.shape)}, V={tuple(v.shape)}"
        )
    if any(size <= 0 for size in k.shape):
        raise ValueError(f"layer {layer_idx} K/V contains an empty dimension in {path}: {tuple(k.shape)}")


def _optional_scalar_int(value: Any, name: str, layer_idx: int, path: Path) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"layer {layer_idx} {name} must be scalar in {path}")
        value = value.item()
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"layer {layer_idx} {name} must be an integer in {path}") from exc
    return result


def _record_metadata(
    *,
    path: Path,
    dump_format: str,
    layer_idx: int,
    k: torch.Tensor,
    source_layout: str,
    source_dtype: torch.dtype,
    dump_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "dump_path": str(path),
        "dump_format": dump_format,
        "layer_idx": layer_idx,
        "layout": "BHSD",
        "source_layout": source_layout,
        "source_dtype": str(source_dtype),
        "num_tokens": int(k.shape[2]),
        "num_heads": int(k.shape[1]),
        "head_dim": int(k.shape[3]),
        "dump_metadata": dict(dump_metadata),
    }


def _available_layer_text(available: Sequence[int]) -> str:
    if not available:
        return "none"
    if tuple(available) == tuple(range(available[-1] + 1)):
        return f"0-{available[-1]}"
    return ",".join(str(index) for index in available)


def _parse_unbounded_layer_spec(layer_spec: str | int | Sequence[int]) -> list[int]:
    if isinstance(layer_spec, int):
        return [layer_spec]
    if not isinstance(layer_spec, str):
        return list(dict.fromkeys(int(index) for index in layer_spec))
    requested: list[int] = []
    for token in layer_spec.split(","):
        token = token.strip()
        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if range_match:
            start, end = (int(value) for value in range_match.groups())
            if end < start:
                raise ValueError(f"layer range must be ascending: {token}")
            requested.extend(range(start, end + 1))
        elif re.fullmatch(r"\d+", token):
            requested.append(int(token))
        else:
            raise ValueError(f"invalid layer selector: {token!r}")
    return list(dict.fromkeys(requested))


__all__ = [
    "LayerRecord",
    "iter_kv_dump_layers",
    "iter_kv_dumps",
    "parse_layer_spec",
    "resolve_dump_paths",
]
