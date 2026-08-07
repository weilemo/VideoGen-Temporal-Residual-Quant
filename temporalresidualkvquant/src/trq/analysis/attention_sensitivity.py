"""Memory-bounded attention diagnostics around an online quantization event."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch


def sampled_attention_metrics(
    query: torch.Tensor,
    reference_k: torch.Tensor,
    reference_v: torch.Tensor,
    candidate_k: torch.Tensor,
    candidate_v: torch.Tensor,
    *,
    max_query_tokens: int = 64,
    max_key_tokens: int = 256,
    topk: int = 8,
) -> dict[str, float | int]:
    """Compare attention on deterministic position samples.

    Inputs use Self-Forcing's ``[B, S, H, D]`` layout. Sampling keeps the
    diagnostic bounded while preserving all heads and the full head dimension.
    """
    tensors = (query, reference_k, reference_v, candidate_k, candidate_v)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("attention diagnostic tensors must use [B, S, H, D]")
    if reference_k.shape != candidate_k.shape or reference_v.shape != candidate_v.shape:
        raise ValueError("reference and candidate K/V shapes must match")
    if reference_k.shape[:3] != reference_v.shape[:3]:
        raise ValueError("K/V batch, sequence, and head dimensions must match")
    if query.shape[0] != reference_k.shape[0] or query.shape[2:] != reference_k.shape[2:]:
        raise ValueError("query batch/head dimensions must match K")
    if max_query_tokens <= 0 or max_key_tokens <= 0 or topk <= 0:
        raise ValueError("sampling limits and topk must be positive")

    q_idx = _sample_indices(query.shape[1], max_query_tokens, query.device)
    k_idx = _sample_indices(reference_k.shape[1], max_key_tokens, reference_k.device)
    q = query.index_select(1, q_idx).float().permute(0, 2, 1, 3)
    rk = reference_k.index_select(1, k_idx).float().permute(0, 2, 1, 3)
    ck = candidate_k.index_select(1, k_idx).float().permute(0, 2, 1, 3)
    rv = reference_v.index_select(1, k_idx).float().permute(0, 2, 1, 3)
    cv = candidate_v.index_select(1, k_idx).float().permute(0, 2, 1, 3)

    scale = 1.0 / math.sqrt(q.shape[-1])
    reference_logits = torch.matmul(q, rk.transpose(-1, -2)) * scale
    candidate_logits = torch.matmul(q, ck.transpose(-1, -2)) * scale
    reference_prob = torch.softmax(reference_logits, dim=-1)
    candidate_prob = torch.softmax(candidate_logits, dim=-1)
    reference_output = torch.matmul(reference_prob, rv)
    candidate_output = torch.matmul(candidate_prob, cv)

    resolved_topk = min(int(topk), reference_logits.shape[-1])
    ref_top = torch.topk(reference_logits, resolved_topk, dim=-1).indices
    cand_top = torch.topk(candidate_logits, resolved_topk, dim=-1).indices
    topk_overlap = (ref_top.unsqueeze(-1) == cand_top.unsqueeze(-2)).any(dim=-1).float().mean()
    eps = torch.finfo(reference_prob.dtype).eps
    kl = (
        reference_prob
        * (reference_prob.clamp_min(eps).log() - candidate_prob.clamp_min(eps).log())
    ).sum(dim=-1).mean()

    return {
        "sampled_query_tokens": int(q_idx.numel()),
        "sampled_key_tokens": int(k_idx.numel()),
        "topk": resolved_topk,
        "k_cache_read_rel_l2": _rel_l2(reference_k, candidate_k),
        "v_cache_read_rel_l2": _rel_l2(reference_v, candidate_v),
        "attention_logits_rel_l2": _rel_l2(reference_logits, candidate_logits),
        "attention_logits_cosine": _cosine(reference_logits, candidate_logits),
        "softmax_kl": float(kl.item()),
        "attention_top1_agreement": float(
            (ref_top[..., 0] == cand_top[..., 0]).float().mean().item()
        ),
        "attention_topk_overlap": float(topk_overlap.item()),
        "attention_output_rel_l2": _rel_l2(reference_output, candidate_output),
    }


def write_attention_trace(
    output_dir: str | Path,
    *,
    layer_idx: int,
    boundary_frame: int,
    metrics: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Path:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"boundary{boundary_frame:04d}_layer{layer_idx:02d}.json"
    payload = {
        "schema_version": 1,
        "layer_idx": int(layer_idx),
        "boundary_frame": int(boundary_frame),
        "sampling": "deterministic_even_positions",
        "metrics": metrics,
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _sample_indices(length: int, limit: int, device: torch.device) -> torch.Tensor:
    count = min(int(length), int(limit))
    if count == length:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, steps=count, device=device).round().long().unique()


def _rel_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    ref = reference.double()
    other = candidate.double()
    denominator = torch.linalg.vector_norm(ref).clamp_min(1e-12)
    return float((torch.linalg.vector_norm(ref - other) / denominator).item())


def _cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    ref = reference.reshape(-1).double()
    other = candidate.reshape(-1).double()
    denominator = (
        torch.linalg.vector_norm(ref) * torch.linalg.vector_norm(other)
    ).clamp_min(1e-12)
    return float((torch.dot(ref, other) / denominator).item())
