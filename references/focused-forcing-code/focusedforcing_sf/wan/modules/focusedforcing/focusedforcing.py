from contextlib import contextmanager
import json
import time

import flash_attn
import torch

from wan.modules.attention import attention

from .cuda_ext import (
    causal_rope_apply_cuda,
    compute_attn_scores_cuda,
    compute_key_diversity_cuda,
    pack_qkv_cuda,
    rope_apply_temporal_shift_cuda,
    select_kv_row_indices_cuda,
)


@contextmanager
def cuda_timer(name: str, enabled: bool = True):
    if not enabled:
        yield
        return
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"{name} time: {(t1 - t0) * 1000:.3f} ms")


def prepare_meta(loss_path, max_budget, min_budget, attn_weight, QF=3, T=1560, device="cuda"):
    with open(loss_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    loss_map = {int(k): float(v) for k, v in raw.items()}

    H = 12
    total = len(loss_map)
    num_blocks = total // H

    all_losses = list(loss_map.values())
    loss_min = min(all_losses)
    loss_max = max(all_losses)

    def map_budget(score, score_min, score_max, min_budget, max_budget, gamma=2.0):
        if abs(score_max - score_min) < 1e-12:
            return int(round((min_budget + max_budget) / 2))

        x = (score - score_min) / (score_max - score_min)
        y = x ** gamma
        budget = min_budget + y * (max_budget - min_budget)
        return int(round(budget))

    meta = {}
    meta.update({"attn_weight": attn_weight})

    for block_index in range(num_blocks):
        kv_budget = torch.zeros((QF, H), dtype=torch.int32, device=device)

        for h in range(H):
            idx = block_index * H + h
            loss_value = loss_map[idx]
            budget = map_budget(loss_value, loss_min, loss_max, min_budget, max_budget)

            kv_budget[:, h] = budget

        k_lens = (kv_budget.reshape(-1) * T).to(torch.int32)

        G = QF * H
        cu_seqlens_q = torch.arange(0, (G + 1) * T, T, device=device).to(torch.int32)
        cu_seqlens_k = torch.cat(
            [torch.zeros(1, device=device), k_lens.cumsum(0)]
        ).to(torch.int32)

        meta[block_index] = {
            "kv_budget": kv_budget,
            "kv_row_indices": None,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": T,
            "max_seqlen_k": int(k_lens.max().item()) if k_lens.numel() else 0,
        }
        
    return meta


def update_kv_and_k_diversity(kv_cache, freqs, current_start_frame, meta):
    num_frame_per_block = meta["num_frame_per_block"]
    frame_seqlen = meta["frame_seqlen"]
    sink_size = meta["sink_size"]
    local_attn_size = meta["local_attn_size"]
    max_attention_size = local_attn_size * frame_seqlen if local_attn_size != -1 else 21 * frame_seqlen
    max_attention_chunks = local_attn_size / num_frame_per_block if local_attn_size != -1 else 21 / num_frame_per_block
    video_index = meta["video_index"]
    chunk_index = meta["chunk_index"]
    step_index = meta["step_index"]
    block_index = meta["block_index"]
    enabled = False #(chunk_index ==  8)

    sink_tokens = sink_size * frame_seqlen
    kv_cache_size = local_attn_size * frame_seqlen
    num_new_tokens = num_frame_per_block * frame_seqlen

    num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"][0] - kv_cache_size
    num_rolled_tokens = kv_cache["local_end_index"][0] - num_evicted_tokens - sink_tokens
    num_clean_frames = local_attn_size - num_frame_per_block

    if chunk_index >= max_attention_chunks:
        with cuda_timer("update_kv", enabled=enabled):
            kv_cache["k"][:, :, sink_tokens:sink_tokens + num_rolled_tokens] = kv_cache["k"][:, :, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            kv_cache["v"][:, :, sink_tokens:sink_tokens + num_rolled_tokens] = kv_cache["v"][:, :, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

        desired_sink_start_frame = current_start_frame - kv_cache_size // frame_seqlen
        delta = desired_sink_start_frame - kv_cache["sink_start_frame"]
        if delta != 0:
            with cuda_timer("rope_apply_temporal_shift", enabled=enabled):
                kv_cache["sink_start_frame"] = desired_sink_start_frame
                rope_apply_temporal_shift_cuda(kv_cache["k"][:, :, :sink_tokens], freqs, delta)

    with cuda_timer("compute_key_diversity", enabled=enabled):
        kv_cache["key_diversity"] = compute_key_diversity_cuda(kv_cache["k"][:, :, :num_clean_frames * frame_seqlen])


def focusedforcing(kv_cache, q, k, v, grid_sizes, freqs, current_start):
    meta = kv_cache["meta"]
    num_frame_per_block = meta["num_frame_per_block"]
    frame_seqlen = meta["frame_seqlen"]
    sink_size = meta["sink_size"]
    local_attn_size = meta["local_attn_size"]
    max_attention_size = local_attn_size * frame_seqlen if local_attn_size != -1 else 21 * frame_seqlen
    max_attention_chunks = local_attn_size / num_frame_per_block if local_attn_size != -1 else 21 / num_frame_per_block
    attn_weight = meta["attn_weight"]
    video_index = meta["video_index"]
    chunk_index = meta["chunk_index"]
    step_index = meta["step_index"]
    block_index = meta["block_index"]
    enabled = False #(chunk_index == 8 and step_index == 1 and block_index == 1)

    current_start_frame = current_start // frame_seqlen
    
    with cuda_timer("causal_rope_apply", enabled=enabled):
        roped_query, roped_key = causal_rope_apply_cuda(q, k, grid_sizes, freqs, start_frame=current_start_frame)

    current_end = current_start + roped_query.shape[1]
    kv_cache_size = kv_cache["k"].shape[1]
    num_new_tokens = roped_query.shape[1]

    if local_attn_size != -1 and (current_end > kv_cache["global_end_index"]) and (num_new_tokens + kv_cache["local_end_index"] > kv_cache_size):
        num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"] - kv_cache_size
        # Insert the new keys/values at the end
        local_end_index = kv_cache["local_end_index"] + current_end - kv_cache["global_end_index"] - num_evicted_tokens
        local_start_index = local_end_index - num_new_tokens
        with cuda_timer("assign_new_keys_values", enabled=enabled):
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v

    else:
        # Assign new keys/values directly up to current_end
        local_end_index = kv_cache["local_end_index"] + current_end - kv_cache["global_end_index"]
        local_start_index = local_end_index - num_new_tokens
        with cuda_timer("assign_new_keys_values", enabled=enabled):
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v

    key = kv_cache["k"][:, max(0, local_end_index - max_attention_size):local_end_index]
    value = kv_cache["v"][:, max(0, local_end_index - max_attention_size):local_end_index]

    kv_cache["global_end_index"] = current_end
    kv_cache["local_end_index"] = local_end_index  

    if chunk_index >= max_attention_chunks - 1:
        with cuda_timer("compute_attn_scores", enabled=enabled):
            attn_scores = compute_attn_scores_cuda(roped_query, key).squeeze(0)
        key_diversity = kv_cache["key_diversity"].expand(attn_scores.size(0), -1, -1) # [1,H,18]

        score = attn_weight * attn_scores + (1 - attn_weight) * key_diversity
        kv_budget = kv_cache["meta"][block_index]["kv_budget"]

        with cuda_timer("select_kv_row_indices", enabled=enabled):
            kv_cache["meta"][block_index]["kv_row_indices"] = select_kv_row_indices_cuda(score, kv_budget)

    if chunk_index < max_attention_chunks - 1:
        x = attention(roped_query, key, value)
    else:
        with cuda_timer("attention_with_selected_frames", enabled=enabled):
            x = attention_with_selected_frames(roped_query, key, value, kv_cache["meta"][block_index])

    return x


def attention_with_selected_frames(
    roped_query,          # [1, H, QF, T, D]
    key,          # [1, H, KvF, T, D]
    value,          # [1, H, KvF, T, D]
    meta,             # prepare_selected_frames_meta_b1(...)
    softmax_scale=None,
    causal=False,
    dropout_p=0.0,
    deterministic=False,
):
    T = 1560

    B, Lq, H, D = roped_query.shape
    assert B == 1
    assert Lq % T == 0
    QF = Lq // T

    _, Lk, H2, D2 = key.shape
    assert H2 == H and D2 == D
    assert value.shape == key.shape
    assert Lk % T == 0

    cu_seqlens_q = meta["cu_seqlens_q"]
    cu_seqlens_k = meta["cu_seqlens_k"]
    max_seqlen_q = meta["max_seqlen_q"]
    max_seqlen_k = meta["max_seqlen_k"]

    G = QF * H

    q_packed, k_packed, v_packed = pack_qkv_cuda(
        roped_query, key, value, meta["kv_row_indices"]
    )

    out_packed = flash_attn.flash_attn_varlen_func(
        q=q_packed,
        k=k_packed,
        v=v_packed,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )

    out = (
        out_packed.view(G, T, D)        # [QF*H, T, D]
        .view(QF, H, T, D)              # [QF, H, T, D]
        .permute(0, 2, 1, 3)            # [QF, T, H, D]
        .reshape(B, QF * T, H, D)
        .contiguous()
    )
    return out


