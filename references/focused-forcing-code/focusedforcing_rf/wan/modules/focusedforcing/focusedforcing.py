from contextlib import contextmanager
import json
import time

import flash_attn
import torch
import torch.nn.functional as F

from wan.modules.attention import attention

from .cuda_ext import (
    causal_rope_apply_cuda,
    compute_attn_scores_cuda,
    compute_key_diversity_cuda,
    pack_qkv_cuda,
    select_kv_row_indices_cuda,
    concat_cuda,
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


def prepare_meta(loss_path, max_budget, min_budget, attn_weight, device="cuda"):
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
        kv_budget = torch.zeros((H,), dtype=torch.int32, device=device)

        for h in range(H):
            idx = block_index * H + h
            loss_value = loss_map[idx]
            budget = map_budget(loss_value, loss_min, loss_max, min_budget, max_budget)
            kv_budget[h] = budget

        meta[block_index] = {
            "kv_budget": kv_budget,   # [H]
            "kv_row_indices": None,
        }

    return meta


def focusedforcing(kv_cache, q, k, v, grid_sizes, freqs, current_start, cache_start, updating_cache, meta):
    num_frame_per_block = meta["num_frame_per_block"]
    frame_seqlen = meta["frame_seqlen"]
    sink_size = meta["sink_size"]
    local_attn_size = meta["local_attn_size"]
    block_length = num_frame_per_block * frame_seqlen
    max_attention_size = local_attn_size * frame_seqlen if local_attn_size != -1 else 21 * frame_seqlen
    video_index = meta["video_index"]
    start_chunk_index = meta["start_chunk_index"]
    step_index = meta["step_index"]
    block_index = meta["block_index"]
    enabled = False #(chunk_index == 8 and step_index == 1 and block_index == 1)

    current_start_frame = current_start // frame_seqlen

    roped_query, roped_key = causal_rope_apply_cuda(q, k, grid_sizes, freqs, start_frame=current_start_frame)

    grid_sizes_one_block = grid_sizes.clone()
    grid_sizes_one_block[:,0] = 3

    # only caching the first block
    cache_end = cache_start + block_length
    num_new_tokens = cache_end - kv_cache["global_end_index"]
    kv_cache_size = kv_cache["k"].shape[1]

    sink_tokens = sink_size * frame_seqlen # we keep the first block in the cache

    if (num_new_tokens > 0) and (
            num_new_tokens + kv_cache["local_end_index"] > kv_cache_size):
        num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"] - kv_cache_size
        num_rolled_tokens = kv_cache["local_end_index"] - num_evicted_tokens - sink_tokens
        kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        
        local_end_index = kv_cache["local_end_index"] + cache_end - \
            kv_cache["global_end_index"] - num_evicted_tokens
        local_start_index = local_end_index - block_length
        kv_cache["k"][:, local_start_index:local_end_index] = roped_key[:, :block_length]
        kv_cache["v"][:, local_start_index:local_end_index] = v[:, :block_length]
    else:
        local_end_index = kv_cache["local_end_index"] + cache_end - kv_cache["global_end_index"]
        local_start_index = local_end_index - block_length
        if local_start_index == 0: # first block is not roped in the cache
            kv_cache["k"][:, local_start_index:local_end_index] = k[:, :block_length]
        else:
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key[:, :block_length]

        kv_cache["v"][:, local_start_index:local_end_index] = v[:, :block_length]

    if num_new_tokens > 0: # prevent updating when caching clean frame
        kv_cache["global_end_index"] = cache_end
        kv_cache["local_end_index"] = local_end_index

    if local_start_index == 0:
        # no kv attn with cache
        x = attention(
            roped_query,
            roped_key,
            v)
    else:
        if updating_cache: # updating working cache with clean frame
            extract_cache_end = local_end_index
            extract_cache_start = max(0, local_end_index-max_attention_size)
            working_cache_key = kv_cache["k"][:, extract_cache_start:extract_cache_end].clone()
            working_cache_v = kv_cache["v"][:, extract_cache_start:extract_cache_end]

            if extract_cache_start == 0: # rope the global first block in working cache
                _, working_cache_key[:,:block_length] = causal_rope_apply_cuda(None, working_cache_key[:,:block_length], grid_sizes_one_block, freqs, start_frame=0)

            if start_chunk_index >= 6:
                attn_score = compute_attn_scores_cuda(roped_query, working_cache_key)
                key_diversity = compute_key_diversity_cuda(working_cache_key)
                attn_weight = meta['attn_weight']
                scores = attn_weight * attn_score + (1 - attn_weight) * key_diversity
                kv_rows, kv_budget = select_kv_row_indices_cuda(scores.squeeze(0), meta[block_index]["kv_budget"], updating_cache)
                x = attention_with_selected_frames(roped_query, working_cache_key, working_cache_v, kv_rows, kv_budget)
            else:
                x = attention(roped_query, working_cache_key, working_cache_v)

        else:
            # 1. extract working cache
            # calculate the length of working cache
            query_length = roped_query.shape[1]
            working_cache_max_length = max_attention_size - query_length - block_length

            extract_cache_end = local_start_index
            extract_cache_start = max(block_length, local_start_index - working_cache_max_length) # working cache does not include the first anchor block
            working_cache_key = kv_cache["k"][:, extract_cache_start:extract_cache_end]
            working_cache_v = kv_cache["v"][:, extract_cache_start:extract_cache_end]

            # 2. extract anchor cache, roped as the past frame
            working_cache_frame_length = working_cache_key.shape[1] // frame_seqlen
            rope_start_frame = current_start_frame - working_cache_frame_length - 3

            _, anchor_cache_key = causal_rope_apply_cuda(None, kv_cache["k"][:, :block_length], grid_sizes_one_block, freqs, start_frame=rope_start_frame)

            anchor_cache_v = kv_cache["v"][:, :block_length]

            # 3. attention with working cache and anchor cache
            input_key = concat_cuda(anchor_cache_key, working_cache_key, roped_key)
            input_v = concat_cuda(anchor_cache_v, working_cache_v, v)

            if start_chunk_index >= 2:
                scores = compute_attn_scores_cuda(roped_query, input_key)
                kv_rows, kv_budget = select_kv_row_indices_cuda(scores.squeeze(0), meta[block_index]["kv_budget"], updating_cache)
                x = attention_with_selected_frames(roped_query, input_key, input_v, kv_rows, kv_budget)
            else:
                x = attention(roped_query, input_key, input_v)

    if block_index == 0:
        pass

    return x


def attention_with_selected_frames(
    roped_query,          # [1, H, QF, T, D]
    key,          # [1, H, KvF, T, D]
    value,          # [1, H, KvF, T, D]
    kv_rows, 
    kv_budget,
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

    G = QF * H

    q_packed, k_packed, v_packed, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = pack_qkv_cuda(
        roped_query, key, value, kv_rows, kv_budget
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
