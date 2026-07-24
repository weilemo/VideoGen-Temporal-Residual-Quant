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


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


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


def focusedforcing(kv_cache, q, k, v, grid_sizes, freqs, current_start, cache_start, sink_recache_after_switch, meta):
    num_frame_per_block = meta["num_frame_per_block"]
    frame_seqlen = meta["frame_seqlen"]
    sink_size = meta["sink_size"]
    local_attn_size = meta["local_attn_size"]
    max_attention_size = local_attn_size * frame_seqlen if local_attn_size != -1 else 21 * frame_seqlen
    video_index = meta["video_index"]
    chunk_index = meta["chunk_index"]
    step_index = meta["step_index"]
    block_index = meta["block_index"]

    current_start_frame = current_start // frame_seqlen
    roped_query = causal_rope_apply(
        q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
    roped_key = causal_rope_apply(
        k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

    current_end = current_start + roped_query.shape[1]
    sink_tokens = sink_size * frame_seqlen
    kv_cache_size = kv_cache["k"].shape[1]
    num_new_tokens = roped_query.shape[1]

    # Compute cache update parameters without modifying kv_cache directly
    cache_update_info = None
    is_recompute = current_end <= kv_cache["global_end_index"] and current_start > 0
    if local_attn_size != -1 and (current_end > kv_cache["global_end_index"]) and (num_new_tokens + kv_cache["local_end_index"] > kv_cache_size):
        # Calculate the number of new tokens added in this step
        # Shift existing cache content left to discard oldest tokens
        num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"] - kv_cache_size
        num_rolled_tokens = kv_cache["local_end_index"] - num_evicted_tokens - sink_tokens

        # Compute updated local indices
        local_end_index = kv_cache["local_end_index"] + current_end - kv_cache["global_end_index"] - num_evicted_tokens
        local_start_index = local_end_index - num_new_tokens

        # Construct full k, v for attention computation (without modifying the original cache)
        # Create temporary k, v for computation
        temp_k = kv_cache["k"].clone()
        temp_v = kv_cache["v"].clone()
        
        # Apply rolling update to the temporary cache
        temp_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = temp_k[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        temp_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = temp_v[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        
        # Insert new key/value into the temporary cache
        # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
        write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
        roped_offset = max(0, write_start_index - local_start_index)
        write_len = max(0, local_end_index - write_start_index)
        if write_len > 0:
            temp_k[:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset + write_len]
            temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

        # Save cache update info for later use
        cache_update_info = {
            "action": "roll_and_insert",
            "sink_tokens": sink_tokens,
            "num_rolled_tokens": num_rolled_tokens,
            "num_evicted_tokens": num_evicted_tokens,
            "local_start_index": local_start_index,
            "local_end_index": local_end_index,
            "write_start_index": write_start_index,
            "write_end_index": local_end_index,
            "new_k": roped_key[:, roped_offset:roped_offset + write_len],
            "new_v": v[:, roped_offset:roped_offset + write_len],
            "current_end": current_end,
            "is_recompute": is_recompute
        }

    else:
        # Assign new keys/values directly up to current_end
        local_end_index = kv_cache["local_end_index"] + current_end - kv_cache["global_end_index"]
        local_start_index = local_end_index - num_new_tokens

        # Construct full k, v for attention computation (without modifying the original cache)
        temp_k = kv_cache["k"].clone()
        temp_v = kv_cache["v"].clone()
        # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
        write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
        if sink_recache_after_switch:
            write_start_index = local_start_index
        roped_offset = max(0, write_start_index - local_start_index)
        write_len = max(0, local_end_index - write_start_index)
        if write_len > 0:
            temp_k[:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset + write_len]
            temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

        # Save cache update info for later use
        cache_update_info = {
            "action": "direct_insert",
            "local_start_index": local_start_index,
            "local_end_index": local_end_index,
            "write_start_index": write_start_index,
            "write_end_index": local_end_index,
            "new_k": roped_key[:, roped_offset:roped_offset + write_len],
            "new_v": v[:, roped_offset:roped_offset + write_len],
            "current_end": current_end,
            "is_recompute": is_recompute
        }

    # Use temporary k, v to compute attention
    if sink_tokens > 0:
        # Concatenate sink tokens and local window tokens, keeping total length strictly below max_attention_size
        local_budget = max_attention_size - sink_tokens
        k_sink = temp_k[:, :sink_tokens]
        v_sink = temp_v[:, :sink_tokens]

        if local_budget > 0:
            local_start_for_window = max(sink_tokens, local_end_index - local_budget)
            k_local = temp_k[:, local_start_for_window:local_end_index]
            v_local = temp_v[:, local_start_for_window:local_end_index]
            k_cat = concat_cuda(k_sink, k_local)
            v_cat = concat_cuda(v_sink, v_local)
        else:
            k_cat = k_sink
            v_cat = v_sink
        
        if chunk_index >= 6:
            attn_score = compute_attn_scores_cuda(roped_query, k_cat)
            key_diversity = compute_key_diversity_cuda(k_cat)
            attn_weight = meta['attn_weight']
            scores = attn_weight * attn_score + (1 - attn_weight) * key_diversity
            kv_rows, kv_budget = select_kv_row_indices_cuda(scores.squeeze(0), meta[block_index]["kv_budget"], True)
            x = attention_with_selected_frames(roped_query, k_cat, v_cat, kv_rows, kv_budget)

        else:
            x = attention(roped_query, k_cat, v_cat)
    else:
        window_start = max(0, local_end_index - max_attention_size)
        x = attention(roped_query, temp_k[:, window_start:local_end_index], temp_v[:, window_start:local_end_index])

    if step_index == 0 and block_index == 0:
        pass

    return x, (current_end, local_end_index, cache_update_info)


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
