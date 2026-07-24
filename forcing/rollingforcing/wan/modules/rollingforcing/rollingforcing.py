import math
import torch
from wan.modules.attention import attention
from .utils import causal_rope_apply
from trq.backends.rolling_forcing import (
    cache_size,
    evict_cache_pair,
    quantize_cache_pair,
    read_cache,
    write_cache,
)

def rollingforcing(kv_cache, q, k, v, grid_sizes, freqs, current_start, cache_start, updating_cache, meta):
    num_frame_per_block = meta["num_frame_per_block"]
    frame_seqlen = meta["frame_seqlen"]
    sink_size = meta["sink_size"]
    local_attn_size = meta["local_attn_size"]
    video_index = meta["video_index"]
    start_chunk_index = meta["start_chunk_index"]
    step_index = meta["step_index"]
    block_index = meta["block_index"]

    block_length = num_frame_per_block * frame_seqlen
    max_attention_size = local_attn_size * frame_seqlen if local_attn_size != -1 else 21 * frame_seqlen
    current_start_frame = current_start // frame_seqlen
    roped_query = causal_rope_apply(
        q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)   # [B, L, 12, 128]
    roped_key = causal_rope_apply(
        k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)   # [B, L, 12, 128]
    
    grid_sizes_one_block = grid_sizes.clone()
    grid_sizes_one_block[:,0] = 3

    # only caching the first block
    cache_end = cache_start + block_length
    num_new_tokens = cache_end - kv_cache["global_end_index"].item()
    kv_cache_size = kv_cache["k"].shape[1]

    sink_tokens = 1 * block_length # we keep the first block in the cache

    if (num_new_tokens > 0) and (
            num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
        num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
        num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
        kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
            kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        
        local_end_index = kv_cache["local_end_index"].item() + cache_end - \
            kv_cache["global_end_index"].item() - num_evicted_tokens
        local_start_index = local_end_index - block_length
        kv_cache["k"][:, local_start_index:local_end_index] = roped_key[:, :block_length]
        kv_cache["v"][:, local_start_index:local_end_index] = v[:, :block_length]
    else:
        local_end_index = kv_cache["local_end_index"].item() + cache_end - kv_cache["global_end_index"].item()
        local_start_index = local_end_index - block_length
        if local_start_index == 0: # first block is not roped in the cache
            kv_cache["k"][:, local_start_index:local_end_index] = k[:, :block_length]
        else:
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key[:, :block_length]

        kv_cache["v"][:, local_start_index:local_end_index] = v[:, :block_length]

    if num_new_tokens > 0: # prevent updating when caching clean frame
        kv_cache["global_end_index"].fill_(cache_end)
    if num_new_tokens > 0:
        quantize_cache_pair(
            kv_cache,
            local_start_index,
            local_end_index,
            meta=meta,
            layer_idx=int(meta.get("block_index", 0)),
        )
        kv_cache["local_end_index"].fill_(local_end_index)

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
                working_cache_key[:,:block_length] = causal_rope_apply(
                    working_cache_key[:,:block_length], grid_sizes_one_block, freqs, start_frame=0).type_as(v)

            x = attention(
                roped_query,
                working_cache_key,
                working_cache_v
            )

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

            anchor_cache_key = causal_rope_apply(
                kv_cache["k"][:, :block_length], grid_sizes_one_block, freqs, start_frame=rope_start_frame).type_as(v)
            anchor_cache_v = kv_cache["v"][:, :block_length]

            # 3. attention with working cache and anchor cache
            input_key = torch.cat([
                anchor_cache_key,
                working_cache_key,
                roped_key
            ], dim=1)

            input_v = torch.cat([
                anchor_cache_v,
                working_cache_v,
                v
            ], dim=1)

            x = attention(
                roped_query,
                input_key,
                input_v
            )

    return x