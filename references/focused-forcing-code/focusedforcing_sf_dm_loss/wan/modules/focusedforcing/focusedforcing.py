import torch
import triton
import triton.language as tl
from .causal_rope_apply_cuda.causal_rope_apply_wrapper import causal_rope_apply_cuda
from .rope_apply_temporal_shift_cuda.rope_apply_temporal_shift_wrapper import rope_apply_temporal_shift_cuda


def focusedforcing(kv_cache, q, k, v, grid_sizes, freqs, current_start, **compression_kwargs):
    sink_size = compression_kwargs["sink_size"]
    local_attn_size = compression_kwargs["local_attn_size"]
    max_attention_size = compression_kwargs["max_attention_size"]
    video_index = compression_kwargs["video_index"]
    chunk_index = compression_kwargs["chunk_index"]
    step_index = compression_kwargs["step_index"]
    block_index = compression_kwargs["block_index"]

    frame_seqlen = 1560
    current_start_frame = current_start // frame_seqlen

    roped_query, roped_key = causal_rope_apply_triton(q, k, grid_sizes, freqs, start_frame=current_start_frame)

    current_end = current_start + roped_query.shape[1]
    sink_tokens = sink_size * frame_seqlen
    kv_cache_size = kv_cache["k"].shape[1]
    num_new_tokens = roped_query.shape[1]

    if local_attn_size != -1 and (current_end > kv_cache["global_end_index"]) and (num_new_tokens + kv_cache["local_end_index"] > kv_cache_size):
        num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"] - kv_cache_size
        num_rolled_tokens = kv_cache["local_end_index"] - num_evicted_tokens - sink_tokens
        kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        # Insert the new keys/values at the end
        local_end_index = kv_cache["local_end_index"] + current_end - kv_cache["global_end_index"] - num_evicted_tokens
        local_start_index = local_end_index - num_new_tokens
        kv_cache["k"][:, local_start_index:local_end_index] = roped_key
        kv_cache["v"][:, local_start_index:local_end_index] = v

        # sink delta rotation
        desired_sink_start_frame = current_start_frame - kv_cache_size // frame_seqlen
        if "sink_start_frame" not in kv_cache:
            kv_cache["sink_start_frame"] = desired_sink_start_frame
        else:
            delta = desired_sink_start_frame - kv_cache["sink_start_frame"]
            rope_apply_temporal_shift(kv_cache["k"][:, :sink_tokens], freqs, delta)
            kv_cache["sink_start_frame"] = desired_sink_start_frame

    else:
        # Assign new keys/values directly up to current_end
        local_end_index = kv_cache["local_end_index"] + current_end - kv_cache["global_end_index"]
        local_start_index = local_end_index - num_new_tokens
        kv_cache["k"][:, local_start_index:local_end_index] = roped_key
        kv_cache["v"][:, local_start_index:local_end_index] = v
        
    key = kv_cache["k"][:, max(0, local_end_index - max_attention_size):local_end_index]
    value = kv_cache["v"][:, max(0, local_end_index - max_attention_size):local_end_index]

    kv_cache["global_end_index"] = current_end
    kv_cache["local_end_index"] = local_end_index

    return roped_query, key, value


_ROPE_FREQ_CACHE = {}


def _get_cached_rope_freqs(freqs: torch.Tensor, D: int, device: torch.device):
    c_half = D // 2
    c0 = c_half - 2 * (c_half // 3)
    c1 = c_half // 3
    c2 = c_half // 3

    key = (freqs.data_ptr(), str(device), D)
    cached = _ROPE_FREQ_CACHE.get(key, None)
    if cached is not None:
        return cached

    freqs_dev = freqs.to(device=device, dtype=torch.complex64, non_blocking=True)
    freqs0, freqs1, freqs2 = freqs_dev.split([c0, c1, c2], dim=1)

    freqs0_re = freqs0.real.contiguous().reshape(-1)
    freqs0_im = freqs0.imag.contiguous().reshape(-1)

    freqs1_re = freqs1.real.contiguous().reshape(-1)
    freqs1_im = freqs1.imag.contiguous().reshape(-1)

    freqs2_re = freqs2.real.contiguous().reshape(-1)
    freqs2_im = freqs2.imag.contiguous().reshape(-1)

    cached = (
        freqs0_re, freqs0_im,
        freqs1_re, freqs1_im,
        freqs2_re, freqs2_im,
        c0, c1, c2,
    )
    _ROPE_FREQ_CACHE[key] = cached
    return cached


@triton.jit
def causal_rope_apply_kernel(
    q_ptr, k_ptr,                     # [B, L, N, D], bf16/fp16/fp32
    grid_sizes_ptr,                   # [B, 3], int64
    freqs0_re_ptr, freqs0_im_ptr,     # [T * c0]
    freqs1_re_ptr, freqs1_im_ptr,     # [H * c1]
    freqs2_re_ptr, freqs2_im_ptr,     # [W * c2]
    out_q_ptr, out_k_ptr,             # [B, L, N, D], same dtype as q/k
    start_frame,
    B, L, N, D,
    stride_qb, stride_ql, stride_qh, stride_qd,
    stride_kb, stride_kl, stride_kh, stride_kd,
    stride_oqb, stride_oql, stride_oqh, stride_oqd,
    stride_okb, stride_okl, stride_okh, stride_okd,
    stride_gb, stride_gd,
    c0: tl.constexpr, c1: tl.constexpr, c2: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_l_blk = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    if pid_b >= B or pid_h >= N:
        return

    tok_offsets = pid_l_blk * BLOCK_L + tl.arange(0, BLOCK_L)  # [BLOCK_L]
    tok_mask = tok_offsets < L

    f = tl.load(grid_sizes_ptr + pid_b * stride_gb + 0 * stride_gd)
    gh = tl.load(grid_sizes_ptr + pid_b * stride_gb + 1 * stride_gd)
    gw = tl.load(grid_sizes_ptr + pid_b * stride_gb + 2 * stride_gd)

    seq_len = f * gh * gw
    in_seq = tok_offsets < seq_len

    hw = gh * gw
    frame_idx = tok_offsets // hw
    hw_idx = tok_offsets % hw
    h_idx = hw_idx // gw
    w_idx = hw_idx % gw

    c_half = D // 2

    for c_start in range(0, c_half, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)     # [BLOCK_C]
        c_mask = c_offs < c_half

        q_base = (
            pid_b * stride_qb
            + tok_offsets[:, None] * stride_ql
            + pid_h * stride_qh
            + (c_offs[None, :] * 2) * stride_qd
        )
        k_base = (
            pid_b * stride_kb
            + tok_offsets[:, None] * stride_kl
            + pid_h * stride_kh
            + (c_offs[None, :] * 2) * stride_kd
        )

        load_mask = tok_mask[:, None] & c_mask[None, :]

        q_re = tl.load(q_ptr + q_base,               mask=load_mask, other=0).to(tl.float32)
        q_im = tl.load(q_ptr + q_base + stride_qd,   mask=load_mask, other=0).to(tl.float32)

        k_re = tl.load(k_ptr + k_base,               mask=load_mask, other=0).to(tl.float32)
        k_im = tl.load(k_ptr + k_base + stride_kd,   mask=load_mask, other=0).to(tl.float32)

        in_c0 = c_offs < c0
        in_c1 = (c_offs >= c0) & (c_offs < c0 + c1)
        in_c2 = c_offs >= c0 + c1

        idx0 = (start_frame + frame_idx[:, None]) * c0 + c_offs[None, :]
        idx1 = h_idx[:, None] * c1 + (c_offs[None, :] - c0)
        idx2 = w_idx[:, None] * c2 + (c_offs[None, :] - c0 - c1)

        idx0 = tl.where(in_c0[None, :], idx0, 0)
        idx1 = tl.where(in_c1[None, :], idx1, 0)
        idx2 = tl.where(in_c2[None, :], idx2, 0)

        m0 = tok_mask[:, None] & in_c0[None, :]
        m1 = tok_mask[:, None] & in_c1[None, :]
        m2 = tok_mask[:, None] & in_c2[None, :]

        f0_re = tl.load(freqs0_re_ptr + idx0, mask=m0, other=1.0)
        f0_im = tl.load(freqs0_im_ptr + idx0, mask=m0, other=0.0)

        f1_re = tl.load(freqs1_re_ptr + idx1, mask=m1, other=1.0)
        f1_im = tl.load(freqs1_im_ptr + idx1, mask=m1, other=0.0)

        f2_re = tl.load(freqs2_re_ptr + idx2, mask=m2, other=1.0)
        f2_im = tl.load(freqs2_im_ptr + idx2, mask=m2, other=0.0)

        freq_re = tl.where(in_c0[None, :], f0_re, tl.where(in_c1[None, :], f1_re, f2_re))
        freq_im = tl.where(in_c0[None, :], f0_im, tl.where(in_c1[None, :], f1_im, f2_im))

        oq_re = q_re * freq_re - q_im * freq_im
        oq_im = q_re * freq_im + q_im * freq_re

        ok_re = k_re * freq_re - k_im * freq_im
        ok_im = k_re * freq_im + k_im * freq_re

        final_q_re = tl.where(in_seq[:, None], oq_re, q_re)
        final_q_im = tl.where(in_seq[:, None], oq_im, q_im)

        final_k_re = tl.where(in_seq[:, None], ok_re, k_re)
        final_k_im = tl.where(in_seq[:, None], ok_im, k_im)

        out_q_base = (
            pid_b * stride_oqb
            + tok_offsets[:, None] * stride_oql
            + pid_h * stride_oqh
            + (c_offs[None, :] * 2) * stride_oqd
        )
        out_k_base = (
            pid_b * stride_okb
            + tok_offsets[:, None] * stride_okl
            + pid_h * stride_okh
            + (c_offs[None, :] * 2) * stride_okd
        )

        tl.store(out_q_ptr + out_q_base,             final_q_re, mask=load_mask)
        tl.store(out_q_ptr + out_q_base + stride_oqd, final_q_im, mask=load_mask)

        tl.store(out_k_ptr + out_k_base,             final_k_re, mask=load_mask)
        tl.store(out_k_ptr + out_k_base + stride_okd, final_k_im, mask=load_mask)


def causal_rope_apply_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0,
):
    """
    q, k:      [B, L, N, D]  (bf16/fp16/fp32)
    grid_sizes:[B, 3] int64
    freqs:     [Tmax, D//2] complex128/complex64
    return:    roped_query, roped_key  (same dtype as q/k)
    """
    assert q.is_cuda and k.is_cuda
    assert q.shape == k.shape
    assert q.ndim == 4
    assert grid_sizes.ndim == 2 and grid_sizes.shape[1] == 3

    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()

    grid_sizes = grid_sizes.to(device=q.device, non_blocking=True).contiguous()

    B, L, N, D = q.shape
    assert D % 2 == 0

    (
        freqs0_re, freqs0_im,
        freqs1_re, freqs1_im,
        freqs2_re, freqs2_im,
        c0, c1, c2,
    ) = _get_cached_rope_freqs(freqs, D, q.device)

    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k)

    BLOCK_C = 32
    BLOCK_L = 8

    grid = (B, triton.cdiv(L, BLOCK_L), N)

    causal_rope_apply_kernel[grid](
        q, k,
        grid_sizes,
        freqs0_re, freqs0_im,
        freqs1_re, freqs1_im,
        freqs2_re, freqs2_im,
        out_q, out_k,
        start_frame,
        B, L, N, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        out_q.stride(0), out_q.stride(1), out_q.stride(2), out_q.stride(3),
        out_k.stride(0), out_k.stride(1), out_k.stride(2), out_k.stride(3),
        grid_sizes.stride(0), grid_sizes.stride(1),
        c0=c0, c1=c1, c2=c2,
        BLOCK_L=BLOCK_L,
        BLOCK_C=BLOCK_C,
    )

    return out_q, out_k


def rope_apply_temporal_shift(k_chunk: torch.Tensor, freqs: torch.Tensor, delta_frames: int) -> None:
    """
    k_chunk: [B, L_sink, H, D] (view of the sink portion of K, with RoPE already applied)
    freqs  : [1024, C/2] complex (self.freqs)
    delta_frames: how many frames to shift the sink to the left (negative) / right (positive)
    In-place, multiplies only the time-axis channels by exp(i * ω * delta_frames)
    """
    if delta_frames == 0:
        return

    B, L, H, D = k_chunk.shape
    assert D % 2 == 0
    c = D // 2
    t_c = c - 2 * (c // 3)   # time channel complex dim
    h_c = c // 3
    w_c = c // 3
    # freqs -> time / height / width split
    freqs_t, _, _ = freqs.split([t_c, h_c, w_c], dim=1)  # [1024, t_c], complex

    #  Complex rotation factor corresponding to the delta (for the time axis)
    shift = abs(int(delta_frames))
    max_pos = freqs_t.shape[0] - 1
    if shift > max_pos:
        shift = max_pos 
    mult = freqs_t[shift] if delta_frames >= 0 else torch.conj(freqs_t[shift])
    mult = mult.view(1, 1, 1, t_c)  # [1,1,1,t_c]

    # Convert only the time-axis channels to complex and multiply (in-place)
    time_ri = k_chunk[..., : 2 * t_c]                                            # [B,L,H,2*t_c]
    time_cx = torch.view_as_complex(time_ri.to(torch.float64).reshape(-1, t_c, 2))  # [(B*L*H), t_c]
    time_cx = time_cx * mult.to(time_cx.dtype)                                   # delta rotate
    time_ri_new = torch.view_as_real(time_cx).reshape(B, L, H, t_c, 2).flatten(-2)  # [B,L,H,2*t_c]
    time_ri.copy_(time_ri_new.to(time_ri.dtype))  # in-placedef _rope_time_delta_mul_(k_chunk: torch.Tensor, freqs: torch.Tensor, delta_frames: int) -> None:
