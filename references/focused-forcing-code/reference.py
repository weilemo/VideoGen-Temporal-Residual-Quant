from contextlib import contextmanager
import json
import time

import torch
import torch.nn.functional as F
import flash_attn


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
        print(f"{name} took {(t1 - t0) * 1000:.3f} ms")


def prepare_meta(loss_path, max_budget, min_budget, attn_weight, H=12, device="cuda"):
    with open(loss_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    loss_map = {int(k): float(v) for k, v in raw.items()}

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
            "kv_budget": kv_budget   # [H]
        }

    return meta


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
    time_ri.copy_(time_ri_new.to(time_ri.dtype))  # in-place


def compute_attn_scores(query, key, include_current, frame_seqlen=1560, pooled_tokens=30, alpha=0.5, softmax_scale=None):
    B, Lq, H, D = query.shape
    if not include_current:
        key = key[:, :-Lq, :, :]
    B, Lk, H, D = key.shape
    L = frame_seqlen
    P = pooled_tokens
    QF = Lq // L
    KF = Lk // L
    block = L // P

    if softmax_scale is None:
        softmax_scale = D ** -0.5

    q = query.view(B, QF, P, block, H, D)
    k = key.view(B, KF, P, block, H, D)

    q = alpha * q.mean(dim=3) + (1 - alpha) * q.amax(dim=3)
    k = alpha * k.mean(dim=3) + (1 - alpha) * k.amax(dim=3)

    q = q.permute(0, 3, 1, 2, 4).reshape(B, H, QF * P, D)
    k = k.permute(0, 3, 1, 2, 4).reshape(B, H, KF * P, D)

    attn_scores = (torch.matmul(q, k.transpose(-2, -1)) * softmax_scale) \
        .view(B, H, QF, P, KF, P).mean(dim=-1).mean(dim=3)

    mean = attn_scores.mean(dim=-1, keepdim=True)
    std = attn_scores.std(dim=-1, keepdim=True, correction=0)
    attn_scores = ((attn_scores - mean) / (std + 1e-8)).permute(0, 2, 1, 3)

    return attn_scores # [B, QF, H, KF]


def compute_key_diversity(query, key, include_current, frame_seqlen=1560):
    B, Lq, H, D = query.shape
    if not include_current:
        key = key[:, :-Lq, :, :]
    B, Lk, H, D = key.shape
    L = frame_seqlen
    QF = Lq // L
    KF = Lk // L

    key = key.view(B, KF, L, H, D)

    mean_pos = key.mean(dim=1)               
    mean_pos = F.normalize(mean_pos, p=2, dim=-1, eps=1e-8) # [B, L, H, D]

    k_norm = F.normalize(key, p=2, dim=-1, eps=1e-8) # [B, KF, L, H, D]

    similarity = (k_norm * mean_pos.unsqueeze(1)).sum(dim=-1)
    similarity = similarity.mean(dim=2).transpose(1, 2)

    diversity = (-similarity).unsqueeze(1).expand(-1, QF, -1, -1)
    mean = diversity.mean(dim=-1, keepdim=True)
    std = diversity.std(dim=-1, keepdim=True, correction=0)
    diversity = (diversity - mean) / (std + 1e-8)

    return diversity # [B, QF, H, KF]


def select_kv_row_indices(scores, kv_budget, include_current, frame_seqlen=1560):
    """
    scores: [B, QF, H, KF]
    kv_budget: [H]
    """
    B, QF, H, KF = scores.shape
    kv_budget = kv_budget.view(1, 1, H).expand(B, QF, H)
    L = frame_seqlen
    device = scores.device

    if not include_current:
        total_frames = QF + KF
        front_best = scores[..., :3].argmax(dim=-1)   # [B, QF, H]

        selected_mask = torch.zeros(B, QF, H, total_frames, device=device, dtype=torch.bool)
        selected_mask.scatter_(-1, front_best.unsqueeze(-1), True)
        selected_mask[..., -QF:] = True

        need = (kv_budget - QF - 1).clamp(min=0, max=KF - 1)   # [B, QF, H]

        cand_scores = scores.clone()
        cand_scores.scatter_(-1, front_best.unsqueeze(-1), torch.finfo(scores.dtype).min)

        order = cand_scores.argsort(dim=-1, descending=True)   # [B, QF, H, KF]
        rank = torch.arange(KF, device=device).view(1, 1, 1, KF)
        take_mask = rank < need.unsqueeze(-1)                  # [B, QF, H, KF]

        extra_mask = torch.zeros(B, QF, H, KF, device=device, dtype=torch.bool)
        extra_mask.scatter_(-1, order, take_mask)

        selected_mask[..., :KF] |= extra_mask

    else:
        total_frames = KF
        tail_start = total_frames - QF

        front_best = scores[..., :3].argmax(dim=-1)   # [B, QF, H]
        q_idx = torch.arange(QF, device=device).view(1, QF, 1).expand(B, QF, H)
        tail_match = tail_start + q_idx               # [B, QF, H]

        selected_mask = torch.zeros(B, QF, H, total_frames, device=device, dtype=torch.bool)
        selected_mask.scatter_(-1, front_best.unsqueeze(-1), True)
        selected_mask.scatter_(-1, tail_match.unsqueeze(-1), True)

        need = (kv_budget - 2).clamp(min=0, max=total_frames - 2)   # [B, QF, H]

        n_prefix = tail_start - 1
        n_tail = QF - 1

        prefix_quota = (need * n_prefix) // (n_prefix + n_tail)
        tail_quota = need - prefix_quota

        prefix_scores = scores[..., :tail_start].clone()   # [B, QF, H, tail_start]
        prefix_scores.scatter_(-1, front_best.unsqueeze(-1), torch.finfo(scores.dtype).min)

        prefix_order = prefix_scores.argsort(dim=-1, descending=True)
        prefix_rank = torch.arange(tail_start, device=device).view(1, 1, 1, tail_start)
        prefix_take = prefix_rank < prefix_quota.unsqueeze(-1)

        prefix_mask = torch.zeros(B, QF, H, tail_start, device=device, dtype=torch.bool)
        prefix_mask.scatter_(-1, prefix_order, prefix_take)
        selected_mask[..., :tail_start] |= prefix_mask

        tail_scores = scores[..., tail_start:]                    # [B, QF, H, QF]
        eye_mask = torch.eye(QF, device=device, dtype=torch.bool).view(1, QF, 1, QF)
        tail_scores = tail_scores.masked_fill(eye_mask, torch.finfo(scores.dtype).min)

        tail_order = tail_scores.argsort(dim=-1, descending=True)
        tail_rank = torch.arange(QF, device=device).view(1, 1, 1, QF)
        tail_take = tail_rank < tail_quota.unsqueeze(-1)

        tail_mask = torch.zeros(B, QF, H, QF, device=device, dtype=torch.bool)
        tail_mask.scatter_(-1, tail_order, tail_take)
        selected_mask[..., tail_start:] |= tail_mask

    frame_ids = torch.arange(total_frames, dtype=torch.int32, device=device).view(1, 1, 1, total_frames, 1)
    token_ids = torch.arange(L, dtype=torch.int32, device=device).view(1, 1, 1, 1, L)
    head_ids  = torch.arange(H, dtype=torch.int32, device=device).view(1, 1, H, 1, 1)

    # assert B == 1
    row = ((frame_ids * L + token_ids) * H + head_ids).expand(B, QF, H, total_frames, L).squeeze(0)
    mask = selected_mask.unsqueeze(-1).expand(-1, -1, -1, -1, L).squeeze(0)
    kv_row_indices = row[mask].to(torch.int32)

    return kv_row_indices


def pack_qkv(query, key, value, kv_budget, kv_row_indices, frame_seqlen=1560):
    '''
    kv_budget: [H], int32
    kv_row_indices: [N], int32, single batch
    '''
    B, Lq, H, D = query.shape
    B, Lk, H, D = key.shape
    # assert B == 1
    L = frame_seqlen
    QF = Lq // L
    G = QF * H

    q_packed = query[0].reshape(QF, L, H, D).permute(0, 2, 1, 3).reshape(Lq * H, D)
    k_packed = key[0].reshape(Lk * H, D).index_select(0, kv_row_indices.to(torch.long))
    v_packed = value[0].reshape(Lk * H, D).index_select(0, kv_row_indices.to(torch.long))

    cu_seqlens_q = torch.arange(0, (G + 1) * L, L, device=query.device, dtype=torch.int32)
    k_lens = (kv_budget.view(1, H).expand(QF, H).reshape(-1) * L).to(torch.int32)   # [G]
    cu_seqlens_k = torch.cat([torch.zeros(1, device=query.device, dtype=torch.int32), torch.cumsum(k_lens, dim=0)])
    max_seqlen_q = L
    max_seqlen_k = int(k_lens.max().item()) if G > 0 else 0

    return (
        q_packed.unsqueeze(1),   # [Lq * H, 1, D]
        k_packed.unsqueeze(1),   # [N, 1, D], N = kv_row_indices.numel()
        v_packed.unsqueeze(1),   # [N, 1, D]
        cu_seqlens_q,            # [QF * H + 1]
        cu_seqlens_k,            # [QF * H + 1]
        max_seqlen_q,            # int, equal to L
        max_seqlen_k,            # int
    )


def selected_frame_attention(
    query,
    key,
    value,
    kv_budget,
    kv_row_indices, 
    frame_seqlen=1560,
    softmax_scale=None,
    causal=False,
    dropout_p=0.0,
    deterministic=False,
):
    L = frame_seqlen
    B, Lq, H, D = query.shape
    # assert B == 1
    QF = Lq // L

    q_packed, k_packed, v_packed, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = pack_qkv(
        query, key, value, kv_budget, kv_row_indices, frame_seqlen
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
        out_packed.view(QF, H, L, D)    # [QF, H, L, D]
        .permute(0, 2, 1, 3)            # [QF, L, H, D]
        .reshape(B, QF * L, H, D)
        .contiguous()
    )

    return out
