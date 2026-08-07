import torch
import importlib
from types import ModuleType

ext: ModuleType = importlib.import_module(".focusedforcing_cuda", package=__package__)



# ============================================================
# fixed constants
# ============================================================

NUM_BLOCKS = 30
B_FIXED = 1
F_FIXED = 18
T_FIXED = 1560
H_FIXED = 12
D_FIXED = 128
L_FIXED = F_FIXED * T_FIXED

# rope split for D=128
C_HALF = 64
C0 = 22
C1 = 21
C2 = 21

# temporal shift split
T_C_FIXED = 22
H_C_FIXED = 21
W_C_FIXED = 21


# ============================================================
# global caches
# ============================================================

_MEAN_POS_CACHE = {}
_OUT_DIV_CACHE = {}

_ROPE_TEMPORAL_SHIFT_CACHE = {}

_ROPE_FREQ_CACHE = {}
_CAUSAL_ROPE_OUT_BUF_CACHE = {}

_ATTN_SCORE_CACHE = {}

_PACK_Q_CACHE = {}
_PACK_KV_CACHE = {}


# ============================================================
# common helpers
# ============================================================

def _ensure_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def _cache_get_or_create(cache: dict, key, factory):
    value = cache.get(key, None)
    if value is None:
        value = factory()
        cache[key] = value
    return value


def _device_key(device: torch.device):
    return (device.type, device.index if device.index is not None else -1)


def _shape_key(shape):
    return tuple(int(v) for v in shape)


# ============================================================
# 1) compute_key_diversity
# ============================================================

def compute_key_diversity_cuda(k: torch.Tensor):
    """
    Args:
      k: [30, 1, 18*1560, 12, 128], bf16/fp16/fp32, CUDA

    Returns:
      div: [30, 1, 12, 18], float32
    """
    assert k.is_cuda
    assert k.shape == (NUM_BLOCKS, B_FIXED, L_FIXED, H_FIXED, D_FIXED)
    assert k.dtype in (torch.float16, torch.bfloat16, torch.float32)

    k = _ensure_contiguous(k)

    mean_key = (_device_key(k.device), k.dtype)
    mean_pos = _cache_get_or_create(
        _MEAN_POS_CACHE,
        mean_key,
        lambda: torch.empty(
            (NUM_BLOCKS, T_FIXED, H_FIXED, D_FIXED),
            device=k.device,
            dtype=k.dtype,
        ),
    )

    out_key = _device_key(k.device)
    out_div = _cache_get_or_create(
        _OUT_DIV_CACHE,
        out_key,
        lambda: torch.empty(
            (NUM_BLOCKS, B_FIXED, H_FIXED, F_FIXED),
            device=k.device,
            dtype=torch.float32,
        ),
    )

    ext.compute_key_diversity(k, mean_pos, out_div)
    return out_div


# ============================================================
# 2) rope_apply_temporal_shift
# ============================================================

def _get_temporal_mult_cached(
    freqs: torch.Tensor,
    delta_frames: int,
    device: torch.device,
):
    shift = abs(int(delta_frames))
    sign = 1 if delta_frames >= 0 else -1

    key = (_device_key(device), int(freqs.data_ptr()), shift, sign)
    cached = _ROPE_TEMPORAL_SHIFT_CACHE.get(key)
    if cached is not None:
        return cached

    freqs_t, _, _ = freqs.split([T_C_FIXED, H_C_FIXED, W_C_FIXED], dim=1)
    max_pos = freqs_t.shape[0] - 1
    if shift > max_pos:
        shift = max_pos

    mult = freqs_t[shift] if sign > 0 else torch.conj(freqs_t[shift])

    mult_ri = torch.view_as_real(mult.to(torch.complex128)).contiguous()
    mult_re = mult_ri[:, 0].to(device=device, dtype=torch.float64).contiguous()
    mult_im = mult_ri[:, 1].to(device=device, dtype=torch.float64).contiguous()

    _ROPE_TEMPORAL_SHIFT_CACHE[key] = (mult_re, mult_im)
    return mult_re, mult_im


def rope_apply_temporal_shift_cuda(
    k_all: torch.Tensor,
    freqs: torch.Tensor,
    delta_frames: int,
) -> None:
    if delta_frames == 0:
        return

    assert k_all.is_cuda
    assert k_all.dim() == 5
    assert k_all.shape[0] == NUM_BLOCKS
    assert k_all.shape[1] == B_FIXED
    assert k_all.shape[-1] == D_FIXED
    assert k_all.dtype in (torch.float16, torch.bfloat16, torch.float32)

    mult_re, mult_im = _get_temporal_mult_cached(
        freqs=freqs,
        delta_frames=delta_frames,
        device=k_all.device,
    )

    ext.rope_apply_temporal_shift(k_all, mult_re, mult_im)


# ============================================================
# 3) compute_attn_scores
# ============================================================

def compute_attn_scores_cuda(query: torch.Tensor, key: torch.Tensor):
    assert query.is_cuda and key.is_cuda
    assert query.shape == (1, 4680, 12, 128)
    assert key.shape == (1, 32760, 12, 128)
    assert query.dtype == key.dtype
    assert query.dtype in (torch.float32, torch.float16, torch.bfloat16)

    query = _ensure_contiguous(query)
    key = _ensure_contiguous(key)

    cache_key = (_device_key(query.device), query.dtype)

    pooled_q, pooled_k, gemm_out, out = _cache_get_or_create(
        _ATTN_SCORE_CACHE,
        cache_key,
        lambda: (
            torch.empty((12, 90, 128), device=query.device, dtype=query.dtype),
            torch.empty((12, 540, 128), device=key.device, dtype=key.dtype),
            torch.empty((12, 90, 540), device=query.device, dtype=torch.float32),
            torch.empty((1, 3, 12, 18), device=query.device, dtype=torch.float32),
        ),
    )

    ext.compute_attn_scores(query, key, pooled_q, pooled_k, gemm_out, out)
    return out


# ============================================================
# 4) causal_rope_apply
# ============================================================

def _get_cached_rope_freqs(freqs: torch.Tensor, device: torch.device):
    key = (int(freqs.data_ptr()), _device_key(device))
    cached = _ROPE_FREQ_CACHE.get(key, None)
    if cached is not None:
        return cached

    if freqs.device == device and freqs.dtype == torch.complex64:
        freqs_dev = freqs
    else:
        freqs_dev = freqs.to(device=device, dtype=torch.complex64, non_blocking=True)

    assert freqs_dev.shape[1] == 64, "freqs must be [Tmax, 64] for D=128"

    freqs0 = freqs_dev[:, :C0]
    freqs1 = freqs_dev[:, C0:C0 + C1]
    freqs2 = freqs_dev[:, C0 + C1:]

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
    )
    _ROPE_FREQ_CACHE[key] = cached
    return cached


def causal_rope_apply_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0,
):
    """
    Fixed-shape fast path:
      q, k      : [1, 4680, 12, 128], bf16/fp16/fp32, CUDA
      grid_sizes: [1, 3], int32/int64
      freqs     : [Tmax, 64], complex128/complex64
    """
    assert q.is_cuda and k.is_cuda
    assert q.shape == (1, 4680, 12, 128)
    assert k.shape == (1, 4680, 12, 128)
    assert q.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert k.dtype == q.dtype
    assert freqs.shape[1] == 64

    q = _ensure_contiguous(q)
    k = _ensure_contiguous(k)

    if grid_sizes.device != q.device or grid_sizes.dtype != torch.int32 or not grid_sizes.is_contiguous():
        grid_sizes = grid_sizes.to(device=q.device, dtype=torch.int32, non_blocking=True).contiguous()

    (
        freqs0_re, freqs0_im,
        freqs1_re, freqs1_im,
        freqs2_re, freqs2_im,
    ) = _get_cached_rope_freqs(freqs, q.device)

    buf_key = (_device_key(q.device), q.dtype, _shape_key(q.shape))
    out_q, out_k = _cache_get_or_create(
        _CAUSAL_ROPE_OUT_BUF_CACHE,
        buf_key,
        lambda: (torch.empty_like(q), torch.empty_like(k)),
    )

    ext.causal_rope_apply(
        q, k,
        grid_sizes,
        freqs0_re, freqs0_im,
        freqs1_re, freqs1_im,
        freqs2_re, freqs2_im,
        out_q, out_k,
        int(start_frame),
    )

    return out_q, out_k


# ============================================================
# 5) pack_qkv
# ============================================================

def pack_qkv_cuda(
    roped_query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_row_indices: torch.Tensor,
):
    """
    roped_query:   [1,4680,12,128] bf16 CUDA
    key:           [1,32760,12,128] bf16 CUDA
    value:         [1,32760,12,128] bf16 CUDA
    kv_row_indices:[N] int32 CUDA

    returns:
      q_packed: [56160,1,128]
      k_packed: [N,1,128]
      v_packed: [N,1,128]
    """
    assert roped_query.is_cuda and key.is_cuda and value.is_cuda and kv_row_indices.is_cuda
    assert roped_query.dtype == torch.bfloat16
    assert key.dtype == torch.bfloat16
    assert value.dtype == torch.bfloat16
    assert kv_row_indices.dtype == torch.int32

    assert roped_query.shape == (1, 4680, 12, 128)
    assert key.shape == (1, 32760, 12, 128)
    assert value.shape == (1, 32760, 12, 128)
    assert kv_row_indices.ndim == 1

    roped_query = _ensure_contiguous(roped_query)
    key = _ensure_contiguous(key)
    value = _ensure_contiguous(value)
    kv_row_indices = _ensure_contiguous(kv_row_indices)

    q_key = _device_key(roped_query.device)
    q_out = _cache_get_or_create(
        _PACK_Q_CACHE,
        q_key,
        lambda: torch.empty((56160, 128), device=roped_query.device, dtype=torch.bfloat16),
    )

    n = int(kv_row_indices.numel())
    kv_key = (_device_key(roped_query.device), n)
    k_out, v_out = _cache_get_or_create(
        _PACK_KV_CACHE,
        kv_key,
        lambda: (
            torch.empty((n, 128), device=roped_query.device, dtype=torch.bfloat16),
            torch.empty((n, 128), device=roped_query.device, dtype=torch.bfloat16),
        ),
    )

    ext.pack_qkv(
        roped_query,
        key,
        value,
        kv_row_indices,
        q_out,
        k_out,
        v_out,
    )

    return q_out.unsqueeze(1), k_out.unsqueeze(1), v_out.unsqueeze(1)


# ============================================================
# 6) select_kv_row_indices
# ============================================================

def select_kv_row_indices_cuda(scores: torch.Tensor, kv_budget: torch.Tensor):
    """
    scores: [3,12,18], float32, CUDA
    kv_budget: [3,12], int32, CUDA

    return:
      kv_row_indices: [sum(kv_budget)*1560], int32, CUDA
    """
    assert scores.is_cuda and kv_budget.is_cuda
    assert scores.dtype == torch.float32
    assert kv_budget.dtype == torch.int32
    assert scores.shape == (3, 12, 18)
    assert kv_budget.shape == (3, 12)

    scores = _ensure_contiguous(scores)
    kv_budget = _ensure_contiguous(kv_budget)

    flat_budget = kv_budget.view(-1)
    offsets = (torch.cumsum(flat_budget, dim=0) - flat_budget).to(torch.int32).view(3, 12).contiguous()

    total_frames_selected = int(flat_budget.sum().item())
    total_rows = total_frames_selected * 1560

    out = torch.empty((total_rows,), device=scores.device, dtype=torch.int32)

    ext.select_kv_row_indices(scores, kv_budget, offsets, out)
    return out
