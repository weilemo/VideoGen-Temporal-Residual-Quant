import importlib
from types import ModuleType

import torch

ext: ModuleType = importlib.import_module(".focusedforcing_cuda", package=__package__)


B_FIXED = 1
T_FIXED = 1560
H_FIXED = 12
D_FIXED = 128

TOTAL_FRAMES = 21
FRONT_FIXED = 3
QF_MAX = TOTAL_FRAMES - FRONT_FIXED  # 18

# rope split for D=128
C_HALF = 64
C0 = 22
C1 = 21
C2 = 21


_MEAN_POS_CACHE = {}
_OUT_DIV_CACHE = {}

_ROPE_FREQ_CACHE = {}
_CAUSAL_ROPE_OUT_BUF_CACHE = {}

_ATTN_SCORE_CACHE = {}

_PACK_Q_CACHE = {}
_PACK_KV_CACHE = {}

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


def compute_key_diversity_cuda(k: torch.Tensor):
    """
    Args:
      k: [B, L, 12, 128], bf16/fp16/fp32, CUDA
         where L % 1560 == 0

    Returns:
      out_div: [B, 12, F], float32
         where F = L // 1560
    """
    assert k.is_cuda
    assert k.ndim == 4
    assert k.shape[2:] == (H_FIXED, D_FIXED)
    assert k.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert k.shape[1] % T_FIXED == 0

    k = _ensure_contiguous(k)

    B = int(k.shape[0])
    L = int(k.shape[1])
    F = L // T_FIXED
    assert 1 <= F <= 21

    mean_key = (_device_key(k.device), k.dtype, B)
    mean_pos = _cache_get_or_create(
        _MEAN_POS_CACHE,
        mean_key,
        lambda: torch.empty(
            (B, T_FIXED, H_FIXED, D_FIXED),
            device=k.device,
            dtype=k.dtype,
        ),
    )

    out_key = (_device_key(k.device), B, F)
    out_div = _cache_get_or_create(
        _OUT_DIV_CACHE,
        out_key,
        lambda: torch.empty(
            (B, H_FIXED, F),
            device=k.device,
            dtype=torch.float32,
        ),
    )

    ext.compute_key_diversity(k, mean_pos, out_div)
    return out_div


def compute_attn_scores_cuda(query: torch.Tensor, key: torch.Tensor):
    """
    Args:
      query: [1, Lq, 12, 128]
      key:   [1, Lk, 12, 128]

    Returns:
      out: [1, qf, 12, kf], float32
    """
    assert query.is_cuda and key.is_cuda
    assert query.ndim == 4 and key.ndim == 4
    assert query.shape[0] == 1 and key.shape[0] == 1
    assert query.shape[2:] == (H_FIXED, D_FIXED)
    assert key.shape[2:] == (H_FIXED, D_FIXED)
    assert query.dtype == key.dtype
    assert query.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert query.shape[1] % T_FIXED == 0
    assert key.shape[1] % T_FIXED == 0

    query = _ensure_contiguous(query)
    key = _ensure_contiguous(key)

    qf = int(query.shape[1] // T_FIXED)
    kf = int(key.shape[1] // T_FIXED)
    qp = qf * 30
    kp = kf * 30

    cache_key = (_device_key(query.device), query.dtype, qf, kf)
    pooled_q, pooled_k, gemm_out, out = _cache_get_or_create(
        _ATTN_SCORE_CACHE,
        cache_key,
        lambda: (
            torch.empty((H_FIXED, qp, D_FIXED), device=query.device, dtype=query.dtype),
            torch.empty((H_FIXED, kp, D_FIXED), device=key.device, dtype=key.dtype),
            torch.empty((H_FIXED, qp, kp), device=query.device, dtype=torch.float32),
            torch.empty((1, qf, H_FIXED, kf), device=query.device, dtype=torch.float32),
        ),
    )

    ext.compute_attn_scores(query, key, pooled_q, pooled_k, gemm_out, out)
    return out


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
    q: torch.Tensor | None,
    k: torch.Tensor | None,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0,
):
    """
    Args:
      q: [1, L, 12, 128] or None
      k: [1, L, 12, 128] or None
      grid_sizes: [1, 3], int32/int64 on CPU/CUDA
      freqs: [Tmax, 64], complex64/complex128

    Returns:
      (out_q, out_k), where a missing side is None
    """
    assert q is not None or k is not None

    do_q = q is not None
    do_k = k is not None

    ref = q if do_q else k
    assert ref is not None
    assert ref.is_cuda
    assert ref.ndim == 4
    assert ref.shape[0] == 1
    assert ref.shape[2:] == (H_FIXED, D_FIXED)
    assert ref.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert ref.shape[1] % T_FIXED == 0
    assert freqs.shape[1] == 64

    if do_q:
        assert q is not None
        assert q.is_cuda
        assert q.dtype == ref.dtype
        assert q.shape == ref.shape
        q = _ensure_contiguous(q)

    if do_k:
        assert k is not None
        assert k.is_cuda
        assert k.dtype == ref.dtype
        assert k.shape == ref.shape
        k = _ensure_contiguous(k)

    if grid_sizes.device != ref.device or grid_sizes.dtype != torch.int32 or not grid_sizes.is_contiguous():
        grid_sizes = grid_sizes.to(device=ref.device, dtype=torch.int32, non_blocking=True).contiguous()

    (
        freqs0_re, freqs0_im,
        freqs1_re, freqs1_im,
        freqs2_re, freqs2_im,
    ) = _get_cached_rope_freqs(freqs, ref.device)

    out_q = None
    out_k = None

    if do_q:
        q_key = (_device_key(ref.device), ref.dtype, "q", _shape_key(q.shape))
        out_q = _cache_get_or_create(
            _CAUSAL_ROPE_OUT_BUF_CACHE,
            q_key,
            lambda: torch.empty_like(q),
        )

    if do_k:
        k_key = (_device_key(ref.device), ref.dtype, "k", _shape_key(k.shape))
        out_k = _cache_get_or_create(
            _CAUSAL_ROPE_OUT_BUF_CACHE,
            k_key,
            lambda: torch.empty_like(k),
        )

    ext.causal_rope_apply(
        q if do_q else torch.Tensor(),
        k if do_k else torch.Tensor(),
        grid_sizes,
        freqs0_re, freqs0_im,
        freqs1_re, freqs1_im,
        freqs2_re, freqs2_im,
        out_q if do_q else torch.Tensor(),
        out_k if do_k else torch.Tensor(),
        int(start_frame),
        bool(do_q),
        bool(do_k),
    )

    return out_q, out_k


def select_kv_row_indices_cuda(
    scores: torch.Tensor,
    kv_budget: torch.Tensor,
    update: bool,
):
    """
    Args:
      scores: [QF_total, 12, 21], float32, CUDA
      kv_budget: [12], int32, CUDA
      update:
        True  -> legacy rule
        False -> new rule

    Returns:
      kv_row_indices: [sum(kv_budget) * 1560], int32, CUDA
      kv_budget: [QF_total, 12], int32, CUDA
    """
    assert scores.is_cuda and kv_budget.is_cuda
    assert scores.dtype == torch.float32
    assert kv_budget.dtype == torch.int32
    assert scores.ndim == 3
    assert kv_budget.ndim == 1

    QF_total = int(scores.shape[0])
    assert 1 <= QF_total <= QF_MAX
    assert scores.shape == (QF_total, H_FIXED, TOTAL_FRAMES)
    assert kv_budget.shape == (H_FIXED,)

    scores = _ensure_contiguous(scores)
    kv_budget = _ensure_contiguous(kv_budget)

    kv_budget = kv_budget.unsqueeze(0).expand(QF_total, -1).contiguous()

    flat_budget = kv_budget.reshape(-1)
    offsets = (
        torch.cumsum(flat_budget, dim=0) - flat_budget
    ).to(torch.int32).view(QF_total, H_FIXED).contiguous()

    total_rows = int(flat_budget.sum().item()) * T_FIXED
    out = torch.empty((total_rows,), device=scores.device, dtype=torch.int32)

    ext.select_kv_row_indices(
        scores,
        kv_budget,
        offsets,
        out,
        bool(update),
    )
    return out, kv_budget


def pack_qkv_cuda(
    roped_query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_row_indices: torch.Tensor,
    kv_budget: torch.Tensor,
):
    """
    Args:
      roped_query:    [1, Lq, 12, 128], fp16/bf16, CUDA
      key:            [1, Lk, 12, 128], same dtype/device
      value:          [1, Lk, 12, 128], same dtype/device
      kv_row_indices: [N], int32, CUDA
      kv_budget:      [QF_total, 12], int32, CUDA

    Returns:
      q_packed:      [Lq*12, 1, 128]
      k_packed:      [N, 1, 128]
      v_packed:      [N, 1, 128]
      cu_seqlens_q:  [QF_total*12 + 1], int32
      cu_seqlens_k:  [QF_total*12 + 1], int32
      max_seqlen_q:  int
      max_seqlen_k:  int
    """
    assert roped_query.is_cuda and key.is_cuda and value.is_cuda
    assert kv_row_indices.is_cuda and kv_budget.is_cuda

    assert roped_query.dtype in (torch.float16, torch.bfloat16)
    assert key.dtype == roped_query.dtype
    assert value.dtype == roped_query.dtype
    assert kv_row_indices.dtype == torch.int32
    assert kv_budget.dtype == torch.int32

    assert roped_query.ndim == 4 and roped_query.shape[0] == 1 and roped_query.shape[2:] == (H_FIXED, D_FIXED)
    assert key.ndim == 4 and key.shape[0] == 1 and key.shape[2:] == (H_FIXED, D_FIXED)
    assert value.shape == key.shape

    roped_query = _ensure_contiguous(roped_query)
    key = _ensure_contiguous(key)
    value = _ensure_contiguous(value)
    kv_row_indices = _ensure_contiguous(kv_row_indices)
    kv_budget = _ensure_contiguous(kv_budget)

    Lq = int(roped_query.shape[1])
    Lk = int(key.shape[1])

    assert Lq % T_FIXED == 0
    assert Lk % T_FIXED == 0

    QF_total = Lq // T_FIXED
    assert kv_budget.ndim == 2 and kv_budget.shape == (QF_total, H_FIXED)

    q_rows = Lq * H_FIXED
    n = int(kv_row_indices.numel())
    G = QF_total * H_FIXED

    # Optional sanity check: kv_rows length should match kv_budget
    expected_n = int(kv_budget.sum().item()) * T_FIXED
    assert n == expected_n, f"kv_row_indices.numel()={n}, expected {expected_n}"

    q_key = (_device_key(roped_query.device), roped_query.dtype, q_rows)
    q_out = _cache_get_or_create(
        _PACK_Q_CACHE,
        q_key,
        lambda: torch.empty((q_rows, D_FIXED), device=roped_query.device, dtype=roped_query.dtype),
    )

    kv_key = (_device_key(roped_query.device), key.dtype, n)
    k_out, v_out = _cache_get_or_create(
        _PACK_KV_CACHE,
        kv_key,
        lambda: (
            torch.empty((n, D_FIXED), device=roped_query.device, dtype=key.dtype),
            torch.empty((n, D_FIXED), device=roped_query.device, dtype=value.dtype),
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

    # ------------------------------------------------------------
    # varlen metadata for flash_attn_varlen_func
    #
    # q is packed as G sequences, each length T_FIXED
    # k is packed as G sequences, lengths kv_budget.flatten() * T_FIXED
    #
    # flatten order must match q pack order:
    #   (qf=0,h=0), (qf=0,h=1), ..., (qf=QF_total-1,h=11)
    # kv_budget.reshape(-1) on contiguous [QF_total, 12] matches this.
    # ------------------------------------------------------------
    cu_seqlens_q = torch.arange(
        0,
        (G + 1) * T_FIXED,
        T_FIXED,
        device=roped_query.device,
        dtype=torch.int32,
    )

    k_lens = (kv_budget.reshape(-1) * T_FIXED).to(torch.int32)   # [G]
    cu_seqlens_k = torch.empty((G + 1,), device=roped_query.device, dtype=torch.int32)
    cu_seqlens_k[0] = 0
    cu_seqlens_k[1:] = torch.cumsum(k_lens, dim=0)

    max_seqlen_q = T_FIXED
    max_seqlen_k = int(k_lens.max().item()) if G > 0 else 0

    return (
        q_out.unsqueeze(1),   # [total_q, 1, 128]
        k_out.unsqueeze(1),   # [total_k, 1, 128]
        v_out.unsqueeze(1),   # [total_k, 1, 128]
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )


def concat_cuda(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
):
    """
    a: [1, La, 12, 128]
    b: [1, Lb, 12, 128]
    c: [1, Lc, 12, 128]

    return:
      out: [1, La+Lb+Lc, 12, 128]
    """
    assert a.is_cuda and b.is_cuda and c.is_cuda
    assert a.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert b.dtype == a.dtype and c.dtype == a.dtype
    assert a.ndim == 4 and b.ndim == 4 and c.ndim == 4
    assert a.shape[0] == 1 and b.shape[0] == 1 and c.shape[0] == 1
    assert a.shape[2:] == (H_FIXED, D_FIXED)
    assert b.shape[2:] == (H_FIXED, D_FIXED)
    assert c.shape[2:] == (H_FIXED, D_FIXED)

    a = _ensure_contiguous(a)
    b = _ensure_contiguous(b)
    c = _ensure_contiguous(c)

    Lo = int(a.shape[1] + b.shape[1] + c.shape[1])

    out = torch.empty((1, Lo, H_FIXED, D_FIXED), device=a.device, dtype=a.dtype)

    ext.concat(a, b, c, out)
    return out
