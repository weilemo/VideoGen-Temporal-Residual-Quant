import torch
from . import causal_rope_apply_cuda as ext

_ROPE_FREQ_CACHE = {}
_OUT_BUF_CACHE = {}

# fixed constants for D=128
C_HALF = 64
C0 = 22
C1 = 21
C2 = 21


def _get_cached_rope_freqs(freqs: torch.Tensor, device: torch.device):
    key = (int(freqs.data_ptr()), device.index if device.index is not None else -1)
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
      freqs     : [1024, 64], complex128/complex64
    """
    assert q.is_cuda and k.is_cuda
    assert q.shape == (1, 4680, 12, 128)
    assert k.shape == (1, 4680, 12, 128)
    assert q.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert k.dtype == q.dtype
    assert freqs.shape[1] == 64

    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()

    if grid_sizes.device != q.device or grid_sizes.dtype != torch.int32 or not grid_sizes.is_contiguous():
        grid_sizes = grid_sizes.to(device=q.device, dtype=torch.int32, non_blocking=True).contiguous()

    (
        freqs0_re, freqs0_im,
        freqs1_re, freqs1_im,
        freqs2_re, freqs2_im,
    ) = _get_cached_rope_freqs(freqs, q.device)

    buf_key = (q.device, q.dtype, q.shape)
    cached = _OUT_BUF_CACHE.get(buf_key, None)
    if cached is None:
        out_q = torch.empty_like(q)
        out_k = torch.empty_like(k)
        _OUT_BUF_CACHE[buf_key] = (out_q, out_k)
    else:
        out_q, out_k = cached

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
