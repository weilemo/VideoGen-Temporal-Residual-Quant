import torch
from . import rope_apply_temporal_shift_cuda as ext

_ROPE_TEMPORAL_SHIFT_CACHE = {}

D_FIXED = 128
C_FIXED = 64
T_C_FIXED = 22
H_C_FIXED = 21
W_C_FIXED = 21

def _get_temporal_mult_cached(
    freqs: torch.Tensor,
    delta_frames: int,
    device: torch.device,
):
    shift = abs(int(delta_frames))
    sign = 1 if delta_frames >= 0 else -1

    key = (device, freqs.data_ptr(), shift, sign)
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
    k_chunk: torch.Tensor,
    freqs: torch.Tensor,
    delta_frames: int,
) -> None:
    if delta_frames == 0:
        return

    assert k_chunk.is_cuda
    assert k_chunk.is_contiguous()
    assert k_chunk.shape[-1] == D_FIXED

    mult_re, mult_im = _get_temporal_mult_cached(
        freqs=freqs,
        delta_frames=delta_frames,
        device=k_chunk.device,
    )

    ext.rope_apply_temporal_shift(
        k_chunk, mult_re, mult_im
    )
