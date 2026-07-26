import torch
import triton
import triton.language as tl


@triton.jit
def _s2pp_anchor_dequant_kernel(
    q_ptr,
    scales_ptr,
    output_ptr,
    S_OUT: tl.constexpr,
    UNIT_LEN: tl.constexpr,
    D: tl.constexpr,
    D_PACKED: tl.constexpr,
    SCALE_D: tl.constexpr,
    NUM_BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d = tl.program_id(2)

    offsets_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offsets_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (offsets_s[:, None] < UNIT_LEN) & (offsets_d[None, :] < D)

    if NUM_BITS == 8:
        q_offsets = pid_bh * UNIT_LEN * D + offsets_s[:, None] * D + offsets_d[None, :]
        q_value = tl.load(q_ptr + q_offsets, mask=mask, other=0).to(tl.float32)
    else:
        values_per_byte = 8 // NUM_BITS
        q_d = offsets_d // values_per_byte
        lane = offsets_d - q_d * values_per_byte
        shifts = NUM_BITS * (values_per_byte - 1 - lane)
        q_offsets = pid_bh * UNIT_LEN * D_PACKED + offsets_s[:, None] * D_PACKED + q_d[None, :]
        q_packed = tl.load(q_ptr + q_offsets, mask=mask, other=0).to(tl.int32)
        q_unsigned = (q_packed >> shifts[None, :]) & ((1 << NUM_BITS) - 1)
        q_value = (q_unsigned - ((1 << (NUM_BITS - 1)) - 1)).to(tl.float32)

    scale_d = offsets_d // BLOCK_SIZE
    scale_offsets = pid_bh * UNIT_LEN * SCALE_D + offsets_s[:, None] * SCALE_D + scale_d[None, :]
    scales = tl.load(scales_ptr + scale_offsets, mask=mask, other=1.0).to(tl.float32)
    value = q_value * scales

    out_offsets = pid_bh * S_OUT * D + offsets_s[:, None] * D + offsets_d[None, :]
    tl.store(output_ptr + out_offsets, value, mask=mask)


@triton.jit
def _s2pp_residual_dequant_kernel(
    q_ptr,
    scales_ptr,
    zero_points_ptr,
    alpha_ptr,
    beta_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    output_ptr,
    S_OUT: tl.constexpr,
    RESIDUAL_S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    D_PACKED: tl.constexpr,
    SCALE_D: tl.constexpr,
    CURRENT_START: tl.constexpr,
    PREV_START: tl.constexpr,
    RESIDUAL_START: tl.constexpr,
    UNIT_LEN: tl.constexpr,
    NUM_BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_AFFINE: tl.constexpr,
    AFFINE_PER_HEAD: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    ASYMMETRIC: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d = tl.program_id(2)

    offsets_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offsets_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    valid = (offsets_s[:, None] < UNIT_LEN) & (offsets_d[None, :] < D)

    q_s = RESIDUAL_START + offsets_s
    if NUM_BITS == 8:
        q_offsets = pid_bh * RESIDUAL_S * D + q_s[:, None] * D + offsets_d[None, :]
        q_loaded = tl.load(q_ptr + q_offsets, mask=valid, other=0).to(tl.int32)
        if ASYMMETRIC:
            q_value = q_loaded.to(tl.float32)
        else:
            q_value = q_loaded.to(tl.float32)
    else:
        values_per_byte = 8 // NUM_BITS
        q_d = offsets_d // values_per_byte
        lane = offsets_d - q_d * values_per_byte
        shifts = NUM_BITS * (values_per_byte - 1 - lane)
        q_offsets = pid_bh * RESIDUAL_S * D_PACKED + q_s[:, None] * D_PACKED + q_d[None, :]
        q_packed = tl.load(q_ptr + q_offsets, mask=valid, other=0).to(tl.int32)
        q_unsigned = (q_packed >> shifts[None, :]) & ((1 << NUM_BITS) - 1)
        if ASYMMETRIC:
            q_value = q_unsigned.to(tl.float32)
        else:
            q_value = (q_unsigned - ((1 << (NUM_BITS - 1)) - 1)).to(tl.float32)

    scale_d = offsets_d // BLOCK_SIZE
    scale_offsets = pid_bh * RESIDUAL_S * SCALE_D + q_s[:, None] * SCALE_D + scale_d[None, :]
    scales = tl.load(scales_ptr + scale_offsets, mask=valid, other=1.0).to(tl.float32)
    if ASYMMETRIC:
        zero_points = tl.load(zero_points_ptr + scale_offsets, mask=valid, other=0).to(tl.float32)
        residual = (q_value - zero_points) * scales
    else:
        residual = q_value * scales

    prev_s = PREV_START + offsets_s
    prev_offsets = pid_bh * S_OUT * D + prev_s[:, None] * D + offsets_d[None, :]
    prev = tl.load(output_ptr + prev_offsets, mask=valid, other=0.0).to(tl.float32)

    if HAS_ROPE:
        pair = offsets_d // 2
        pair_d = pair * 2
        prev_even_offsets = pid_bh * S_OUT * D + prev_s[:, None] * D + pair_d[None, :]
        prev_odd_offsets = prev_even_offsets + 1
        prev_even = tl.load(output_ptr + prev_even_offsets, mask=valid, other=0.0).to(tl.float32)
        prev_odd = tl.load(output_ptr + prev_odd_offsets, mask=valid, other=0.0).to(tl.float32)
        cos = tl.load(rope_cos_ptr + pair, mask=offsets_d < D, other=1.0).to(tl.float32)
        sin = tl.load(rope_sin_ptr + pair, mask=offsets_d < D, other=0.0).to(tl.float32)
        even_rot = prev_even * cos[None, :] - prev_odd * sin[None, :]
        odd_rot = prev_even * sin[None, :] + prev_odd * cos[None, :]
        prev = tl.where(offsets_d[None, :] - pair_d[None, :] == 0, even_rot, odd_rot)

    prediction = prev
    if HAS_AFFINE:
        affine_offsets = offsets_d
        if AFFINE_PER_HEAD:
            head_id = pid_bh % H
            affine_offsets = head_id * D + offsets_d
        alpha = tl.load(alpha_ptr + affine_offsets, mask=offsets_d < D, other=1.0).to(tl.float32)
        beta = tl.load(beta_ptr + affine_offsets, mask=offsets_d < D, other=0.0).to(tl.float32)
        prediction = prediction * alpha[None, :] + beta[None, :]

    current = prediction + residual
    out_s = CURRENT_START + offsets_s
    out_offsets = pid_bh * S_OUT * D + out_s[:, None] * D + offsets_d[None, :]
    tl.store(output_ptr + out_offsets, current, mask=valid)


@triton.jit
def _s2pp_multiframe_residual_dequant_kernel(
    q_ptr,
    scales_ptr,
    zero_points_ptr,
    history_weights_ptr,
    history_bias_ptr,
    output_ptr,
    S_OUT: tl.constexpr,
    RESIDUAL_S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    D_PACKED: tl.constexpr,
    SCALE_D: tl.constexpr,
    CURRENT_START: tl.constexpr,
    RESIDUAL_START: tl.constexpr,
    PREDICTOR_STRIDE: tl.constexpr,
    HISTORY_COUNT: tl.constexpr,
    UNIT_LEN: tl.constexpr,
    NUM_BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ASYMMETRIC: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Dequantize one unit and apply an AR(p) reconstructed-history predictor."""
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d = tl.program_id(2)

    offsets_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offsets_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    valid = (offsets_s[:, None] < UNIT_LEN) & (offsets_d[None, :] < D)

    q_s = RESIDUAL_START + offsets_s
    if NUM_BITS == 8:
        q_offsets = pid_bh * RESIDUAL_S * D + q_s[:, None] * D + offsets_d[None, :]
        q_loaded = tl.load(q_ptr + q_offsets, mask=valid, other=0).to(tl.int32)
        q_value = q_loaded.to(tl.float32)
    else:
        values_per_byte = 8 // NUM_BITS
        q_d = offsets_d // values_per_byte
        lane = offsets_d - q_d * values_per_byte
        shifts = NUM_BITS * (values_per_byte - 1 - lane)
        q_offsets = pid_bh * RESIDUAL_S * D_PACKED + q_s[:, None] * D_PACKED + q_d[None, :]
        q_packed = tl.load(q_ptr + q_offsets, mask=valid, other=0).to(tl.int32)
        q_unsigned = (q_packed >> shifts[None, :]) & ((1 << NUM_BITS) - 1)
        if ASYMMETRIC:
            q_value = q_unsigned.to(tl.float32)
        else:
            q_value = (q_unsigned - ((1 << (NUM_BITS - 1)) - 1)).to(tl.float32)

    scale_d = offsets_d // BLOCK_SIZE
    scale_offsets = pid_bh * RESIDUAL_S * SCALE_D + q_s[:, None] * SCALE_D + scale_d[None, :]
    scales = tl.load(scales_ptr + scale_offsets, mask=valid, other=1.0).to(tl.float32)
    if ASYMMETRIC:
        zero_points = tl.load(zero_points_ptr + scale_offsets, mask=valid, other=0).to(tl.float32)
        residual = (q_value - zero_points) * scales
    else:
        residual = q_value * scales

    head_id = pid_bh % H
    bias = tl.load(
        history_bias_ptr + head_id * D + offsets_d,
        mask=offsets_d < D,
        other=0.0,
    ).to(tl.float32)
    prediction = bias[None, :] + tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    for lag in tl.static_range(0, HISTORY_COUNT):
        previous_s = CURRENT_START - (lag + 1) * PREDICTOR_STRIDE + offsets_s
        previous_offsets = (
            pid_bh * S_OUT * D
            + previous_s[:, None] * D
            + offsets_d[None, :]
        )
        previous = tl.load(
            output_ptr + previous_offsets, mask=valid, other=0.0
        ).to(tl.float32)
        coefficient = tl.load(
            history_weights_ptr
            + lag * H * D
            + head_id * D
            + offsets_d,
            mask=offsets_d < D,
            other=0.0,
        ).to(tl.float32)
        prediction += previous * coefficient[None, :]

    current = prediction + residual
    out_s = CURRENT_START + offsets_s
    out_offsets = pid_bh * S_OUT * D + out_s[:, None] * D + offsets_d[None, :]
    tl.store(output_ptr + out_offsets, current, mask=valid)


@triton.jit
def _s2pp_cross_kv_residual_dequant_kernel(
    q_ptr,
    scales_ptr,
    zero_points_ptr,
    source_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    S_OUT: tl.constexpr,
    RESIDUAL_S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    D_PACKED: tl.constexpr,
    SCALE_D: tl.constexpr,
    CURRENT_START: tl.constexpr,
    RESIDUAL_START: tl.constexpr,
    UNIT_LEN: tl.constexpr,
    NUM_BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ASYMMETRIC: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d = tl.program_id(2)

    offsets_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offsets_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offsets_k = tl.arange(0, BLOCK_K)
    valid = (offsets_s[:, None] < UNIT_LEN) & (offsets_d[None, :] < D)

    q_s = RESIDUAL_START + offsets_s
    if NUM_BITS == 8:
        q_offsets = pid_bh * RESIDUAL_S * D + q_s[:, None] * D + offsets_d[None, :]
        q_loaded = tl.load(q_ptr + q_offsets, mask=valid, other=0).to(tl.float32)
        q_value = q_loaded
    else:
        values_per_byte = 8 // NUM_BITS
        q_d = offsets_d // values_per_byte
        lane = offsets_d - q_d * values_per_byte
        shifts = NUM_BITS * (values_per_byte - 1 - lane)
        q_offsets = pid_bh * RESIDUAL_S * D_PACKED + q_s[:, None] * D_PACKED + q_d[None, :]
        q_packed = tl.load(q_ptr + q_offsets, mask=valid, other=0).to(tl.int32)
        q_unsigned = (q_packed >> shifts[None, :]) & ((1 << NUM_BITS) - 1)
        if ASYMMETRIC:
            q_value = q_unsigned.to(tl.float32)
        else:
            q_value = (q_unsigned - ((1 << (NUM_BITS - 1)) - 1)).to(tl.float32)

    scale_d = offsets_d // BLOCK_SIZE
    scale_offsets = pid_bh * RESIDUAL_S * SCALE_D + q_s[:, None] * SCALE_D + scale_d[None, :]
    scales = tl.load(scales_ptr + scale_offsets, mask=valid, other=1.0).to(tl.float32)
    if ASYMMETRIC:
        zero_points = tl.load(zero_points_ptr + scale_offsets, mask=valid, other=0).to(tl.float32)
        residual = (q_value - zero_points) * scales
    else:
        residual = q_value * scales

    source_s = CURRENT_START + offsets_s
    source_offsets = pid_bh * S_OUT * D + source_s[:, None] * D + offsets_k[None, :]
    source = tl.load(
        source_ptr + source_offsets,
        mask=(offsets_s[:, None] < UNIT_LEN) & (offsets_k[None, :] < D),
        other=0.0,
    ).to(tl.float32)

    head_id = pid_bh % H
    weight_offsets = head_id * D * D + offsets_k[:, None] * D + offsets_d[None, :]
    weight = tl.load(
        weight_ptr + weight_offsets,
        mask=(offsets_k[:, None] < D) & (offsets_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    prediction = tl.dot(source, weight, out_dtype=tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + head_id * D + offsets_d, mask=offsets_d < D, other=0.0).to(tl.float32)
        prediction += bias[None, :]

    current = prediction + residual
    out_s = CURRENT_START + offsets_s
    out_offsets = pid_bh * S_OUT * D + out_s[:, None] * D + offsets_d[None, :]
    tl.store(output_ptr + out_offsets, current, mask=valid)


@triton.jit
def _s2pp_hybrid_innovation_dequant_kernel(
    q_ptr,
    scales_ptr,
    zero_points_ptr,
    cross_pred_ptr,  # pre-computed cross(K) for all tokens [BH, S, D], FP32
    gamma_ptr,       # per-head innovation gamma [H, D], FP32
    output_ptr,
    S_OUT: tl.constexpr,
    RESIDUAL_S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    D_PACKED: tl.constexpr,
    SCALE_D: tl.constexpr,
    CURRENT_START: tl.constexpr,
    PREV_START: tl.constexpr,
    RESIDUAL_START: tl.constexpr,
    UNIT_LEN: tl.constexpr,
    NUM_BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ASYMMETRIC: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused hybrid-innovation dequant using pre-computed Cross predictions.

    All Cross(K) values are pre-computed once on the Python side via
    einsum(FP32), so this kernel only does: bit-unpack, block-wise dequant,
    load cached cross predictions (current + previous), gamma-weighted
    innovation correction, residual addition, and store.
    """

    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d = tl.program_id(2)

    offsets_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offsets_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    valid = (offsets_s[:, None] < UNIT_LEN) & (offsets_d[None, :] < D)

    # ---- residual dequant ----
    q_s = RESIDUAL_START + offsets_s
    if NUM_BITS == 8:
        q_offsets = pid_bh * RESIDUAL_S * D + q_s[:, None] * D + offsets_d[None, :]
        q_loaded = tl.load(q_ptr + q_offsets, mask=valid, other=0).to(tl.float32)
        q_value = q_loaded
    else:
        values_per_byte = 8 // NUM_BITS
        q_d = offsets_d // values_per_byte
        lane = offsets_d - q_d * values_per_byte
        shifts = NUM_BITS * (values_per_byte - 1 - lane)
        q_offsets = pid_bh * RESIDUAL_S * D_PACKED + q_s[:, None] * D_PACKED + q_d[None, :]
        q_packed = tl.load(q_ptr + q_offsets, mask=valid, other=0).to(tl.int32)
        q_unsigned = (q_packed >> shifts[None, :]) & ((1 << NUM_BITS) - 1)
        if ASYMMETRIC:
            q_value = q_unsigned.to(tl.float32)
        else:
            q_value = (q_unsigned - ((1 << (NUM_BITS - 1)) - 1)).to(tl.float32)

    scale_d = offsets_d // BLOCK_SIZE
    scale_offsets = pid_bh * RESIDUAL_S * SCALE_D + q_s[:, None] * SCALE_D + scale_d[None, :]
    scales = tl.load(scales_ptr + scale_offsets, mask=valid, other=1.0).to(tl.float32)
    if ASYMMETRIC:
        zero_points = tl.load(zero_points_ptr + scale_offsets, mask=valid, other=0).to(tl.float32)
        residual = (q_value - zero_points) * scales
    else:
        residual = q_value * scales

    # ---- pre-computed cross(K_t) for current unit ----
    cur_s = CURRENT_START + offsets_s
    cur_offsets = pid_bh * S_OUT * D + cur_s[:, None] * D + offsets_d[None, :]
    current_cross = tl.load(cross_pred_ptr + cur_offsets, mask=valid, other=0.0).to(tl.float32)

    # ---- pre-computed cross(K_{t-1}) for previous unit ----
    prev_s = PREV_START + offsets_s
    prev_offsets = pid_bh * S_OUT * D + prev_s[:, None] * D + offsets_d[None, :]
    previous_cross = tl.load(cross_pred_ptr + prev_offsets, mask=valid, other=0.0).to(tl.float32)

    # ---- previous reconstructed V (already written to output) ----
    prev_v = tl.load(output_ptr + prev_offsets, mask=valid, other=0.0).to(tl.float32)

    # ---- per-head per-channel innovation gamma ----
    head_id = pid_bh % H
    gamma = tl.load(
        gamma_ptr + head_id * D + offsets_d,
        mask=offsets_d < D, other=0.0,
    ).to(tl.float32)

    # ---- innovation combination:  V̂_t = cross(K_t) + γ·(V̂_{t-1} - cross(K_{t-1})) ----
    innovation = prev_v - previous_cross
    prediction = current_cross + gamma[None, :] * innovation

    current = prediction + residual
    out_s = CURRENT_START + offsets_s
    out_offsets = pid_bh * S_OUT * D + out_s[:, None] * D + offsets_d[None, :]
    tl.store(output_ptr + out_offsets, current, mask=valid)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _block_d(head_dim: int) -> int:
    if head_dim <= 64:
        return 64
    if head_dim <= 128:
        return 128
    return 256


def _is_cuda_tensor(x: torch.Tensor | None) -> bool:
    return isinstance(x, torch.Tensor) and x.is_cuda


def _check_supported_state(packed_state: dict) -> bool:
    if packed_state.get("method") != "s2pp":
        return False
    if "residual_mean" in packed_state or "residual_inv_std" in packed_state:
        return False
    if packed_state.get("use_lloyd_max", False):
        return False
    if str(packed_state.get("predictor_kind", "affine")) not in {
        "affine",
        "identity",
        "temporal_multiframe",
        "cross_kv",
        "hybrid_kv_innovation",
    }:
        return False
    if int(packed_state["num_bits"]) not in (2, 4, 8):
        return False
    if int(packed_state["anchor_bits"]) not in (2, 4, 8):
        return False
    residual_quant_mode = packed_state.get("residual_quant_mode", "symmetric")
    if residual_quant_mode not in {"symmetric", "asym_zero_point"}:
        return False
    anchor_quant = packed_state.get("anchor_quant")
    anchor_scales = packed_state.get("anchor_scales")
    if not (_is_cuda_tensor(anchor_quant) and _is_cuda_tensor(anchor_scales)):
        return False
    residual_quant = packed_state.get("residual_quant")
    residual_scales = packed_state.get("residual_scales")
    if residual_quant is not None and not (_is_cuda_tensor(residual_quant) and _is_cuda_tensor(residual_scales)):
        return False
    if residual_quant_mode == "asym_zero_point" and residual_quant is not None:
        if not _is_cuda_tensor(packed_state.get("residual_zero_points")):
            return False
    return True


def s2pp_dequantize_tensor_triton(
    packed_state: dict,
    output_dtype: torch.dtype = torch.bfloat16,
    source_tensor: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if not _check_supported_state(packed_state):
        return None
    if packed_state.get("predictor_kind") == "cross_kv":
        return _s2pp_dequantize_cross_kv_triton(
            packed_state,
            output_dtype=output_dtype,
            source_tensor=source_tensor,
        )
    if packed_state.get("predictor_kind") == "hybrid_kv_innovation":
        return _s2pp_dequantize_hybrid_innovation_triton(
            packed_state,
            output_dtype=output_dtype,
            source_tensor=source_tensor,
        )
    if packed_state.get("predictor_kind") == "temporal_multiframe":
        return _s2pp_dequantize_temporal_multiframe_triton(
            packed_state,
            output_dtype=output_dtype,
        )

    B, H, S, D = [int(x) for x in packed_state["shape"]]
    block_size = int(packed_state["block_size"])
    anchor_bits = int(packed_state["anchor_bits"])
    num_bits = int(packed_state["num_bits"])
    unit_lengths = [int(x) for x in packed_state["unit_lengths"]]
    if not unit_lengths:
        return None
    if D % block_size != 0:
        return None

    anchor_quant = packed_state["anchor_quant"].contiguous()
    anchor_scales = packed_state["anchor_scales"].contiguous()
    device = anchor_quant.device
    out = torch.empty((B, H, S, D), device=device, dtype=output_dtype)

    BH = B * H
    scale_d = D // block_size
    block_s = 16
    block_d = _block_d(D)

    anchor_len = int(unit_lengths[0])
    anchor_pack = 8 // anchor_bits if anchor_bits in (2, 4) else 1
    anchor_d_packed = D // anchor_pack
    grid_anchor = (BH, _ceil_div(anchor_len, block_s), _ceil_div(D, block_d))
    _s2pp_anchor_dequant_kernel[grid_anchor](
        anchor_quant.reshape(BH, anchor_len, anchor_d_packed),
        anchor_scales.reshape(BH, anchor_len, scale_d),
        out.reshape(BH, S, D),
        S,
        anchor_len,
        D,
        anchor_d_packed,
        scale_d,
        anchor_bits,
        block_size,
        BLOCK_S=block_s,
        BLOCK_D=block_d,
    )

    residual_quant = packed_state.get("residual_quant")
    if residual_quant is None or len(unit_lengths) == 1:
        return out

    residual_scales = packed_state["residual_scales"].contiguous()
    residual_zero_points = packed_state.get("residual_zero_points")
    residual_quant_mode = packed_state.get("residual_quant_mode", "symmetric")
    asymmetric = residual_quant_mode == "asym_zero_point"
    if asymmetric:
        residual_zero_points = residual_zero_points.contiguous()
    else:
        residual_zero_points = residual_scales

    residual_quant = residual_quant.contiguous()
    residual_s = int(residual_quant.shape[2])
    residual_pack = 8 // num_bits if num_bits in (2, 4) else 1
    residual_d_packed = D // residual_pack
    alpha = packed_state.get("alpha")
    beta = packed_state.get("beta")
    rope_cos = packed_state.get("rope_cos")
    rope_sin = packed_state.get("rope_sin")
    # ``mode=identity`` deliberately overrides the predictor kind after loading
    # the parameter file.  Such a state may still carry the file's alpha/beta;
    # matching the Torch path requires treating identity as alpha=1, beta=0.
    has_affine = (
        packed_state.get("predictor_kind") != "identity"
        and _is_cuda_tensor(alpha)
        and _is_cuda_tensor(beta)
    )
    affine_per_head = False
    if has_affine:
        if tuple(alpha.shape) != tuple(beta.shape):
            return None
        if alpha.ndim == 1:
            if tuple(alpha.shape) != (D,):
                return None
        elif alpha.ndim == 2:
            if tuple(alpha.shape) != (H, D):
                return None
            affine_per_head = True
        else:
            return None
    has_rope = _is_cuda_tensor(rope_cos) and _is_cuda_tensor(rope_sin)
    if not has_affine:
        alpha = anchor_scales.new_empty((1,), dtype=torch.float32)
        beta = anchor_scales.new_empty((1,), dtype=torch.float32)
    else:
        alpha = alpha.contiguous()
        beta = beta.contiguous()
    if not has_rope:
        rope_cos = anchor_scales.new_empty((1,), dtype=torch.float32)
        rope_sin = anchor_scales.new_empty((1,), dtype=torch.float32)
    else:
        rope_cos = rope_cos.contiguous()
        rope_sin = rope_sin.contiguous()

    current_start = anchor_len
    residual_start = 0
    prev_start = 0
    out_flat = out.reshape(BH, S, D)
    q_flat = residual_quant.reshape(BH, residual_s, residual_d_packed)
    scales_flat = residual_scales.reshape(BH, residual_s, scale_d)
    zp_flat = residual_zero_points.reshape(BH, residual_s, scale_d)
    for unit_len in unit_lengths[1:]:
        unit_len = int(unit_len)
        grid = (BH, _ceil_div(unit_len, block_s), _ceil_div(D, block_d))
        _s2pp_residual_dequant_kernel[grid](
            q_flat,
            scales_flat,
            zp_flat,
            alpha,
            beta,
            rope_cos,
            rope_sin,
            out_flat,
            S,
            residual_s,
            H,
            D,
            residual_d_packed,
            scale_d,
            current_start,
            prev_start,
            residual_start,
            unit_len,
            num_bits,
            block_size,
            HAS_AFFINE=has_affine,
            AFFINE_PER_HEAD=affine_per_head,
            HAS_ROPE=has_rope,
            ASYMMETRIC=asymmetric,
            BLOCK_S=block_s,
            BLOCK_D=block_d,
        )
        prev_start = current_start
        current_start += unit_len
        residual_start += unit_len

    return out


def _s2pp_dequantize_temporal_multiframe_triton(
    packed_state: dict,
    output_dtype: torch.dtype,
) -> torch.Tensor | None:
    history_weights = packed_state.get("history_weights")
    history_bias = packed_state.get("history_bias")
    history_length = packed_state.get("history_length")
    if (
        not _is_cuda_tensor(history_weights)
        or not _is_cuda_tensor(history_bias)
        or history_length is None
    ):
        return None

    B, H, S, D = [int(x) for x in packed_state["shape"]]
    history_length = int(history_length)
    if history_length < 2:
        return None
    if history_weights.ndim == 3:
        if tuple(history_weights.shape) != (
            history_length,
            history_length,
            D,
        ) or tuple(history_bias.shape) != (history_length, D):
            return None
        shared_history = True
    elif history_weights.ndim == 4:
        if tuple(history_weights.shape) != (
            history_length,
            history_length,
            H,
            D,
        ) or tuple(history_bias.shape) != (history_length, H, D):
            return None
        shared_history = False
    else:
        return None

    block_size = int(packed_state["block_size"])
    anchor_bits = int(packed_state["anchor_bits"])
    num_bits = int(packed_state["num_bits"])
    unit_lengths = [int(x) for x in packed_state["unit_lengths"]]
    if not unit_lengths or D % block_size != 0:
        return None
    predictor_stride = int(packed_state["predictor_stride"])
    if int(unit_lengths[0]) != min(predictor_stride, S):
        return None

    anchor_quant = packed_state["anchor_quant"].contiguous()
    anchor_scales = packed_state["anchor_scales"].contiguous()
    out = torch.empty(
        (B, H, S, D), device=anchor_quant.device, dtype=output_dtype
    )

    BH = B * H
    scale_d = D // block_size
    block_s = 16
    block_d = _block_d(D)

    anchor_len = int(unit_lengths[0])
    anchor_pack = 8 // anchor_bits if anchor_bits in (2, 4) else 1
    anchor_d_packed = D // anchor_pack
    grid_anchor = (BH, _ceil_div(anchor_len, block_s), _ceil_div(D, block_d))
    _s2pp_anchor_dequant_kernel[grid_anchor](
        anchor_quant.reshape(BH, anchor_len, anchor_d_packed),
        anchor_scales.reshape(BH, anchor_len, scale_d),
        out.reshape(BH, S, D),
        S,
        anchor_len,
        D,
        anchor_d_packed,
        scale_d,
        anchor_bits,
        block_size,
        BLOCK_S=block_s,
        BLOCK_D=block_d,
    )

    residual_quant = packed_state.get("residual_quant")
    if residual_quant is None or len(unit_lengths) == 1:
        return out
    residual_scales = packed_state["residual_scales"].contiguous()
    residual_zero_points = packed_state.get("residual_zero_points")
    asymmetric = (
        packed_state.get("residual_quant_mode", "symmetric")
        == "asym_zero_point"
    )
    if asymmetric:
        residual_zero_points = residual_zero_points.contiguous()
    else:
        residual_zero_points = residual_scales

    residual_quant = residual_quant.contiguous()
    residual_s = int(residual_quant.shape[2])
    residual_pack = 8 // num_bits if num_bits in (2, 4) else 1
    residual_d_packed = D // residual_pack
    out_flat = out.reshape(BH, S, D)
    q_flat = residual_quant.reshape(BH, residual_s, residual_d_packed)
    scales_flat = residual_scales.reshape(BH, residual_s, scale_d)
    zp_flat = residual_zero_points.reshape(BH, residual_s, scale_d)

    current_start = anchor_len
    residual_start = 0
    for unit_index, unit_len in enumerate(unit_lengths[1:], start=1):
        unit_len = int(unit_len)
        order = min(unit_index, history_length)
        selected_weights = history_weights[order - 1, :order]
        selected_bias = history_bias[order - 1]
        if shared_history:
            selected_weights = selected_weights[:, None, :].expand(
                order, H, D
            )
            selected_bias = selected_bias[None, :].expand(H, D)
        selected_weights = selected_weights.contiguous()
        selected_bias = selected_bias.contiguous()
        grid = (BH, _ceil_div(unit_len, block_s), _ceil_div(D, block_d))
        _s2pp_multiframe_residual_dequant_kernel[grid](
            q_flat,
            scales_flat,
            zp_flat,
            selected_weights,
            selected_bias,
            out_flat,
            S,
            residual_s,
            H,
            D,
            residual_d_packed,
            scale_d,
            current_start,
            residual_start,
            predictor_stride,
            order,
            unit_len,
            num_bits,
            block_size,
            ASYMMETRIC=asymmetric,
            BLOCK_S=block_s,
            BLOCK_D=block_d,
        )
        current_start += unit_len
        residual_start += unit_len
    return out


def _s2pp_dequantize_cross_kv_triton(
    packed_state: dict,
    output_dtype: torch.dtype,
    source_tensor: torch.Tensor | None,
) -> torch.Tensor | None:
    if source_tensor is None or not _is_cuda_tensor(source_tensor):
        return None
    cross_weight = packed_state.get("cross_weight")
    if not _is_cuda_tensor(cross_weight) or cross_weight.ndim != 3:
        return None
    cross_bias = packed_state.get("cross_bias")
    has_bias = _is_cuda_tensor(cross_bias)

    B, H, S, D = [int(x) for x in packed_state["shape"]]
    if tuple(source_tensor.shape) != (B, H, S, D):
        return None
    if tuple(cross_weight.shape) != (H, D, D):
        return None
    if has_bias and tuple(cross_bias.shape) != (H, D):
        return None

    block_size = int(packed_state["block_size"])
    anchor_bits = int(packed_state["anchor_bits"])
    num_bits = int(packed_state["num_bits"])
    unit_lengths = [int(x) for x in packed_state["unit_lengths"]]
    if not unit_lengths or D % block_size != 0:
        return None

    anchor_quant = packed_state["anchor_quant"].contiguous()
    anchor_scales = packed_state["anchor_scales"].contiguous()
    out = torch.empty((B, H, S, D), device=anchor_quant.device, dtype=output_dtype)

    BH = B * H
    scale_d = D // block_size
    block_s = 16
    block_d = 32 if D >= 128 else _block_d(D)

    anchor_len = int(unit_lengths[0])
    anchor_pack = 8 // anchor_bits if anchor_bits in (2, 4) else 1
    anchor_d_packed = D // anchor_pack
    grid_anchor = (BH, _ceil_div(anchor_len, block_s), _ceil_div(D, _block_d(D)))
    _s2pp_anchor_dequant_kernel[grid_anchor](
        anchor_quant.reshape(BH, anchor_len, anchor_d_packed),
        anchor_scales.reshape(BH, anchor_len, scale_d),
        out.reshape(BH, S, D),
        S,
        anchor_len,
        D,
        anchor_d_packed,
        scale_d,
        anchor_bits,
        block_size,
        BLOCK_S=block_s,
        BLOCK_D=_block_d(D),
    )

    residual_quant = packed_state.get("residual_quant")
    if residual_quant is None or len(unit_lengths) == 1:
        return out

    residual_scales = packed_state["residual_scales"].contiguous()
    residual_zero_points = packed_state.get("residual_zero_points")
    residual_quant_mode = packed_state.get("residual_quant_mode", "symmetric")
    asymmetric = residual_quant_mode == "asym_zero_point"
    if asymmetric:
        residual_zero_points = residual_zero_points.contiguous()
    else:
        residual_zero_points = residual_scales

    residual_quant = residual_quant.contiguous()
    source_tensor = source_tensor.contiguous()
    cross_weight = cross_weight.contiguous()
    if not has_bias:
        cross_bias = anchor_scales.new_empty((1,), dtype=torch.float32)
    else:
        cross_bias = cross_bias.contiguous()

    residual_s = int(residual_quant.shape[2])
    residual_pack = 8 // num_bits if num_bits in (2, 4) else 1
    residual_d_packed = D // residual_pack
    out_flat = out.reshape(BH, S, D)
    source_flat = source_tensor.reshape(BH, S, D)
    q_flat = residual_quant.reshape(BH, residual_s, residual_d_packed)
    scales_flat = residual_scales.reshape(BH, residual_s, scale_d)
    zp_flat = residual_zero_points.reshape(BH, residual_s, scale_d)

    current_start = anchor_len
    residual_start = 0
    for unit_len in unit_lengths[1:]:
        unit_len = int(unit_len)
        grid = (BH, _ceil_div(unit_len, block_s), _ceil_div(D, block_d))
        _s2pp_cross_kv_residual_dequant_kernel[grid](
            q_flat,
            scales_flat,
            zp_flat,
            source_flat,
            cross_weight,
            cross_bias,
            out_flat,
            S,
            residual_s,
            H,
            D,
            residual_d_packed,
            scale_d,
            current_start,
            residual_start,
            unit_len,
            num_bits,
            block_size,
            HAS_BIAS=has_bias,
            ASYMMETRIC=asymmetric,
            BLOCK_S=block_s,
            BLOCK_D=block_d,
            BLOCK_K=D,
        )
        current_start += unit_len
        residual_start += unit_len

    return out


def _s2pp_dequantize_hybrid_innovation_triton(
    packed_state: dict,
    output_dtype: torch.dtype,
    source_tensor: torch.Tensor | None,
) -> torch.Tensor | None:
    """Triton hybrid-innovation dequantization with pre-computed Cross(K).

    Cross predictions are computed once for all tokens via FP32 einsum
    (matching the Torch quantize-prediction precision), then passed to the
    fused kernel which does bit-unpack, block-wise dequant, innovation
    combination, and store in one launch per residual unit.
    """
    # ---- validate mandatory inputs ----
    if source_tensor is None or not _is_cuda_tensor(source_tensor):
        return None
    cross_weight = packed_state.get("cross_weight")
    if not _is_cuda_tensor(cross_weight) or cross_weight.ndim != 3:
        return None
    cross_bias = packed_state.get("cross_bias")
    has_bias = _is_cuda_tensor(cross_bias)

    gamma = packed_state.get("innovation_gamma")
    if not _is_cuda_tensor(gamma):
        return None

    B, H, S, D = [int(x) for x in packed_state["shape"]]
    if tuple(source_tensor.shape) != (B, H, S, D):
        return None
    if tuple(cross_weight.shape) != (H, D, D):
        return None
    if has_bias and tuple(cross_bias.shape) != (H, D):
        return None

    # ---- reject unsupported features (kernel does not implement them) ----
    # NOTE: the same limitation exists in _s2pp_cross_kv_residual_dequant_kernel;
    # these guards are explicit here so hybrid fails safely rather than silently
    # producing wrong values.
    if "residual_mean" in packed_state or "residual_inv_std" in packed_state:
        return None  # per-channel residual normalization not yet fused
    if packed_state.get("use_lloyd_max", False):
        return None  # Lloyd-Max quantization not yet fused
    if packed_state.get("rope_cos") is not None:
        return None  # RoPE is not applicable to Cross-KV (source is pre-RoPE K)

    # ---- normalize gamma to per-head [H, D] ----
    if gamma.ndim == 1:
        if gamma.shape[0] != D:
            return None
        gamma = gamma.view(1, D).expand(H, D).contiguous()
    elif gamma.ndim == 2:
        if tuple(gamma.shape) != (H, D):
            return None
    else:
        return None

    block_size = int(packed_state["block_size"])
    anchor_bits = int(packed_state["anchor_bits"])
    num_bits = int(packed_state["num_bits"])
    unit_lengths = [int(x) for x in packed_state["unit_lengths"]]
    if not unit_lengths or D % block_size != 0:
        return None

    anchor_quant = packed_state["anchor_quant"].contiguous()
    anchor_scales = packed_state["anchor_scales"].contiguous()
    device = anchor_quant.device
    out = torch.empty((B, H, S, D), device=device, dtype=output_dtype)

    BH = B * H
    scale_d = D // block_size
    block_s = 16
    block_d = 32 if D >= 128 else _block_d(D)

    # ---- anchor ----
    anchor_len = int(unit_lengths[0])
    anchor_pack = 8 // anchor_bits if anchor_bits in (2, 4) else 1
    anchor_d_packed = D // anchor_pack
    grid_anchor = (BH, _ceil_div(anchor_len, block_s), _ceil_div(D, _block_d(D)))
    _s2pp_anchor_dequant_kernel[grid_anchor](
        anchor_quant.reshape(BH, anchor_len, anchor_d_packed),
        anchor_scales.reshape(BH, anchor_len, scale_d),
        out.reshape(BH, S, D),
        S,
        anchor_len,
        D,
        anchor_d_packed,
        scale_d,
        anchor_bits,
        block_size,
        BLOCK_S=block_s,
        BLOCK_D=_block_d(D),
    )

    residual_quant = packed_state.get("residual_quant")
    if residual_quant is None or len(unit_lengths) == 1:
        return out

    # ---- pre-compute all Cross(K) predictions in one FP32 einsum ----
    # This matches the precision used by the Torch quantize path
    # (s2pp_quantize_tensor: prediction = current.float() gives FP32).
    # One batched call replaces 2 × N_unit × tl.dot calls in the old kernel.
    source_flat = source_tensor.reshape(BH, S, D).float()
    cross_weight_f32 = cross_weight.float()
    cross_bias_f32 = cross_bias.float() if has_bias else None
    # source_4d: [BH, S, D] -> [B, H, S, D]
    source_4d = source_flat.view(B, H, S, D)
    # cross_pred[b,h,s,:] = source_4d[b,h,s,:] @ cross_weight[h,:,:] (+ bias[h,:])
    cross_pred_4d = torch.einsum("bhsd,hde->bhse", source_4d, cross_weight_f32)
    if has_bias:
        cross_pred_4d = cross_pred_4d + cross_bias_f32.view(1, H, 1, D)
    # Flatten to [BH, S, D]
    cross_pred_flat = cross_pred_4d.reshape(BH, S, D).contiguous()

    # ---- residual dequant params ----
    residual_scales = packed_state["residual_scales"].contiguous()
    residual_zero_points = packed_state.get("residual_zero_points")
    residual_quant_mode = packed_state.get("residual_quant_mode", "symmetric")
    asymmetric = residual_quant_mode == "asym_zero_point"
    if asymmetric:
        residual_zero_points = residual_zero_points.contiguous()
    else:
        residual_zero_points = residual_scales

    residual_quant = residual_quant.contiguous()
    gamma = gamma.contiguous()

    residual_s = int(residual_quant.shape[2])
    residual_pack = 8 // num_bits if num_bits in (2, 4) else 1
    residual_d_packed = D // residual_pack
    out_flat = out.reshape(BH, S, D)
    q_flat = residual_quant.reshape(BH, residual_s, residual_d_packed)
    scales_flat = residual_scales.reshape(BH, residual_s, scale_d)
    zp_flat = residual_zero_points.reshape(BH, residual_s, scale_d)

    current_start = anchor_len
    prev_start = 0
    residual_start = 0
    for unit_len in unit_lengths[1:]:
        unit_len = int(unit_len)
        grid = (BH, _ceil_div(unit_len, block_s), _ceil_div(D, block_d))
        _s2pp_hybrid_innovation_dequant_kernel[grid](
            q_flat,
            scales_flat,
            zp_flat,
            cross_pred_flat,
            gamma,
            out_flat,
            S,
            residual_s,
            H,
            D,
            residual_d_packed,
            scale_d,
            current_start,
            prev_start,
            residual_start,
            unit_len,
            num_bits,
            block_size,
            ASYMMETRIC=asymmetric,
            BLOCK_S=block_s,
            BLOCK_D=block_d,
        )
        prev_start = current_start
        current_start += unit_len
        residual_start += unit_len

    return out
