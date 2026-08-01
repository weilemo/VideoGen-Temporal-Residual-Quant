from __future__ import annotations

from functools import lru_cache
import os
import warnings
from typing import Optional, Sequence

import numpy as np
import torch


_S2PP_TRITON_WARNED = False
_S2PP_CROSS_KV_DEVICE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor | None]] = {}
_S2PP_DEVICE_PARAMETER_CACHE: dict[tuple, torch.Tensor] = {}


_CROSS_PREDICTOR_KINDS = {
    "cross_kv",
    "cross_kv_allhead_low_rank",
    "hybrid_kv_innovation",
    "hybrid_kv_allhead_low_rank",
}


def _is_cross_predictor_kind(kind: str) -> bool:
    return str(kind) in _CROSS_PREDICTOR_KINDS


def s2pp_quantize_tensor(
    tensor: torch.Tensor,
    num_bits: int,
    block_size: int,
    anchor_bits: int = 8,
    predictor_stride: int = 1560,
    affine_path: str = "",
    mode: str = "affine",
    scale_precision: torch.dtype = torch.bfloat16,
    residual_quant_mode: str = "asym_zero_point",
    use_lloyd_max: bool = False,
    source_tensor: Optional[torch.Tensor] = None,
    source_state: Optional[dict] = None,
    layer_idx: Optional[int] = None,
    parameter_role: Optional[str] = None,
    return_debug_reconstruction: bool = False,
) -> dict:
    """
    Predictive residual quantization for a selected QVG cache span.

    Supports S2-Opt v2 predictor variants (block_var, error_feedback, ar2),
    per-channel residual normalisation, and Lloyd-Max quantisation levels.
    All extra features are auto-detected from the .npz file contents.
    """
    if tensor.ndim != 4:
        raise ValueError(f"expected [B, H, S, D], got shape={tuple(tensor.shape)}")
    mode = _normalize_predictor_mode(mode)
    # extend accepted modes for S2-Opt v2
    _valid_modes = {
        "affine", "identity", "rope_affine", "rope_rotation_affine",
        "block_var", "error_feedback", "ar2", "cross_kv",
    }
    if mode not in _valid_modes:
        raise ValueError(f"unsupported s2pp mode: {mode}")

    B, H, S, D = tensor.shape
    requested_predictor_stride = int(predictor_stride)
    predictor_stride = _validate_stride(requested_predictor_stride, S)
    block_size = _validate_block_size(block_size, D)
    num_bits = _validate_bits(num_bits)
    anchor_bits = _validate_bits(anchor_bits)
    scale_precision = _resolve_scale_precision(scale_precision)
    residual_quant_mode = _validate_residual_quant_mode(residual_quant_mode)

    # --- load predictor + quantiser params ---
    # Identity prediction has no learned parameters.  Ignoring affine_path here
    # keeps the core API independent of unrelated or stale predictor files, not
    # just the Self-Forcing shell entry point.
    predictor_path = "" if mode == "identity" else affine_path
    pred_params = _load_predictor_params(
        predictor_path,
        tensor.device,
        D,
        layer_idx,
        parameter_role=parameter_role,
        num_heads=H,
    )
    pred_kind = pred_params.get("predictor_kind", "affine")
    fitted_stride = pred_params.get("predictor_stride")
    if fitted_stride is not None and int(fitted_stride) != requested_predictor_stride:
        raise ValueError(
            f"predictor was fitted for stride={int(fitted_stride)}, "
            f"but runtime requested stride={requested_predictor_stride}"
        )

    if mode == "identity":
        pred_params["predictor_kind"] = "identity"
        pred_kind = "identity"
    elif mode == "error_feedback":
        raise NotImplementedError(
            "error_feedback predictor mode is disabled: the decoder cannot "
            "reconstruct the encoder's error state. The encoder computes "
            "prev_error = x[t] - x_hat[t] (true quantization error from the "
            "original input), but the decoder only has access to reconstructed "
            "values and computes prev_slice - current instead — a different "
            "quantity. Disabled until a decoder-reconstructable formulation "
            "is implemented. See CODE_REVIEW.md §3.2."
        )
    elif mode == "cross_kv":
        pred_kind = str(pred_params.get("predictor_kind", "cross_kv"))
        if not _is_cross_predictor_kind(pred_kind):
            pred_kind = "cross_kv"
            pred_params["predictor_kind"] = pred_kind
        if (
            pred_params.get("cross_weight") is None
            and pred_params.get("cross_input_factor") is None
        ):
            raise ValueError(
                "s2pp mode cross_kv requires dense cross_weight or low-rank "
                f"cross_input_factor in affine_path={affine_path!r}"
            )
    elif mode in ("rope_affine", "rope_rotation_affine"):
        if pred_params.get("cos") is None or pred_params.get("sin") is None:
            raise ValueError(
                f"s2pp mode {mode} requires cos/sin in "
                f"affine_path={affine_path!r}"
            )

        # --- fail-closed: ALL metadata fields must be present ---
        _REQUIRED_ROTATION_METADATA = {
            "parameter_role",
            "coordinate_system",
            "predictor_mode",
            "predictor_stride",
            "fit_quant_bits",
            "fit_group_size",
            "fit_anchor_bits",
            "fit_mode",
            "head_dim",
        }
        missing = sorted(
            k for k in _REQUIRED_ROTATION_METADATA
            if pred_params.get(k) is None
        )
        if missing:
            raise ValueError(
                f"Rotation parameter file lacks required metadata: "
                f"{missing}. Legacy/post-RoPE parameter files are not "
                f"accepted. Re-fit using --mode rope_affine on a "
                f"production pre-RoPE K trace with matching quant config "
                f"(stride={requested_predictor_stride}, "
                f"int{num_bits}, group_size={block_size}, "
                f"anchor_bits={anchor_bits})."
            )

        # --- strict value checks (all fields guaranteed non-None) ---
        if str(pred_params["parameter_role"]) != str(parameter_role):
            raise ValueError(
                f"Rotation parameter role mismatch: "
                f"file={pred_params['parameter_role']!r}, "
                f"runtime={parameter_role!r}"
            )

        if str(pred_params["coordinate_system"]) != "pre_rope_norm_k":
            raise ValueError(
                f"K rotation must be fitted on pre_rope_norm_k, "
                f"got coordinate_system="
                f"{pred_params['coordinate_system']!r}. Old post-RoPE "
                f"parameters are incompatible with production pre-RoPE "
                f"cache — re-fit on production trace."
            )

        if str(pred_params["fit_mode"]) not in {"open_loop", "closed_loop"}:
            raise ValueError(
                f"Rotation fit_mode must be open_loop or closed_loop, "
                f"got {pred_params['fit_mode']!r}"
            )

        # stride: use requested value (before min() clipping) for comparison,
        # since the file was fitted at a specific stride regardless of S.
        if int(pred_params["predictor_stride"]) != int(requested_predictor_stride):
            raise ValueError(
                f"Rotation stride mismatch: "
                f"file={int(pred_params['predictor_stride'])}, "
                f"runtime={int(requested_predictor_stride)}"
            )

        if int(pred_params["fit_quant_bits"]) != int(num_bits):
            raise ValueError(
                f"Rotation parameter was fitted for "
                f"quant_bits={int(pred_params['fit_quant_bits'])}, "
                f"but runtime uses num_bits={int(num_bits)}"
            )

        if int(pred_params["fit_group_size"]) != int(block_size):
            raise ValueError(
                f"Rotation parameter was fitted for "
                f"group_size={int(pred_params['fit_group_size'])}, "
                f"but runtime uses block_size={int(block_size)}"
            )

        if int(pred_params["fit_anchor_bits"]) != int(anchor_bits):
            raise ValueError(
                f"Rotation parameter was fitted for "
                f"anchor_bits={int(pred_params['fit_anchor_bits'])}, "
                f"but runtime uses anchor_bits={int(anchor_bits)}"
            )

        if int(pred_params["head_dim"]) != int(D):
            raise ValueError(
                f"Rotation parameter was fitted for "
                f"head_dim={int(pred_params['head_dim'])}, "
                f"but runtime head_dim={int(D)}"
            )

        # num_heads: optional but validated if present
        fit_num_heads = pred_params.get("num_heads")
        if fit_num_heads is not None and int(fit_num_heads) != int(H):
            raise ValueError(
                f"Rotation parameter was fitted for "
                f"num_heads={int(fit_num_heads)}, "
                f"but runtime num_heads={int(H)}"
            )

    roped = mode in ("rope_affine", "rope_rotation_affine")
    rope_cos = pred_params.get("cos") if roped else None
    rope_sin = pred_params.get("sin") if roped else None

    # --- per-channel norm parameters ---
    norm_mean = pred_params.get("residual_mean")
    norm_inv_std = pred_params.get("residual_inv_std")
    norm_std = pred_params.get("residual_std")
    do_norm = norm_mean is not None and norm_inv_std is not None

    cross_source = None
    previous_cross_source = None
    previous_cross_prediction = None
    reconstruction_history: list[torch.Tensor] = []
    if _is_cross_predictor_kind(pred_kind):
        cross_source = _resolve_cross_source_tensor(
            source_tensor=source_tensor,
            source_state=source_state,
            expected_shape=(B, H, S, D),
            device=tensor.device,
            output_dtype=tensor.dtype,
        )

    # --- Lloyd-Max levels (auto-select by bit-width) ---
    lloyd_levels = None
    if use_lloyd_max:
        # try bit-specific key first, fall back to generic
        bit_key = f"lloyd_levels_{num_bits}bit"
        lloyd_levels = pred_params.get(bit_key)
        if lloyd_levels is None:
            lloyd_levels = pred_params.get("lloyd_levels")
        if lloyd_levels is not None and lloyd_levels.numel() != (1 << num_bits):
            lloyd_levels = None

    # --- anchor (unchanged) ---
    anchor_len = min(predictor_stride, S)
    anchor = tensor[:, :, :anchor_len, :]
    if use_lloyd_max and lloyd_levels is not None and residual_quant_mode == "symmetric":
        anchor_quant, anchor_scales = _quantize_blockwise(
            anchor, bits=anchor_bits, block_size=block_size, scale_precision=scale_precision,
        )
    else:
        anchor_quant, anchor_scales = _quantize_blockwise(
            anchor, bits=anchor_bits, block_size=block_size, scale_precision=scale_precision,
        )
    prev_recon = _dequantize_blockwise(
        anchor_quant, anchor_scales, bits=anchor_bits, block_size=block_size,
        head_dim=D, output_dtype=tensor.dtype,
    )
    reconstruction_history = [prev_recon]
    _debug_recon_parts = [prev_recon] if return_debug_reconstruction else []
    if pred_kind == "hybrid_kv_innovation":
        previous_cross_source = cross_source[:, :, :anchor_len, :]
        if anchor_len < S:
            previous_cross_prediction = _predict_cross_kv(
                previous_cross_source,
                pred_params.get("cross_weight"),
                pred_params.get("cross_bias"),
            )

    residual_quant_parts = []
    residual_scale_parts = []
    residual_zero_point_parts = []
    unit_lengths = [anchor_len]

    # --- multi-step predictor state ---
    prev_error = None   # for error_feedback: x[t-1] - x_hat[t-1]
    prev2_recon = None  # for ar2: x_hat[t-2]

    offset = anchor_len
    while offset < S:
        unit_len = min(predictor_stride, S - offset)
        current = tensor[:, :, offset:offset + unit_len, :]

        if pred_kind in {"cross_kv", "cross_kv_allhead_low_rank"}:
            source_slice = cross_source[:, :, offset:offset + unit_len, :]
            prediction = _predict(source_slice, pred_params)
            prev_slice = None
        elif pred_kind == "hybrid_kv_allhead_low_rank":
            source_slice = cross_source[:, :, offset:offset + unit_len, :]
            prev_slice = prev_recon[:, :, :unit_len, :]
            prediction = _predict_hybrid_kv(source_slice, prev_slice, pred_params)
        elif pred_kind == "hybrid_kv_innovation":
            source_slice = cross_source[:, :, offset:offset + unit_len, :]
            previous_source_slice = previous_cross_source[:, :, :unit_len, :]
            prev_slice = prev_recon[:, :, :unit_len, :]
            current_cross_prediction = _predict_cross_kv(
                source_slice,
                pred_params.get("cross_weight"),
                pred_params.get("cross_bias"),
            )
            if previous_cross_prediction is None:
                raise RuntimeError("innovation Hybrid lacks previous Cross prediction")
            if previous_cross_prediction.shape[2] == unit_len:
                previous_cross_for_unit = previous_cross_prediction
            else:
                # Preserve the original matmul shape and rounding for a final
                # partial unit. Full-size units reuse the prior result.
                previous_cross_for_unit = _predict_cross_kv(
                    previous_source_slice,
                    pred_params.get("cross_weight"),
                    pred_params.get("cross_bias"),
                )
            prediction = _combine_hybrid_innovation(
                current_cross_prediction,
                previous_cross_for_unit,
                prev_slice,
                pred_params.get("innovation_gamma"),
            )
        elif pred_kind == "temporal_multiframe":
            history_slices = [
                item[:, :, :unit_len, :]
                for item in reconstruction_history
            ]
            prediction = _predict_temporal_multiframe(
                history_slices,
                pred_params.get("history_weights"),
                pred_params.get("history_bias"),
            )
            prev_slice = history_slices[0]
        else:
            # slice prev reconstructions to current unit length
            prev_slice = prev_recon[:, :, :unit_len, :]
            prev2_slice = prev2_recon[:, :, :unit_len, :] if prev2_recon is not None else None
            prev_err_slice = prev_error[:, :, :unit_len, :] if prev_error is not None else None

            prediction = _predict(
                prev_slice, pred_params,
                rope_cos=rope_cos, rope_sin=rope_sin,
                prev_error=prev_err_slice, prev2=prev2_slice,
            )
        residual = current.float() - prediction

        # per-channel normalisation before quantisation
        if do_norm:
            residual_norm = _normalize_residual(residual, norm_mean, norm_inv_std)
        else:
            residual_norm = residual

        quant_residual = residual_norm

        if use_lloyd_max and lloyd_levels is not None:
            # Lloyd-Max: quantise to Gaussian-optimal reconstruction levels
            residual_int = _quantize_to_levels(quant_residual, lloyd_levels)
            residual_recon = _dequantize_from_levels(
                residual_int, lloyd_levels, output_dtype=torch.float32,
            )
            # pack Lloyd-Max indices into same format as uniform quant for storage
            residual_quant = residual_int
            # store dummy scales for interface compatibility
            residual_scales = torch.ones(
                *residual.shape[:-1], D // block_size,
                device=tensor.device, dtype=scale_precision,
            )
            residual_zero_point_part = None
        elif residual_quant_mode == "asym_zero_point":
            residual_quant, residual_scales, residual_zero_point_part = _quantize_blockwise_asymmetric(
                quant_residual,
                bits=num_bits, block_size=block_size, scale_precision=scale_precision,
            )
            residual_recon = _dequantize_blockwise_asymmetric(
                residual_quant, residual_scales, residual_zero_point_part,
                bits=num_bits, block_size=block_size, head_dim=D, output_dtype=torch.float32,
            )
        else:
            residual_quant, residual_scales = _quantize_blockwise(
                quant_residual,
                bits=num_bits, block_size=block_size, scale_precision=scale_precision,
            )
            residual_recon = _dequantize_blockwise(
                residual_quant, residual_scales,
                bits=num_bits, block_size=block_size, head_dim=D, output_dtype=torch.float32,
            )
            residual_zero_point_part = None

        # per-channel denormalisation after dequantisation
        if do_norm:
            residual_recon = _denormalize_residual(residual_recon, norm_mean, norm_std)

        prev_recon_new = (prediction + residual_recon).to(dtype=tensor.dtype)

        # --- update multi-step state ---
        if pred_kind == "error_feedback":
            prev_error = (current - prev_recon_new).to(dtype=torch.float32)
        elif pred_kind == "ar2":
            prev2_recon = prev_recon[:, :, :unit_len, :]
        if pred_kind not in {"cross_kv", "cross_kv_allhead_low_rank"}:
            prev_recon = prev_recon_new
        if pred_kind == "temporal_multiframe":
            history_length = int(pred_params["history_length"])
            reconstruction_history = [
                prev_recon_new,
                *reconstruction_history[: history_length - 1],
            ]
        if return_debug_reconstruction:
            _debug_recon_parts.append(prev_recon_new)
        if pred_kind == "hybrid_kv_innovation":
            previous_cross_source = source_slice
            previous_cross_prediction = current_cross_prediction

        residual_quant_parts.append(residual_quant)
        residual_scale_parts.append(residual_scales)
        if residual_zero_point_part is not None:
            residual_zero_point_parts.append(residual_zero_point_part)
        unit_lengths.append(unit_len)
        offset += unit_len

    if residual_quant_parts:
        residual_quant = torch.cat(residual_quant_parts, dim=2)
        residual_scales = torch.cat(residual_scale_parts, dim=2)
        residual_zero_points = (
            torch.cat(residual_zero_point_parts, dim=2)
            if residual_zero_point_parts
            else None
        )
    else:
        residual_quant = None
        residual_scales = None
        residual_zero_points = None

    state = {
        "format": "s2pp",
        "version": 1,
        "method": "s2pp",
        "mode": mode,
        "predictor_mode": mode,
        "predictor_kind": pred_kind,
        "anchor_quant": anchor_quant,
        "anchor_scales": anchor_scales,
        "residual_quant": residual_quant,
        "residual_scales": residual_scales,
        "residual_zero_points": residual_zero_points,
        "residual_quant_mode": residual_quant_mode,
        "unit_lengths": unit_lengths,
        "num_bits": int(num_bits),
        "anchor_bits": int(anchor_bits),
        "block_size": int(block_size),
        "predictor_stride": int(predictor_stride),
        "shape": (int(B), int(H), int(S), int(D)),
        "scale_precision": str(scale_precision).replace("torch.", ""),
        "use_lloyd_max": use_lloyd_max,
        "dependency": (
            "previous_reconstructions"
            if pred_kind == "temporal_multiframe"
            else "previous_reconstruction"
        ),
    }
    if layer_idx is not None:
        state["layer_idx"] = int(layer_idx)
    if _is_cross_predictor_kind(pred_kind):
        state["cross_affine_path"] = affine_path
    # store predictor params needed for dequant
    _store_pred_params_to_state(state, pred_params, tensor.device, roped, do_norm, use_lloyd_max)
    if (
        _is_cross_predictor_kind(pred_kind)
        and source_state is not None
        and os.environ.get("S2PP_STORE_CROSS_SOURCE_STATE", "0").strip().lower() in {"1", "true", "yes", "on"}
    ):
        state["cross_source_state"] = source_state
    if return_debug_reconstruction and _debug_recon_parts:
        state["_debug_reconstruction"] = torch.cat(
            _debug_recon_parts, dim=2
        )
    return state


def _store_pred_params_to_state(
    state: dict, pred_params: dict, device: torch.device,
    roped: bool, do_norm: bool, use_lloyd: bool,
) -> None:
    """Copy predictor/quantiser params into state dict for dequantisation."""
    alpha = pred_params.get("alpha")
    beta = pred_params.get("beta")
    if alpha is not None and beta is not None:
        state["alpha"] = alpha.to(device=device, dtype=torch.float32)
        state["beta"] = beta.to(device=device, dtype=torch.float32)
    skip_cross_params = (
        _is_cross_predictor_kind(state.get("predictor_kind", ""))
        and bool(state.get("cross_affine_path"))
    )
    cross_keys = {
        "cross_weight", "cross_bias", "cross_input_factor", "cross_output_factor",
        "cross_rank", "temporal_weight", "temporal_bias",
    }
    # innovation_gamma is intentionally excluded from cross_keys: it is small
    # ([D] or [H,D]) so always persist it in the packed state, even when
    # cross_affine_path is present.  The fused Triton hybrid-innovation kernel
    # needs it without a separate disk load.
    for key in (
        "gamma", "alpha1", "alpha2", "A", "b",
        "history_weights", "history_bias",
        "cross_weight", "cross_bias", "cross_input_factor", "cross_output_factor",
        "cross_rank", "temporal_weight", "temporal_bias", "innovation_gamma",
    ):
        if skip_cross_params and key in cross_keys:
            continue
        val = pred_params.get(key)
        if val is not None:
            state[key] = val.to(device=device, dtype=torch.float32)
    if pred_params.get("group_size") is not None:
        state["group_size"] = int(pred_params["group_size"])
    if pred_params.get("history_length") is not None:
        state["history_length"] = int(pred_params["history_length"])
    if roped:
        state["rope_cos"] = pred_params["cos"].to(device=device, dtype=torch.float32)
        state["rope_sin"] = pred_params["sin"].to(device=device, dtype=torch.float32)
    if do_norm:
        state["residual_mean"] = pred_params["residual_mean"].to(device=device, dtype=torch.float32)
        state["residual_inv_std"] = pred_params["residual_inv_std"].to(device=device, dtype=torch.float32)
        state["residual_std"] = pred_params["residual_std"].to(device=device, dtype=torch.float32)
    if use_lloyd:
        for key in ("lloyd_levels", "lloyd_levels_2bit", "lloyd_levels_3bit", "lloyd_levels_4bit"):
            val = pred_params.get(key)
            if val is not None:
                state[key] = val.to(device=device, dtype=torch.float32)


def s2pp_dequantize_tensor(
    packed_state: dict,
    output_dtype: torch.dtype = torch.bfloat16,
    source_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    method = packed_state.get("method")
    if method != "s2pp":
        raise ValueError(f"not an s2pp packed state: {method}")
    _validate_s2pp_predictor_state(packed_state)

    if _s2pp_use_triton() and not _is_cross_predictor_kind(
        packed_state.get("predictor_kind", "")
    ):
        triton_output = _try_s2pp_triton_dequantize(packed_state, output_dtype)
        if triton_output is not None:
            return triton_output

    B, H, S, D = packed_state["shape"]
    block_size = int(packed_state["block_size"])
    anchor_bits = int(packed_state["anchor_bits"])
    num_bits = int(packed_state["num_bits"])
    unit_lengths = [int(x) for x in packed_state["unit_lengths"]]

    # rebuild pred_params dict from state (backward-compatible)
    pred_params: dict = {
        "predictor_kind": packed_state.get("predictor_kind", "affine"),
        "alpha": packed_state.get("alpha"),
        "beta": packed_state.get("beta"),
        "gamma": packed_state.get("gamma"),
        "alpha1": packed_state.get("alpha1"),
        "alpha2": packed_state.get("alpha2"),
        "history_weights": packed_state.get("history_weights"),
        "history_bias": packed_state.get("history_bias"),
        "history_length": packed_state.get("history_length"),
        "A": packed_state.get("A"),
        "b": packed_state.get("b"),
        "cross_weight": packed_state.get("cross_weight"),
        "cross_bias": packed_state.get("cross_bias"),
        "cross_input_factor": packed_state.get("cross_input_factor"),
        "cross_output_factor": packed_state.get("cross_output_factor"),
        "cross_rank": packed_state.get("cross_rank"),
        "temporal_weight": packed_state.get("temporal_weight"),
        "temporal_bias": packed_state.get("temporal_bias"),
        "innovation_gamma": packed_state.get("innovation_gamma"),
        "group_size": packed_state.get("group_size"),
    }
    pred_kind = pred_params["predictor_kind"]

    if pred_kind == "error_feedback":
        raise NotImplementedError(
            "error_feedback predictor is disabled: the decoder computes "
            "prev_error = prev_slice - current (line 632-633, the difference "
            "between consecutive reconstructions), which does not match the "
            "encoder's prev_error = x[t] - x_hat[t] (the true quantization error). "
            "See CODE_REVIEW.md §3.2 for details."
        )

    if (
        _is_cross_predictor_kind(pred_kind)
        and pred_params.get("cross_weight") is None
        and pred_params.get("cross_input_factor") is None
    ):
        cross_path = packed_state.get("cross_affine_path", "")
        if not cross_path:
            raise ValueError("cross_kv state lacks cross_weight and cross_affine_path")
        cross_params = _load_predictor_params(
            cross_path,
            packed_state["anchor_quant"].device,
            int(D),
            packed_state.get("layer_idx"),
        )
        for key in (
            "cross_weight", "cross_bias", "cross_input_factor", "cross_output_factor",
            "cross_rank", "temporal_weight", "temporal_bias",
            "innovation_gamma",
        ):
            pred_params[key] = cross_params.get(key)
    if (
        _s2pp_use_triton()
        and pred_kind == "cross_kv"
        and source_tensor is not None
        and isinstance(pred_params.get("cross_weight"), torch.Tensor)
        and pred_params["cross_weight"].ndim == 3
    ):
        triton_state = packed_state.copy()
        triton_state["cross_weight"] = pred_params.get("cross_weight")
        triton_state["cross_bias"] = pred_params.get("cross_bias")
        triton_output = _try_s2pp_triton_dequantize(
            triton_state,
            output_dtype,
            source_tensor=source_tensor,
        )
        if triton_output is not None:
            return triton_output
    if (
        _s2pp_use_triton()
        and pred_kind == "hybrid_kv_innovation"
        and source_tensor is not None
        and isinstance(pred_params.get("cross_weight"), torch.Tensor)
        and pred_params["cross_weight"].ndim == 3
        and isinstance(pred_params.get("innovation_gamma"), torch.Tensor)
    ):
        triton_state = packed_state.copy()
        triton_state["cross_weight"] = pred_params.get("cross_weight")
        triton_state["cross_bias"] = pred_params.get("cross_bias")
        triton_state["innovation_gamma"] = pred_params.get("innovation_gamma")
        triton_output = _try_s2pp_triton_dequantize(
            triton_state,
            output_dtype,
            source_tensor=source_tensor,
        )
        if triton_output is not None:
            return triton_output
    roped = (packed_state.get("rope_cos") is not None)
    rope_cos = packed_state.get("rope_cos")
    rope_sin = packed_state.get("rope_sin")

    do_norm = "residual_mean" in packed_state
    norm_mean = packed_state.get("residual_mean")
    norm_inv_std = packed_state.get("residual_inv_std")
    norm_std = packed_state.get("residual_std")

    use_lloyd = packed_state.get("use_lloyd_max", False)
    # auto-select by bit-width saved in packed_state
    bit_key = f"lloyd_levels_{num_bits}bit"
    lloyd_levels = packed_state.get(bit_key)
    if lloyd_levels is None:
        lloyd_levels = packed_state.get("lloyd_levels")

    anchor = _dequantize_blockwise(
        packed_state["anchor_quant"],
        packed_state["anchor_scales"],
        bits=anchor_bits,
        block_size=block_size,
        head_dim=int(D),
        output_dtype=output_dtype,
    )
    outputs = [anchor]
    prev_recon = anchor
    reconstruction_history = [anchor]

    residual_quant = packed_state.get("residual_quant")
    residual_scales = packed_state.get("residual_scales")
    residual_zero_points = packed_state.get("residual_zero_points")
    residual_quant_mode = packed_state.get("residual_quant_mode", "symmetric")
    residual_offset = 0

    # multi-step state
    prev_error = None
    prev2_recon = None
    current_start = int(unit_lengths[0])
    cross_source = None
    previous_cross_source = None
    previous_cross_prediction = None
    if _is_cross_predictor_kind(pred_kind):
        cross_source = _resolve_cross_source_tensor(
            source_tensor=source_tensor,
            source_state=packed_state.get("cross_source_state"),
            expected_shape=(B, H, S, D),
            device=anchor.device,
            output_dtype=output_dtype,
        )
    if pred_kind == "hybrid_kv_innovation":
        previous_cross_source = cross_source[:, :, :current_start, :]
        if len(unit_lengths) > 1:
            previous_cross_prediction = _predict_cross_kv(
                previous_cross_source,
                pred_params.get("cross_weight"),
                pred_params.get("cross_bias"),
            )

    for unit_len in unit_lengths[1:]:
        q_part = residual_quant[:, :, residual_offset:residual_offset + unit_len, :]
        scale_part = residual_scales[:, :, residual_offset:residual_offset + unit_len, :]
        zero_point_part = (
            residual_zero_points[:, :, residual_offset:residual_offset + unit_len, :]
            if residual_zero_points is not None
            else None
        )
        residual_offset += unit_len

        if pred_kind in {"cross_kv", "cross_kv_allhead_low_rank"}:
            source_slice = cross_source[:, :, current_start:current_start + unit_len, :]
            prediction = _predict(source_slice, pred_params)
            prev_slice = None
        elif pred_kind == "hybrid_kv_allhead_low_rank":
            source_slice = cross_source[:, :, current_start:current_start + unit_len, :]
            prev_slice = prev_recon[:, :, :unit_len, :]
            prediction = _predict_hybrid_kv(source_slice, prev_slice, pred_params)
        elif pred_kind == "hybrid_kv_innovation":
            source_slice = cross_source[:, :, current_start:current_start + unit_len, :]
            previous_source_slice = previous_cross_source[:, :, :unit_len, :]
            prev_slice = prev_recon[:, :, :unit_len, :]
            current_cross_prediction = _predict_cross_kv(
                source_slice,
                pred_params.get("cross_weight"),
                pred_params.get("cross_bias"),
            )
            if previous_cross_prediction is None:
                raise RuntimeError("innovation Hybrid lacks previous Cross prediction")
            if previous_cross_prediction.shape[2] == unit_len:
                previous_cross_for_unit = previous_cross_prediction
            else:
                previous_cross_for_unit = _predict_cross_kv(
                    previous_source_slice,
                    pred_params.get("cross_weight"),
                    pred_params.get("cross_bias"),
                )
            prediction = _combine_hybrid_innovation(
                current_cross_prediction,
                previous_cross_for_unit,
                prev_slice,
                pred_params.get("innovation_gamma"),
            )
        elif pred_kind == "temporal_multiframe":
            history_slices = [
                item[:, :, :unit_len, :]
                for item in reconstruction_history
            ]
            prediction = _predict_temporal_multiframe(
                history_slices,
                pred_params.get("history_weights"),
                pred_params.get("history_bias"),
            )
            prev_slice = history_slices[0]
        else:
            prev_slice = prev_recon[:, :, :unit_len, :]
            prev2_slice = prev2_recon[:, :, :unit_len, :] if prev2_recon is not None else None
            prev_err_slice = prev_error[:, :, :unit_len, :] if prev_error is not None else None

            prediction = _predict(
                prev_slice, pred_params,
                rope_cos=rope_cos, rope_sin=rope_sin,
                prev_error=prev_err_slice, prev2=prev2_slice,
            )

        if use_lloyd and lloyd_levels is not None:
            residual = _dequantize_from_levels(
                q_part, lloyd_levels, output_dtype=torch.float32,
            )
        elif residual_quant_mode == "asym_zero_point" and zero_point_part is not None:
            residual = _dequantize_blockwise_asymmetric(
                q_part, scale_part, zero_point_part,
                bits=num_bits, block_size=block_size,
                head_dim=int(D), output_dtype=torch.float32,
            )
        else:
            residual = _dequantize_blockwise(
                q_part, scale_part,
                bits=num_bits, block_size=block_size,
                head_dim=int(D), output_dtype=torch.float32,
            )

        if do_norm and norm_mean is not None and norm_std is not None:
            residual = _denormalize_residual(residual, norm_mean, norm_std)

        current = (prediction + residual).to(dtype=output_dtype)
        outputs.append(current)

        # update multi-step state
        if pred_kind == "error_feedback":
            prev_error = (prev_slice - current).to(dtype=torch.float32)
        elif pred_kind == "ar2":
            prev2_recon = prev_slice

        if pred_kind not in {"cross_kv", "cross_kv_allhead_low_rank"}:
            prev_recon = current
        if pred_kind == "temporal_multiframe":
            history_length = int(pred_params["history_length"])
            reconstruction_history = [
                current,
                *reconstruction_history[: history_length - 1],
            ]
        if pred_kind == "hybrid_kv_innovation":
            previous_cross_source = source_slice
            previous_cross_prediction = current_cross_prediction
        current_start += int(unit_len)

    return torch.cat(outputs, dim=2).reshape(B, H, S, D)


def _validate_s2pp_predictor_state(packed_state: dict) -> None:
    """Reject incomplete predictor state instead of silently decoding as identity."""
    kind = str(packed_state.get("predictor_kind", "affine"))
    if kind == "affine":
        if packed_state.get("alpha") is None or packed_state.get("beta") is None:
            raise ValueError("S2++ affine state is missing embedded alpha/beta predictor parameters")
    if kind == "temporal_multiframe":
        if (
            packed_state.get("history_weights") is None
            or packed_state.get("history_bias") is None
            or packed_state.get("history_length") is None
        ):
            raise ValueError(
                "S2++ multi-frame state is missing history_weights, "
                "history_bias, or history_length"
            )
    mode = str(packed_state.get("predictor_mode", packed_state.get("mode", "")))
    rope_cos = packed_state.get("rope_cos")
    rope_sin = packed_state.get("rope_sin")
    if (rope_cos is None) != (rope_sin is None):
        raise ValueError("S2++ state must contain both rope_cos and rope_sin")
    if mode in {"rope_affine", "rope_rotation_affine"} and rope_cos is None:
        raise ValueError("S2++ rope_affine state is missing embedded rope_cos/rope_sin")


def _s2pp_use_triton() -> bool:
    value = os.environ.get("S2PP_USE_TRITON", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _s2pp_triton_strict() -> bool:
    # Default strict=True per CODE_REVIEW.md §4.4: formal benchmarks must fail
    # loudly on any fallback.  Set S2PP_TRITON_STRICT=0 for debugging.
    value = os.environ.get("S2PP_TRITON_STRICT", "1").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _warn_s2pp_triton_once(message: str) -> None:
    global _S2PP_TRITON_WARNED
    if _S2PP_TRITON_WARNED:
        return
    _S2PP_TRITON_WARNED = True
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _try_s2pp_triton_dequantize(
    packed_state: dict,
    output_dtype: torch.dtype,
    source_tensor: torch.Tensor | None = None,
) -> torch.Tensor | None:
    unsupported_reason = _s2pp_triton_unsupported_reason(packed_state)
    if unsupported_reason is not None:
        if _s2pp_triton_strict():
            raise RuntimeError(
                "S2PP_USE_TRITON=1 + S2PP_TRITON_STRICT=1: "
                f"Triton S2++ does not support {unsupported_reason}"
            )
        return None

    try:
        try:
            from .s2pp_triton import s2pp_dequantize_tensor_triton
        except ImportError:
            from s2pp_triton import s2pp_dequantize_tensor_triton
    except Exception as exc:
        if _s2pp_triton_strict():
            raise RuntimeError("S2PP_USE_TRITON=1 requested but Triton S2++ import failed") from exc
        _warn_s2pp_triton_once(f"S2++ Triton dequant import failed; falling back to Torch path: {exc}")
        return None

    try:
        result = s2pp_dequantize_tensor_triton(
            packed_state,
            output_dtype=output_dtype,
            source_tensor=source_tensor,
        )
    except Exception as exc:
        if _s2pp_triton_strict():
            raise RuntimeError("S2PP_USE_TRITON=1 requested but Triton S2++ dequantization failed") from exc
        _warn_s2pp_triton_once(f"S2++ Triton dequant failed; falling back to Torch path: {exc}")
        return None

    if result is None and _s2pp_triton_strict():
        raise RuntimeError(
            "S2PP_USE_TRITON=1 + S2PP_TRITON_STRICT=1: Triton S2++ returned None "
            "(unsupported config such as residual norm, Lloyd-Max, or missing "
            f"parameters). predictor_kind={packed_state.get('predictor_kind', '?')}"
        )
    return result


def _s2pp_triton_unsupported_reason(packed_state: dict) -> Optional[str]:
    """Return why a packed state must use the feature-complete Torch decoder."""
    if "residual_mean" in packed_state or "residual_inv_std" in packed_state:
        return "per-channel residual normalization"
    if packed_state.get("use_lloyd_max", False):
        return "Lloyd-Max residual quantization"
    predictor_kind = str(packed_state.get("predictor_kind", "affine"))
    if predictor_kind not in {
        "affine",
        "identity",
        "temporal_multiframe",
        "cross_kv",
        "hybrid_kv_innovation",
    }:
        return f"predictor_kind={predictor_kind!r}"
    # rope_affine / rope_rotation_affine: the Triton decoder does not yet
    # implement learned pair rotation.  Route through the feature-complete
    # Torch path so identity, affine, and rotation all use the same backend
    # during debug and evaluation.
    if (
        predictor_kind == "affine"
        and (
            packed_state.get("rope_cos") is not None
            or packed_state.get("cos") is not None
        )
    ):
        return "learned pair rotation (rope_affine)"
    return None


def _quantize_blockwise(
    x: torch.Tensor,
    bits: int,
    block_size: int,
    scale_precision: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    bits = _validate_bits(bits)
    block_size = _validate_block_size(block_size, x.shape[-1])
    work = x.float()
    qmax = (1 << (bits - 1)) - 1
    grouped = work.reshape(*work.shape[:-1], work.shape[-1] // block_size, block_size)
    scales = _symmetric_scales(grouped, qmax, scale_precision)
    q = torch.round(grouped / scales.float()).clamp(-qmax, qmax).to(torch.int16)
    q = q.reshape(*work.shape)

    if bits in (2, 4):
        q = _pack_lowbit(q, bits)
    else:
        q = q.to(torch.int8)
    return q.contiguous(), scales.squeeze(-1).contiguous()


def _quantize_blockwise_asymmetric(
    x: torch.Tensor,
    bits: int,
    block_size: int,
    scale_precision: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bits = _validate_bits(bits)
    block_size = _validate_block_size(block_size, x.shape[-1])
    work = x.float()
    qmin = 0
    qmax = (1 << bits) - 1
    grouped = work.reshape(*work.shape[:-1], work.shape[-1] // block_size, block_size)
    # Integer zero-points can represent a purely positive/negative range only
    # when zero belongs to that range.  Without this, clamping the zero-point
    # makes same-sign (and especially constant) groups reconstruct near zero.
    scales, zero_points = _asymmetric_scale_and_zero_point(
        grouped, qmin, qmax, scale_precision
    )
    q = torch.round(grouped / scales.float() + zero_points.float()).clamp(qmin, qmax).to(torch.uint8)
    q = q.reshape(*work.shape)

    if bits in (2, 4):
        q = _pack_lowbit_unsigned(q, bits)
    else:
        q = q.to(torch.uint8)
    return q.contiguous(), scales.squeeze(-1).contiguous(), zero_points.squeeze(-1).contiguous()


def _ceil_scale_to_precision(
    required: torch.Tensor,
    scale_precision: torch.dtype,
) -> torch.Tensor:
    """Store a scale without rounding below the required representable range."""
    rounded = required.to(dtype=scale_precision)
    rounded_float = rounded.float()
    upward = torch.nextafter(rounded, torch.full_like(rounded, float("inf")))
    return torch.where(rounded_float < required.float(), upward, rounded)


def _symmetric_scales(
    grouped: torch.Tensor,
    qmax: int,
    scale_precision: torch.dtype,
) -> torch.Tensor:
    required = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / max(qmax, 1)
    return _ceil_scale_to_precision(required, scale_precision)


def _asymmetric_scale_and_zero_point(
    grouped: torch.Tensor,
    qmin: int,
    qmax: int,
    scale_precision: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    zeros = torch.zeros((), device=grouped.device, dtype=grouped.dtype)
    x_min = torch.minimum(grouped.amin(dim=-1, keepdim=True), zeros)
    x_max = torch.maximum(grouped.amax(dim=-1, keepdim=True), zeros)
    base_scale = (x_max - x_min).clamp_min(1e-12) / float(qmax - qmin)
    zero_points_float = torch.round(qmin - x_min / base_scale).clamp(qmin, qmax)

    lower_capacity = zero_points_float - qmin
    upper_capacity = qmax - zero_points_float
    lower_scale = torch.where(
        lower_capacity > 0,
        (-x_min) / lower_capacity.clamp_min(1),
        torch.zeros_like(base_scale),
    )
    upper_scale = torch.where(
        upper_capacity > 0,
        x_max / upper_capacity.clamp_min(1),
        torch.zeros_like(base_scale),
    )
    required_scale = torch.maximum(base_scale, torch.maximum(lower_scale, upper_scale))
    scales = _ceil_scale_to_precision(required_scale.clamp_min(1e-12), scale_precision)
    return scales, zero_points_float.to(torch.uint8)


def _dequantize_blockwise(
    q: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    block_size: int,
    head_dim: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    bits = _validate_bits(bits)
    block_size = _validate_block_size(block_size, head_dim)
    if bits in (2, 4):
        q = _unpack_lowbit(q, bits, head_dim)
    else:
        q = q.to(torch.int16)

    expanded_scales = scales.float().unsqueeze(-1).expand(
        *scales.shape,
        block_size,
    ).reshape(*scales.shape[:-1], head_dim)
    return (q.float() * expanded_scales).to(dtype=output_dtype)


def _dequantize_blockwise_asymmetric(
    q: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    bits: int,
    block_size: int,
    head_dim: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    bits = _validate_bits(bits)
    block_size = _validate_block_size(block_size, head_dim)
    if bits in (2, 4):
        q = _unpack_lowbit_unsigned(q, bits, head_dim)
    else:
        q = q.to(torch.int16)

    expanded_scales = scales.float().unsqueeze(-1).expand(
        *scales.shape,
        block_size,
    ).reshape(*scales.shape[:-1], head_dim)
    expanded_zero_points = zero_points.float().unsqueeze(-1).expand(
        *zero_points.shape,
        block_size,
    ).reshape(*zero_points.shape[:-1], head_dim)
    return ((q.float() - expanded_zero_points) * expanded_scales).to(dtype=output_dtype)


def _pack_lowbit(q_signed: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    q_unsigned = (q_signed + qmax).to(torch.uint8)
    values_per_byte = 8 // bits
    if q_unsigned.shape[-1] % values_per_byte != 0:
        raise ValueError(
            f"head_dim={q_unsigned.shape[-1]} must be divisible by {values_per_byte} for int{bits} packing"
        )
    q_grouped = q_unsigned.reshape(*q_unsigned.shape[:-1], q_unsigned.shape[-1] // values_per_byte, values_per_byte)
    packed = torch.zeros(*q_grouped.shape[:-1], device=q_grouped.device, dtype=torch.uint8)
    for idx in range(values_per_byte):
        shift = bits * (values_per_byte - 1 - idx)
        packed |= q_grouped[..., idx] << shift
    return packed


def _pack_lowbit_unsigned(q_unsigned: torch.Tensor, bits: int) -> torch.Tensor:
    values_per_byte = 8 // bits
    if q_unsigned.shape[-1] % values_per_byte != 0:
        raise ValueError(
            f"head_dim={q_unsigned.shape[-1]} must be divisible by {values_per_byte} for int{bits} packing"
        )
    q_grouped = q_unsigned.to(torch.uint8).reshape(
        *q_unsigned.shape[:-1],
        q_unsigned.shape[-1] // values_per_byte,
        values_per_byte,
    )
    packed = torch.zeros(*q_grouped.shape[:-1], device=q_grouped.device, dtype=torch.uint8)
    for idx in range(values_per_byte):
        shift = bits * (values_per_byte - 1 - idx)
        packed |= q_grouped[..., idx] << shift
    return packed


def _unpack_lowbit(q_packed: torch.Tensor, bits: int, head_dim: int) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    values_per_byte = 8 // bits
    if head_dim % values_per_byte != 0:
        raise ValueError(f"head_dim={head_dim} must be divisible by {values_per_byte} for int{bits} unpacking")
    parts = []
    mask = (1 << bits) - 1
    for idx in range(values_per_byte):
        shift = bits * (values_per_byte - 1 - idx)
        parts.append(((q_packed >> shift) & mask).to(torch.int16) - qmax)
    return torch.stack(parts, dim=-1).reshape(*q_packed.shape[:-1], head_dim)


def _unpack_lowbit_unsigned(q_packed: torch.Tensor, bits: int, head_dim: int) -> torch.Tensor:
    values_per_byte = 8 // bits
    if head_dim % values_per_byte != 0:
        raise ValueError(f"head_dim={head_dim} must be divisible by {values_per_byte} for int{bits} unpacking")
    parts = []
    mask = (1 << bits) - 1
    for idx in range(values_per_byte):
        shift = bits * (values_per_byte - 1 - idx)
        parts.append(((q_packed >> shift) & mask).to(torch.int16))
    return torch.stack(parts, dim=-1).reshape(*q_packed.shape[:-1], head_dim)


def _predict(
    prev: torch.Tensor,
    pred_params: dict,
    rope_cos: torch.Tensor | None = None,
    rope_sin: torch.Tensor | None = None,
    prev_error: torch.Tensor | None = None,
    prev2: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Dispatch to correct predictor implementation based on pred_params["predictor_kind"].

    Legacy signature (alpha, beta as separate tensors) is still supported via
    pred_params dict containing at least "alpha" and "beta".
    """
    kind = pred_params.get("predictor_kind", "affine")
    if kind == "cross_kv":
        return _predict_cross_kv(
            prev,
            pred_params.get("cross_weight"),
            pred_params.get("cross_bias"),
        )
    if kind == "cross_kv_allhead_low_rank":
        return _predict_cross_kv_allhead_low_rank(
            prev,
            pred_params.get("cross_input_factor"),
            pred_params.get("cross_output_factor"),
            pred_params.get("cross_bias"),
            pred_params.get("cross_rank"),
        )
    if kind == "temporal_multiframe":
        return _predict_temporal_multiframe(
            [prev],
            pred_params.get("history_weights"),
            pred_params.get("history_bias"),
        )

    if rope_cos is not None or rope_sin is not None:
        prev = _apply_pair_rotation(prev, rope_cos, rope_sin)

    alpha = pred_params.get("alpha")
    beta = pred_params.get("beta")

    if kind == "identity" or (alpha is None and beta is None):
        if kind != "identity" and (alpha is None and beta is None):
            import warnings
            warnings.warn(
                f"Predictor kind '{kind}' is missing alpha/beta parameters. "
                f"Falling back to identity prediction — this means the actual "
                f"predictor differs from the declared kind. "
                f"Check predictor file path and parameters.",
                RuntimeWarning,
                stacklevel=2,
            )
        return prev.float()

    if kind == "block_var":
        A = pred_params.get("A")
        b = pred_params.get("b")
        group_size = pred_params.get("group_size")
        if A is None or b is None or group_size is None:
            raise ValueError("block_var predictor requires A, b, group_size")
        return _predict_block_var(prev, A, b, group_size)

    if kind == "error_feedback":
        raise NotImplementedError(
            "error_feedback is disabled because its predictor state "
            "is not decoder-reconstructable. The encoder computes "
            "prev_error = x[t] - x_hat[t] (true quantization error from the "
            "original input), but the decoder only has access to reconstructed "
            "values and computes prev_slice - current instead — a different "
            "quantity. Disabled until a decoder-reconstructable formulation "
            "is implemented. See CODE_REVIEW.md §3.2."
        )

    if kind == "ar2":
        alpha1 = pred_params.get("alpha1")
        alpha2 = pred_params.get("alpha2")
        if alpha1 is None or alpha2 is None:
            raise ValueError("ar2 predictor requires alpha1, alpha2")
        if prev2 is None:
            # first step after anchor: only 1 past unit, fall back to AR(1)
            return _predict_affine(prev, alpha1, beta)
        return _predict_ar2(prev, prev2, alpha1, alpha2, beta)

    # default: legacy affine
    return _predict_affine(prev, alpha, beta)


def _predict_affine(
    prev: torch.Tensor,
    alpha: torch.Tensor | None,
    beta: torch.Tensor | None,
) -> torch.Tensor:
    """Legacy per-channel affine: alpha * prev + beta."""
    if alpha is None or beta is None:
        return prev.float()
    if alpha.shape != beta.shape:
        raise ValueError(f"affine alpha/beta shapes differ: {tuple(alpha.shape)} vs {tuple(beta.shape)}")
    if alpha.ndim == 1:
        if alpha.shape[0] != prev.shape[-1]:
            raise ValueError(f"affine [D] parameter does not match D={prev.shape[-1]}: {tuple(alpha.shape)}")
        view_shape = [1] * prev.ndim
        view_shape[-1] = prev.shape[-1]
    elif alpha.ndim == 2 and prev.ndim == 4:
        if tuple(alpha.shape) != (prev.shape[1], prev.shape[-1]):
            raise ValueError(
                f"affine [H,D] parameter must be {(prev.shape[1], prev.shape[-1])}, got {tuple(alpha.shape)}"
            )
        view_shape = [1, prev.shape[1], 1, prev.shape[-1]]
    else:
        raise ValueError(f"affine parameters must be [D] or [H,D], got {tuple(alpha.shape)}")
    return prev.float() * alpha.view(*view_shape) + beta.view(*view_shape)


def _predict_cross_kv(
    source: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if weight is None:
        raise ValueError("cross_kv predictor requires cross_weight")
    work = source.float()
    weight = weight.to(device=source.device, dtype=torch.float32)
    if weight.ndim == 2:
        full_width = work.shape[1] * work.shape[-1]
        if tuple(weight.shape) == (full_width, full_width):
            flat = work.permute(0, 2, 1, 3).reshape(work.shape[0], work.shape[2], full_width)
            pred = torch.matmul(flat, weight).reshape(
                work.shape[0], work.shape[2], work.shape[1], work.shape[-1]
            ).permute(0, 2, 1, 3)
        elif tuple(weight.shape) != (work.shape[-1], work.shape[-1]):
            raise ValueError(
                "cross_kv weight must be per-head shared "
                f"{(work.shape[-1], work.shape[-1])} or all-head {(full_width, full_width)}, "
                f"got {tuple(weight.shape)}"
            )
        else:
            pred = torch.matmul(work, weight)
    elif weight.ndim == 3:
        if weight.shape[0] != work.shape[1] or tuple(weight.shape[1:]) != (work.shape[-1], work.shape[-1]):
            raise ValueError(
                f"cross_kv per-head weight must have shape {(work.shape[1], work.shape[-1], work.shape[-1])}, "
                f"got {tuple(weight.shape)}"
            )
        pred = torch.matmul(
            work.permute(1, 0, 2, 3),
            weight[:, None, :, :],
        ).permute(1, 0, 2, 3)
    else:
        raise ValueError(f"cross_kv weight must be [D,D] or [H,D,D], got {tuple(weight.shape)}")

    if bias is None:
        return pred
    bias = bias.to(device=source.device, dtype=torch.float32)
    if bias.ndim == 1:
        if bias.shape[0] == work.shape[1] * work.shape[-1]:
            return pred + bias.view(1, work.shape[1], 1, work.shape[-1])
        if bias.shape[0] == work.shape[-1]:
            return pred + bias.view(1, 1, 1, work.shape[-1])
        raise ValueError(
            f"cross_kv bias must have D or H*D values, got {tuple(bias.shape)}"
        )
    if bias.ndim == 2:
        if tuple(bias.shape) != (work.shape[1], work.shape[-1]):
            raise ValueError(
                f"cross_kv per-head bias must have shape {(work.shape[1], work.shape[-1])}, "
                f"got {tuple(bias.shape)}"
            )
        return pred + bias.view(1, work.shape[1], 1, work.shape[-1])
    raise ValueError(f"cross_kv bias must be [D] or [H,D], got {tuple(bias.shape)}")


def _predict_cross_kv_allhead_low_rank(
    source: torch.Tensor,
    input_factor: torch.Tensor | None,
    output_factor: torch.Tensor | None,
    bias: torch.Tensor | None,
    rank: torch.Tensor | Optional[int] = None,
) -> torch.Tensor:
    if input_factor is None or output_factor is None or bias is None:
        raise ValueError("all-head low-rank predictor requires both factors and bias")
    work = source.float()
    batch, input_heads, tokens, dim = work.shape
    input_width = input_heads * dim
    left = input_factor.to(device=source.device, dtype=torch.float32)
    right = output_factor.to(device=source.device, dtype=torch.float32)
    if rank is not None:
        keep = int(rank.item()) if isinstance(rank, torch.Tensor) else int(rank)
        left = left[:, :keep]
        right = right[:keep]
    if left.shape[0] != input_width or left.shape[1] != right.shape[0]:
        raise ValueError(
            f"invalid all-head factors {tuple(left.shape)} and {tuple(right.shape)} "
            f"for input width {input_width}"
        )
    bias_work = bias.to(device=source.device, dtype=torch.float32)
    output_width = int(right.shape[1])
    if output_width % dim != 0:
        raise ValueError(f"all-head output width {output_width} is not divisible by D={dim}")
    output_heads = output_width // dim
    flat = work.permute(0, 2, 1, 3).reshape(batch, tokens, input_width)
    pred = torch.matmul(torch.matmul(flat, left), right)
    if bias_work.ndim == 2:
        if tuple(bias_work.shape) != (output_heads, dim):
            raise ValueError(
                f"all-head bias must be {(output_heads, dim)}, got {tuple(bias_work.shape)}"
            )
        pred = pred + bias_work.reshape(1, 1, output_width)
    elif bias_work.ndim == 1 and bias_work.shape[0] == output_width:
        pred = pred + bias_work.reshape(1, 1, output_width)
    else:
        raise ValueError(f"invalid all-head bias shape {tuple(bias_work.shape)}")
    return pred.reshape(batch, tokens, output_heads, dim).permute(0, 2, 1, 3)


def _predict_hybrid_kv(
    current_k: torch.Tensor,
    previous_v: torch.Tensor,
    pred_params: dict,
) -> torch.Tensor:
    if current_k.shape != previous_v.shape:
        raise ValueError(
            f"Hybrid K/Vprev shapes differ: {tuple(current_k.shape)} and {tuple(previous_v.shape)}"
        )
    joint = torch.cat([current_k, previous_v], dim=1)
    return _predict_cross_kv_allhead_low_rank(
        joint,
        pred_params.get("cross_input_factor"),
        pred_params.get("cross_output_factor"),
        pred_params.get("cross_bias"),
        pred_params.get("cross_rank"),
    )


def _predict_hybrid_innovation(
    current_k: torch.Tensor,
    previous_k: torch.Tensor,
    previous_v: torch.Tensor,
    pred_params: dict,
) -> torch.Tensor:
    """Cross base plus a bounded correction from the previous residual."""
    if current_k.shape != previous_k.shape or current_k.shape != previous_v.shape:
        raise ValueError(
            "innovation Hybrid requires matching current K, previous K, and previous V"
        )
    current_cross = _predict_cross_kv(
        current_k,
        pred_params.get("cross_weight"),
        pred_params.get("cross_bias"),
    )
    previous_cross = _predict_cross_kv(
        previous_k,
        pred_params.get("cross_weight"),
        pred_params.get("cross_bias"),
    )
    return _combine_hybrid_innovation(
        current_cross,
        previous_cross,
        previous_v,
        pred_params.get("innovation_gamma"),
    )


def _combine_hybrid_innovation(
    current_cross: torch.Tensor,
    previous_cross: torch.Tensor,
    previous_v: torch.Tensor,
    gamma: torch.Tensor | None,
) -> torch.Tensor:
    """Combine already computed Cross predictions with previous V innovation."""
    if (
        current_cross.shape != previous_cross.shape
        or current_cross.shape != previous_v.shape
    ):
        raise ValueError(
            "innovation Hybrid requires matching current Cross, previous Cross, "
            "and previous V"
        )
    if gamma is None:
        raise ValueError("innovation Hybrid requires innovation_gamma")
    gamma = gamma.to(device=current_cross.device, dtype=torch.float32)
    if gamma.ndim == 1 and gamma.shape[0] == current_cross.shape[-1]:
        gamma = gamma.view(1, 1, 1, current_cross.shape[-1])
    elif gamma.ndim == 2 and tuple(gamma.shape) == (
        current_cross.shape[1],
        current_cross.shape[-1],
    ):
        gamma = gamma.view(
            1, current_cross.shape[1], 1, current_cross.shape[-1]
        )
    else:
        raise ValueError(
            "innovation_gamma must be [D] or [H,D], got "
            f"{tuple(gamma.shape)}"
        )
    return current_cross + gamma * (previous_v.float() - previous_cross)


def _resolve_cross_source_tensor(
    source_tensor: Optional[torch.Tensor],
    source_state: Optional[dict],
    expected_shape: tuple[int, int, int, int],
    device: torch.device,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if source_tensor is None:
        if source_state is None:
            raise ValueError("cross_kv predictor requires source_tensor or cross_source_state")
        source_tensor = s2pp_dequantize_tensor(source_state, output_dtype=output_dtype)
    if tuple(source_tensor.shape) != expected_shape:
        raise ValueError(f"cross_kv source shape must be {expected_shape}, got {tuple(source_tensor.shape)}")
    if source_tensor.device != device:
        source_tensor = source_tensor.to(device=device)
    return source_tensor


def _apply_pair_rotation(prev: torch.Tensor, cos: torch.Tensor | None, sin: torch.Tensor | None) -> torch.Tensor:
    """Apply learned pair rotation to *prev*.

    Supports two parameter shapes:
      - ``[D/2]`` — same rotation angles for all heads (shared).
      - ``[H, D/2]`` — per-head rotation angles.

    The rotation is applied independently to each (real, imag) pair
    formed from adjacent channels ``[2j, 2j+1]``:

        y_{2j}   = x_{2j} * cos_j - x_{2j+1} * sin_j
        y_{2j+1} = x_{2j} * sin_j + x_{2j+1} * cos_j
    """
    if cos is None or sin is None:
        raise ValueError("both rotation cos and sin are required for pair rotation")

    if prev.ndim != 4:
        raise ValueError(
            f"pair rotation requires [B, H, S, D] input, got {tuple(prev.shape)}"
        )
    B, H, S, D = prev.shape

    if D % 2 != 0:
        raise ValueError(f"pair rotation requires even head_dim, got {D}")

    pair_dim = D // 2

    if cos.shape != sin.shape:
        raise ValueError(
            f"rotation cos/sin shapes differ: "
            f"{tuple(cos.shape)} vs {tuple(sin.shape)}"
        )

    if cos.ndim == 1:
        if cos.shape[0] != pair_dim:
            raise ValueError(
                f"rotation cos/sin [D/2] must have length {pair_dim}, "
                f"got {cos.shape[0]}"
            )
        c = cos.float().view(1, 1, 1, pair_dim)
        s = sin.float().view(1, 1, 1, pair_dim)

    elif cos.ndim == 2:
        expected = (H, pair_dim)
        if tuple(cos.shape) != expected:
            raise ValueError(
                f"per-head rotation cos/sin must be [{H}, {pair_dim}] "
                f"for input with {H} heads, got {tuple(cos.shape)}"
            )
        c = cos.float().view(1, H, 1, pair_dim)
        s = sin.float().view(1, H, 1, pair_dim)

    else:
        raise ValueError(
            f"rotation cos/sin must be [D/2] or [H,D/2], "
            f"got ndim={cos.ndim}, shape={tuple(cos.shape)}"
        )

    # --- unit-magnitude check ---
    norm_error = (c.square() + s.square() - 1.0).abs().max().item()
    if norm_error > 1e-4:
        raise ValueError(
            f"rotation parameters are not unit magnitude: "
            f"max |cos²+sin²-1| = {norm_error:.2e}"
        )

    work = prev.float()
    real = work[..., 0::2]
    imag = work[..., 1::2]
    rotated = torch.empty_like(work)
    rotated[..., 0::2] = real * c - imag * s
    rotated[..., 1::2] = real * s + imag * c
    return rotated


# ---------------------------------------------------------------------------
#  S2-Opt v2: extended predictor implementations
# ---------------------------------------------------------------------------

_PREDICTOR_KINDS = {
    "affine",          # original: alpha * x + beta (per dim)
    "block_var",       # block-diagonal VAR: A_g @ x[group] + b_g
    "error_feedback",  # AR(1) + gamma * prev_error (Delta-Sigma)
    "ar2",             # AR(2): alpha1 * x[t-1] + alpha2 * x[t-2] + beta
    "temporal_multiframe",  # AR(p) over decoder-visible reconstructed history
    "cross_kv",        # V prediction from reconstructed K: K_hat @ M + b
    "cross_kv_allhead_low_rank",
    "hybrid_kv_innovation",
    "hybrid_kv_allhead_low_rank",
}


def _detect_predictor_kind(params: dict) -> str:
    """Detect predictor kind from loaded npz parameter dict."""
    explicit = params.get("predictor_kind")
    if explicit is not None:
        return str(explicit)
    if params.get("cross_input_factor") is not None:
        return "cross_kv_allhead_low_rank"
    if params.get("cross_weight") is not None:
        return "cross_kv"
    if params.get("gamma") is not None:
        return "error_feedback"
    if params.get("alpha1") is not None or params.get("alpha2") is not None:
        return "ar2"
    if params.get("history_weights") is not None:
        return "temporal_multiframe"
    if params.get("A") is not None:
        return "block_var"
    return "affine"


def _predict_block_var(
    prev: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Block-diagonal VAR predictor.

    prev:   (..., D)   reconstructed previous unit
    A:      (num_groups, g, g)   group-wise prediction matrices
    b:      (num_groups, g)      group-wise biases
    returns: (..., D)   prediction, same shape as prev
    """
    D = prev.shape[-1]
    num_groups = D // group_size
    work = prev.float()
    # reshape to (..., num_groups, group_size)
    grouped = work.reshape(*work.shape[:-1], num_groups, group_size)
    # batched matmul: (..., num_groups, 1, g) @ (num_groups, g, g) -> (..., num_groups, 1, g)
    A_view = A.float().unsqueeze(0)  # (1, num_groups, g, g)
    pred = torch.matmul(grouped.unsqueeze(-2), A_view).squeeze(-2)  # (..., num_groups, g)
    pred = pred + b.float()  # broadcast over trailing dims
    return pred.reshape_as(work)


def _predict_error_feedback(
    prev: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    prev_error: torch.Tensor,
) -> torch.Tensor:
    """
    AR(1) with Delta-Sigma error feedback.

    x_pred[t] = alpha * x_hat[t-1] + beta + gamma * e[t-1]
    where e[t-1] = x[t-1] - x_hat[t-1] is the quantization error.
    """
    view_shape = [1] * prev.ndim
    view_shape[-1] = prev.shape[-1]
    base = prev.float() * alpha.view(*view_shape) + beta.view(*view_shape)
    feedback = prev_error.float() * gamma.view(*view_shape)
    return base + feedback


def _predict_ar2(
    prev1: torch.Tensor,
    prev2: torch.Tensor,
    alpha1: torch.Tensor,
    alpha2: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    AR(2) multi-step predictor.

    x_pred[t] = alpha1 * x_hat[t-1] + alpha2 * x_hat[t-2] + beta
    """
    view_shape = [1] * prev1.ndim
    view_shape[-1] = prev1.shape[-1]
    return (
        prev1.float() * alpha1.view(*view_shape)
        + prev2.float() * alpha2.view(*view_shape)
        + beta.view(*view_shape)
    )


def _predict_temporal_multiframe(
    history: Sequence[torch.Tensor],
    history_weights: torch.Tensor | None,
    history_bias: torch.Tensor | None,
) -> torch.Tensor:
    """Channel-wise AR(p) over history ordered newest to oldest."""
    if history_weights is None or history_bias is None:
        raise ValueError(
            "temporal_multiframe predictor requires history_weights and history_bias"
        )
    if history_weights.ndim not in (3, 4):
        raise ValueError(
            "history_weights must be [P,P,D] or [P,P,H,D], got "
            f"{tuple(history_weights.shape)}"
        )
    history_length = int(history_weights.shape[0])
    if history_weights.shape[1] != history_length:
        raise ValueError(
            "history_weights first two dimensions must both equal P, got "
            f"{tuple(history_weights.shape)}"
        )
    expected_bias_shape = (
        (history_length, history_weights.shape[-1])
        if history_weights.ndim == 3
        else (
            history_length,
            history_weights.shape[-2],
            history_weights.shape[-1],
        )
    )
    if tuple(history_bias.shape) != expected_bias_shape:
        raise ValueError(
            f"history_bias must have shape {expected_bias_shape}, got "
            f"{tuple(history_bias.shape)}"
        )
    if not history:
        raise ValueError("temporal_multiframe predictor requires history")
    order = min(len(history), history_length)
    reference = history[0]
    prediction = torch.zeros_like(reference, dtype=torch.float32)
    for lag in range(order):
        source = history[lag]
        if source.shape != reference.shape:
            raise ValueError(
                "temporal_multiframe history shapes differ: "
                f"{tuple(source.shape)} vs {tuple(reference.shape)}"
            )
        coefficient = history_weights[order - 1, lag]
        if history_weights.ndim == 3:
            if coefficient.shape[0] != source.shape[-1]:
                raise ValueError(
                    f"shared history coefficient D={coefficient.shape[0]} "
                    f"does not match source D={source.shape[-1]}"
                )
            coefficient = coefficient.view(
                *([1] * (source.ndim - 1)), source.shape[-1]
            )
        else:
            if source.ndim != 4 or tuple(coefficient.shape) != (
                source.shape[1],
                source.shape[-1],
            ):
                raise ValueError(
                    "per-head history coefficient must match [H,D], got "
                    f"{tuple(coefficient.shape)} for {tuple(source.shape)}"
                )
            coefficient = coefficient.view(
                1, source.shape[1], 1, source.shape[-1]
            )
        prediction += source.float() * coefficient
    bias = history_bias[order - 1]
    if history_bias.ndim == 2:
        bias = bias.view(*([1] * (reference.ndim - 1)), reference.shape[-1])
    else:
        bias = bias.view(1, reference.shape[1], 1, reference.shape[-1])
    return prediction + bias


# ---------------------------------------------------------------------------
#  S2-Opt v2: per-channel residual normalisation
# ---------------------------------------------------------------------------


def _normalize_residual(
    residual: torch.Tensor,
    mean: torch.Tensor,
    inv_std: torch.Tensor,
) -> torch.Tensor:
    """Scale residual to ~N(0,1) per channel before quantisation."""
    view_shape = [1] * residual.ndim
    view_shape[-1] = residual.shape[-1]
    return (residual.float() - mean.view(*view_shape)) * inv_std.view(*view_shape)


def _denormalize_residual(
    normed: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Reverse per-channel normalisation after dequantisation."""
    view_shape = [1] * normed.ndim
    view_shape[-1] = normed.shape[-1]
    return normed.float() * std.view(*view_shape) + mean.view(*view_shape)


# ---------------------------------------------------------------------------
#  S2-Opt v2: Lloyd-Max (Gaussian-optimal) quantisation tables
# ---------------------------------------------------------------------------


def _build_lloyd_max_table(bits: int, num_iters: int = 50) -> torch.Tensor:
    """
    Build Lloyd-Max decision + reconstruction levels for N(0,1).

    Returns tensor of shape (2**bits,)  — reconstruction levels.
    Decision boundaries are midpoints between adjacent levels.
    """
    import math
    n_levels = 1 << bits
    # initialise with uniform spacing in [-3, 3]
    levels = torch.linspace(-3.0 + 6.0 / n_levels, 3.0 - 6.0 / n_levels, n_levels)
    for _ in range(num_iters):
        # decision boundaries: midpoints
        boundaries = torch.empty(n_levels + 1)
        boundaries[0] = -float("inf")
        boundaries[-1] = float("inf")
        boundaries[1:-1] = (levels[:-1] + levels[1:]) / 2.0
        # update reconstruction levels = conditional mean in each interval
        new_levels = torch.empty(n_levels)
        for i in range(n_levels):
            lo, hi = boundaries[i].item(), boundaries[i + 1].item()
            # E[x | lo < x < hi] for N(0,1)
            if lo == -float("inf"):
                # leftmost bin
                new_levels[i] = -math.sqrt(2.0 / math.pi) * math.exp(-(hi ** 2) / 2.0) / (
                    0.5 * (1.0 + math.erf(hi / math.sqrt(2.0)))
                )
            elif hi == float("inf"):
                # rightmost bin
                new_levels[i] = math.sqrt(2.0 / math.pi) * math.exp(-(lo ** 2) / 2.0) / (
                    0.5 * (1.0 - math.erf(lo / math.sqrt(2.0)))
                )
            else:
                phi_lo = math.exp(-(lo ** 2) / 2.0) / math.sqrt(2.0 * math.pi)
                phi_hi = math.exp(-(hi ** 2) / 2.0) / math.sqrt(2.0 * math.pi)
                Phi_lo = 0.5 * (1.0 + math.erf(lo / math.sqrt(2.0)))
                Phi_hi = 0.5 * (1.0 + math.erf(hi / math.sqrt(2.0)))
                new_levels[i] = (phi_lo - phi_hi) / max(Phi_hi - Phi_lo, 1e-15)
        if torch.allclose(new_levels, levels, atol=1e-6):
            break
        levels = new_levels
    return levels.float()


@lru_cache(maxsize=8)
def _get_lloyd_max_levels(bits: int) -> torch.Tensor:
    """Cached Lloyd-Max reconstruction levels for a given bit-width."""
    return _build_lloyd_max_table(bits)


def _quantize_to_levels(
    x: torch.Tensor,
    levels: torch.Tensor,
) -> torch.Tensor:
    """
    Quantize x to nearest Lloyd-Max reconstruction level.
    Returns integer indices into levels tensor.
    """
    n_levels = levels.numel()
    # x: (..., D), levels: (n_levels,)
    x_flat = x.float().reshape(-1, 1)  # (N, 1)
    dist = (x_flat - levels.to(x.device).float().view(1, -1)).abs()  # (N, n_levels)
    indices = dist.argmin(dim=-1)  # (N,)
    return indices.reshape(x.shape).to(torch.uint8)


def _dequantize_from_levels(
    indices: torch.Tensor,
    levels: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Look up reconstruction values from Lloyd-Max level indices."""
    return levels.to(indices.device, dtype=output_dtype)[indices.long()]


# ---------------------------------------------------------------------------
#  S2-Opt v2: extended parameter loading
# ---------------------------------------------------------------------------


def _empty_predictor_params_cpu() -> dict[str, np.ndarray | None]:
    return {
        "predictor_kind": None,
        "alpha": None, "beta": None, "cos": None, "sin": None,
        "gamma": None, "alpha1": None, "alpha2": None,
        "history_weights": None, "history_bias": None, "history_length": None,
        "A": None, "b": None, "group_size": None,
        "cross_weight": None, "cross_bias": None,
        "cross_input_factor": None, "cross_output_factor": None,
        "cross_rank": None, "temporal_weight": None, "temporal_bias": None,
        "innovation_gamma": None,
        "predictor_stride": None,
        "residual_mean": None, "residual_inv_std": None,
        "lloyd_levels": None, "lloyd_levels_2bit": None,
        "lloyd_levels_3bit": None, "lloyd_levels_4bit": None,
        # Rotation metadata — required for rope_affine validation
        "parameter_role": None,
        "coordinate_system": None,
        "predictor_mode": None,
        "fit_quant_bits": None,
        "fit_group_size": None,
        "fit_anchor_bits": None,
        "fit_mode": None,
        "num_heads": None,
        "head_dim": None,
    }


@lru_cache(maxsize=4)
def _load_torch_predictor_registry(path: str) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"S2 affine .pt must contain a dictionary: {path}")
    if str(payload.get("predictor_mode", "")) not in {"affine", "affine_channel"}:
        raise ValueError(
            f"S2 affine .pt predictor_mode must be affine_channel: {path}"
        )
    for key in ("K", "V"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"S2 affine .pt lacks {key} layer parameters: {path}")
    return payload


def _load_torch_affine_params_cpu(
    path: str,
    parameter_role: Optional[str],
    layer_idx: Optional[int],
) -> dict[str, np.ndarray | None]:
    role = str(parameter_role or "").upper()
    if role not in {"K", "V"}:
        raise ValueError(f"S2 affine .pt requires parameter_role K or V: {path}")
    if layer_idx is None:
        raise ValueError(f"S2 affine .pt is layer-specific; layer_idx is required: {path}")
    payload = _load_torch_predictor_registry(path)
    layer_table = payload[role]
    layer = layer_table.get(int(layer_idx), layer_table.get(str(int(layer_idx))))
    if not isinstance(layer, dict):
        raise ValueError(f"S2 affine .pt has no {role} layer {layer_idx}: {path}")
    alpha = layer.get("alpha")
    beta = layer.get("beta")
    if alpha is None or beta is None:
        raise ValueError(f"S2 affine .pt {role} layer {layer_idx} lacks alpha/beta: {path}")
    alpha_np = torch.as_tensor(alpha, dtype=torch.float32).detach().cpu().numpy()
    beta_np = torch.as_tensor(beta, dtype=torch.float32).detach().cpu().numpy()
    if alpha_np.shape != beta_np.shape:
        raise ValueError(
            f"S2 affine .pt {role} layer {layer_idx} alpha/beta shapes differ: "
            f"{alpha_np.shape} vs {beta_np.shape}"
        )
    if alpha_np.ndim not in (1, 2):
        raise ValueError(
            f"S2 affine .pt {role} layer {layer_idx} must be [D] or [H,D], got {alpha_np.shape}"
        )
    if not np.isfinite(alpha_np).all() or not np.isfinite(beta_np).all():
        raise ValueError(f"S2 affine .pt {role} layer {layer_idx} contains NaN/Inf: {path}")
    result = _empty_predictor_params_cpu()
    result.update(
        predictor_kind="affine",
        alpha=np.ascontiguousarray(alpha_np, dtype=np.float32),
        beta=np.ascontiguousarray(beta_np, dtype=np.float32),
    )
    return result


@lru_cache(maxsize=128)
def _load_predictor_params_cpu(
    path: str,
    parameter_role: Optional[str] = None,
    layer_idx: Optional[int] = None,
) -> dict[str, np.ndarray | None]:
    if not path:
        return _empty_predictor_params_cpu()
    if str(path).lower().endswith((".pt", ".pth")):
        return _load_torch_affine_params_cpu(path, parameter_role, layer_idx)
    data = np.load(path, allow_pickle=True)
    cross_weight = _npz_first(data, ("cross_weight", "k_to_v_weight", "k_to_v_matrix", "M", "W", "matrix", "weight"))
    cross_bias = _npz_first(data, ("cross_bias", "k_to_v_bias", "bias"))
    if cross_bias is None and cross_weight is not None and "b" in data and "A" not in data:
        cross_bias = data["b"]
    return {
        "predictor_kind": (
            str(np.asarray(data["predictor_kind"]).item())
            if "predictor_kind" in data
            else None
        ),
        "alpha": data["alpha"].astype(np.float32) if "alpha" in data else None,
        "beta": data["beta"].astype(np.float32) if "beta" in data else None,
        "cos": data["cos"].astype(np.float32) if "cos" in data else None,
        "sin": data["sin"].astype(np.float32) if "sin" in data else None,
        # S2-Opt v2: error feedback
        "gamma": data["gamma"].astype(np.float32) if "gamma" in data else None,
        # S2-Opt v2: AR(2)
        "alpha1": data["alpha1"].astype(np.float32) if "alpha1" in data else None,
        "alpha2": data["alpha2"].astype(np.float32) if "alpha2" in data else None,
        "history_weights": data["history_weights"].astype(np.float32) if "history_weights" in data else None,
        "history_bias": data["history_bias"].astype(np.float32) if "history_bias" in data else None,
        "history_length": int(np.asarray(data["history_length"]).item()) if "history_length" in data else None,
        # S2-Opt v2: block-VAR
        "A": data["A"].astype(np.float32) if "A" in data else None,
        "b": data["b"].astype(np.float32) if "b" in data else None,
        "group_size": int(data["group_size"]) if "group_size" in data else None,
        "cross_weight": cross_weight.astype(np.float32) if cross_weight is not None else None,
        "cross_bias": cross_bias.astype(np.float32) if cross_bias is not None else None,
        "cross_input_factor": data["cross_input_factor"].astype(np.float32) if "cross_input_factor" in data else None,
        "cross_output_factor": data["cross_output_factor"].astype(np.float32) if "cross_output_factor" in data else None,
        "cross_rank": data["cross_rank"].astype(np.int32) if "cross_rank" in data else None,
        "temporal_weight": data["temporal_weight"].astype(np.float32) if "temporal_weight" in data else None,
        "temporal_bias": data["temporal_bias"].astype(np.float32) if "temporal_bias" in data else None,
        "innovation_gamma": data["innovation_gamma"].astype(np.float32) if "innovation_gamma" in data else None,
        "predictor_stride": int(np.asarray(data["predictor_stride"]).item()) if "predictor_stride" in data else None,
        # S2-Opt v2: per-channel normalisation
        "residual_mean": data["residual_mean"].astype(np.float32) if "residual_mean" in data else None,
        "residual_inv_std": data["residual_inv_std"].astype(np.float32) if "residual_inv_std" in data else None,
        # S2-Opt v2: Lloyd-Max levels (multi-bit support)
        "lloyd_levels": data["lloyd_levels"].astype(np.float32) if "lloyd_levels" in data else None,
        "lloyd_levels_2bit": data["lloyd_levels_2bit"].astype(np.float32) if "lloyd_levels_2bit" in data else None,
        "lloyd_levels_3bit": data["lloyd_levels_3bit"].astype(np.float32) if "lloyd_levels_3bit" in data else None,
        "lloyd_levels_4bit": data["lloyd_levels_4bit"].astype(np.float32) if "lloyd_levels_4bit" in data else None,
        # Rotation metadata
        "parameter_role": _npz_str(data, "parameter_role"),
        "coordinate_system": _npz_str(data, "coordinate_system"),
        "predictor_mode": _npz_str(data, "predictor_mode"),
        "fit_quant_bits": _npz_int(data, "fit_quant_bits"),
        "fit_group_size": _npz_int(data, "fit_group_size"),
        "fit_anchor_bits": _npz_int(data, "fit_anchor_bits"),
        "fit_mode": _npz_str(data, "fit_mode"),
        "num_heads": _npz_int(data, "num_heads"),
        "head_dim": _npz_int(data, "head_dim"),
    }


def _npz_first(data, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _npz_str(data: dict, key: str) -> str | None:
    """Read a scalar string from an npz array, or return None if missing."""
    val = data.get(key)
    if val is None:
        return None
    arr = np.asarray(val)
    if arr.ndim == 0:
        return str(arr.item())
    return str(arr.flat[0])


def _npz_int(data: dict, key: str) -> int | None:
    """Read a scalar integer from an npz array, or return None if missing."""
    val = data.get(key)
    if val is None:
        return None
    arr = np.asarray(val)
    if arr.ndim == 0:
        return int(arr.item())
    return int(arr.flat[0])


def _load_affine(path: str, device: torch.device, head_dim: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    params = _load_predictor_params(path, device, head_dim)
    return params["alpha"], params["beta"]


def _select_cross_kv_arrays(
    params: dict[str, np.ndarray | None],
    head_dim: int,
    layer_idx: Optional[int],
) -> tuple[np.ndarray, np.ndarray | None]:
    cross_weight_np = params["cross_weight"]
    cross_bias_np = params["cross_bias"]
    if cross_weight_np is None:
        raise ValueError("cross_kv predictor requires cross_weight")
    if cross_weight_np.ndim not in (2, 3, 4):
        raise ValueError(f"cross_kv weight must be [D,D], [H,D,D], or [L,H,D,D], got {cross_weight_np.shape}")
    trailing = tuple(cross_weight_np.shape[-2:])
    layer_dense = (
        cross_weight_np.ndim == 3
        and trailing[0] == trailing[1]
        and trailing[0] > head_dim
        and trailing[0] % head_dim == 0
    )
    global_dense = (
        cross_weight_np.ndim == 2
        and trailing[0] == trailing[1]
        and trailing[0] % head_dim == 0
    )
    if not global_dense and not layer_dense and trailing != (head_dim, head_dim):
        raise ValueError(
            f"cross_kv weight has invalid trailing dimensions: {cross_weight_np.shape}"
        )
    if cross_weight_np.ndim == 4 or layer_dense:
        if layer_idx is None:
            raise ValueError(f"cross_kv weight is layer-specific {cross_weight_np.shape}; layer_idx is required")
        layer_idx_int = int(layer_idx)
        if layer_idx_int < 0 or layer_idx_int >= cross_weight_np.shape[0]:
            raise ValueError(f"cross_kv layer_idx={layer_idx_int} outside [0, {cross_weight_np.shape[0] - 1}]")
        cross_weight_np = cross_weight_np[layer_idx_int]
        if cross_bias_np is not None and cross_bias_np.ndim in (2, 3):
            cross_bias_np = cross_bias_np[layer_idx_int]
    if cross_bias_np is not None:
        if cross_bias_np.ndim not in (1, 2):
            raise ValueError(f"cross_kv bias must be [D], [H,D], or [L,H,D] with layer-specific weight, got {cross_bias_np.shape}")
        if cross_bias_np.ndim == 1 and cross_bias_np.shape[0] % head_dim != 0:
            raise ValueError(f"flat cross_kv bias width must be divisible by {head_dim}, got {cross_bias_np.shape}")
        if cross_bias_np.ndim == 2 and cross_bias_np.shape[-1] != head_dim:
            raise ValueError(f"cross_kv bias last dim must be {head_dim}, got {cross_bias_np.shape[-1]}")
    return cross_weight_np, cross_bias_np


def _load_cross_kv_tensors(
    path: str,
    device: torch.device,
    head_dim: int,
    layer_idx: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    key = (path, str(device), int(head_dim), None if layer_idx is None else int(layer_idx))
    cached = _S2PP_CROSS_KV_DEVICE_CACHE.get(key)
    if cached is not None:
        return cached
    params = _load_predictor_params_cpu(path)
    cross_weight_np, cross_bias_np = _select_cross_kv_arrays(params, head_dim, layer_idx)
    weight = torch.from_numpy(cross_weight_np).to(device=device, dtype=torch.float32)
    bias = (
        torch.from_numpy(cross_bias_np).to(device=device, dtype=torch.float32)
        if cross_bias_np is not None
        else None
    )
    _S2PP_CROSS_KV_DEVICE_CACHE[key] = (weight, bias)
    return weight, bias


def _load_device_parameter(
    path: str,
    name: str,
    array: np.ndarray,
    device: torch.device,
    layer_idx: Optional[int],
) -> torch.Tensor:
    """Cache immutable predictor parameters on device across cache reads."""
    key = (
        path,
        name,
        str(device),
        None if layer_idx is None else int(layer_idx),
    )
    cached = _S2PP_DEVICE_PARAMETER_CACHE.get(key)
    if cached is not None:
        return cached
    tensor = torch.from_numpy(np.ascontiguousarray(array)).to(
        device=device, dtype=torch.float32
    )
    _S2PP_DEVICE_PARAMETER_CACHE[key] = tensor
    return tensor


def _select_layer_parameter(
    array: np.ndarray | None,
    *,
    layer_idx: Optional[int],
    unlayered_ndim: int,
    name: str,
) -> np.ndarray | None:
    if array is None:
        return None
    if array.ndim == unlayered_ndim:
        return array
    if array.ndim != unlayered_ndim + 1:
        raise ValueError(f"{name} has unsupported shape {array.shape}")
    if layer_idx is None:
        raise ValueError(f"layer_idx is required for layer-specific {name} {array.shape}")
    index = int(layer_idx)
    if index < 0 or index >= array.shape[0]:
        raise ValueError(f"layer_idx={index} outside {name} layers {array.shape[0]}")
    return array[index]


def _load_predictor_params(
    path: str,
    device: torch.device,
    head_dim: int,
    layer_idx: Optional[int] = None,
    *,
    parameter_role: Optional[str] = None,
    num_heads: Optional[int] = None,
) -> dict:
    """
    Load predictor parameters from an existing .npz or an affine registry .pt.

    Returns a dict with keys for all supported predictor/quantiser features.
    Backward-compatible: old files (alpha+beta only) work as "affine" mode.
    """
    params = _load_predictor_params_cpu(path, parameter_role, layer_idx)
    result: dict = {}
    result["predictor_kind"] = _detect_predictor_kind(params)
    result["predictor_stride"] = params.get("predictor_stride")

    # --- common / legacy affine ---
    alpha_np = params["alpha"]
    beta_np = params["beta"]
    cos_np = params["cos"]
    sin_np = params["sin"]
    if (alpha_np is None) != (beta_np is None):
        raise ValueError(f"affine predictor file must contain both alpha and beta: {path}")
    if alpha_np is not None and beta_np is not None and alpha_np.shape != beta_np.shape:
        raise ValueError(
            f"affine alpha/beta shapes must match, got {alpha_np.shape} and {beta_np.shape}: {path}"
        )
    if alpha_np is not None and beta_np is not None and (
        alpha_np.shape[-1] != head_dim or beta_np.shape[-1] != head_dim
    ):
        raise ValueError(
            f"affine alpha/beta last dim must be {head_dim}, got {alpha_np.shape[-1]} and {beta_np.shape[-1]}"
        )
    if alpha_np is not None and alpha_np.ndim not in (1, 2):
        raise ValueError(f"affine alpha/beta must be [D] or [H,D], got {alpha_np.shape}: {path}")
    if alpha_np is not None and alpha_np.ndim == 2:
        if num_heads is None:
            raise ValueError(f"num_heads is required for per-head affine parameters {alpha_np.shape}: {path}")
        if alpha_np.shape[0] != int(num_heads):
            raise ValueError(
                f"affine alpha/beta head count must be {int(num_heads)}, got {alpha_np.shape[0]}: {path}"
            )
    if (cos_np is None) != (sin_np is None):
        raise ValueError(f"rope predictor file must contain both cos and sin: {path}")
    if cos_np is not None and sin_np is not None:
        if head_dim % 2 != 0:
            raise ValueError(f"rope predictor requires even head_dim, got {head_dim}")
        if cos_np.shape[-1] != head_dim // 2 or sin_np.shape[-1] != head_dim // 2:
            raise ValueError(
                f"rope cos/sin last dim must be {head_dim // 2}, got {cos_np.shape[-1]} and {sin_np.shape[-1]}"
            )
    result["alpha"] = torch.from_numpy(alpha_np).to(device=device, dtype=torch.float32) if alpha_np is not None else None
    result["beta"] = torch.from_numpy(beta_np).to(device=device, dtype=torch.float32) if beta_np is not None else None
    result["cos"] = torch.from_numpy(cos_np).to(device=device, dtype=torch.float32) if cos_np is not None else None
    result["sin"] = torch.from_numpy(sin_np).to(device=device, dtype=torch.float32) if sin_np is not None else None

    # --- error feedback (gamma) ---
    gamma_np = params["gamma"]
    if gamma_np is not None and gamma_np.shape[-1] != head_dim:
        raise ValueError(f"gamma last dim must be {head_dim}, got {gamma_np.shape[-1]}")
    result["gamma"] = torch.from_numpy(gamma_np).to(device=device, dtype=torch.float32) if gamma_np is not None else None

    # --- AR(2) ---
    for key in ("alpha1", "alpha2"):
        arr = params[key]
        if arr is not None and arr.shape[-1] != head_dim:
            raise ValueError(f"{key} last dim must be {head_dim}, got {arr.shape[-1]}")
        result[key] = torch.from_numpy(arr).to(device=device, dtype=torch.float32) if arr is not None else None

    # --- decoder-visible multi-frame temporal predictor ---
    history_weights_np = params["history_weights"]
    history_bias_np = params["history_bias"]
    history_length = params["history_length"]
    has_multiframe_params = (
        result["predictor_kind"] == "temporal_multiframe"
        or history_weights_np is not None
        or history_bias_np is not None
    )
    if has_multiframe_params:
        if (
            history_weights_np is None
            or history_bias_np is None
            or history_length is None
        ):
            raise ValueError(
                "temporal_multiframe predictor requires history_weights, "
                f"history_bias, and history_length: {path}"
            )
        history_length = int(history_length)
        if history_length < 2:
            raise ValueError(
                f"temporal_multiframe history_length must be at least 2, got "
                f"{history_length}: {path}"
            )
        if history_weights_np.ndim not in (3, 4) or history_weights_np.shape[:2] != (
            history_length,
            history_length,
        ):
            raise ValueError(
                "history_weights must be [P,P,D] or [P,P,H,D], got "
                f"{history_weights_np.shape}: {path}"
            )
        if history_weights_np.shape[-1] != head_dim:
            raise ValueError(
                f"history_weights last dim must be {head_dim}, got "
                f"{history_weights_np.shape[-1]}: {path}"
            )
        if history_weights_np.ndim == 4:
            if num_heads is None or history_weights_np.shape[-2] != int(num_heads):
                raise ValueError(
                    f"per-head history_weights must match H={num_heads}, got "
                    f"{history_weights_np.shape}: {path}"
                )
            expected_bias_shape = (
                history_length,
                int(num_heads),
                head_dim,
            )
        else:
            expected_bias_shape = (history_length, head_dim)
        if history_bias_np.shape != expected_bias_shape:
            raise ValueError(
                f"history_bias must be {expected_bias_shape}, got "
                f"{history_bias_np.shape}: {path}"
            )
        if not np.all(np.isfinite(history_weights_np)) or not np.all(
            np.isfinite(history_bias_np)
        ):
            raise ValueError(
                f"temporal_multiframe parameters contain NaN/Inf: {path}"
            )
        result["history_weights"] = _load_device_parameter(
            path, "history_weights", history_weights_np, device, layer_idx
        )
        result["history_bias"] = _load_device_parameter(
            path, "history_bias", history_bias_np, device, layer_idx
        )
        result["history_length"] = history_length
    else:
        result["history_weights"] = None
        result["history_bias"] = None
        result["history_length"] = None

    # --- block-VAR ---
    A_np = params["A"]
    b_np = params["b"]
    gs = params["group_size"]
    if A_np is not None:
        if b_np is None or gs is None:
            raise ValueError("block_var predictor requires A, b, and group_size in npz")
        result["A"] = torch.from_numpy(A_np).to(device=device, dtype=torch.float32)
        result["b"] = torch.from_numpy(b_np).to(device=device, dtype=torch.float32)
        result["group_size"] = int(gs)
    else:
        result["A"] = None
        result["b"] = None
        result["group_size"] = None

    # --- K -> V cross-modal predictor ---
    cross_weight_np = params["cross_weight"]
    cross_bias_np = params["cross_bias"]
    if cross_weight_np is not None:
        result["cross_weight"], result["cross_bias"] = _load_cross_kv_tensors(
            path, device, head_dim, layer_idx
        )
    else:
        result["cross_weight"] = None
        result["cross_bias"] = None
    for key, ndim in (
        ("cross_input_factor", 2),
        ("cross_output_factor", 2),
        ("temporal_weight", 3),
        ("temporal_bias", 2),
        ("innovation_gamma", 2),
    ):
        selected = _select_layer_parameter(
            params.get(key), layer_idx=layer_idx, unlayered_ndim=ndim, name=key
        )
        if selected is not None and key == "innovation_gamma":
            if selected.shape[-1] != head_dim:
                raise ValueError(
                    f"innovation_gamma last dim must be {head_dim}, "
                    f"got {selected.shape[-1]}"
                )
            if not np.all(np.isfinite(selected)):
                raise ValueError("innovation_gamma contains non-finite values")
            if float(np.max(np.abs(selected))) >= 1.0:
                raise ValueError(
                    "innovation_gamma must satisfy max(abs(gamma)) < 1"
                )
        result[key] = (
            _load_device_parameter(path, key, selected, device, layer_idx)
            if selected is not None
            else None
        )
    rank_np = params.get("cross_rank")
    if rank_np is None:
        result["cross_rank"] = None
    else:
        rank_array = np.asarray(rank_np)
        if rank_array.ndim == 0:
            selected_rank = int(rank_array)
        else:
            if layer_idx is None:
                raise ValueError("layer_idx is required for layer-specific cross_rank")
            selected_rank = int(rank_array[int(layer_idx)])
        result["cross_rank"] = torch.tensor(selected_rank, device=device, dtype=torch.int32)

    # Factorized files use cross_bias even when no dense cross_weight exists.
    if result["cross_bias"] is None and params.get("cross_bias") is not None:
        selected_bias = _select_layer_parameter(
            params["cross_bias"], layer_idx=layer_idx, unlayered_ndim=2, name="cross_bias"
        )
        result["cross_bias"] = torch.from_numpy(selected_bias).to(
            device=device, dtype=torch.float32
        )

    # --- per-channel normalisation ---
    for key in ("residual_mean", "residual_inv_std"):
        arr = params[key]
        if arr is not None and arr.shape[-1] != head_dim:
            raise ValueError(f"{key} last dim must be {head_dim}, got {arr.shape[-1]}")
        result[key] = torch.from_numpy(arr).to(device=device, dtype=torch.float32) if arr is not None else None
    # derive residual_std for denormalisation
    result["residual_std"] = (
        1.0 / result["residual_inv_std"].float()
        if result["residual_inv_std"] is not None
        else None
    )

    # --- Lloyd-Max levels ---
    for key in ("lloyd_levels", "lloyd_levels_2bit", "lloyd_levels_3bit", "lloyd_levels_4bit"):
        arr = params[key]
        result[key] = torch.from_numpy(arr).to(device=device, dtype=torch.float32) if arr is not None else None

    # --- Rotation metadata (plain values, not tensors) ---
    for meta_key in (
        "parameter_role", "coordinate_system", "predictor_mode",
        "fit_quant_bits", "fit_group_size", "fit_anchor_bits",
        "fit_mode",
    ):
        val = params.get(meta_key)
        if val is not None:
            result[meta_key] = val
    # num_heads / head_dim from metadata overrides (stored as int, not tensor)
    for meta_key in ("num_heads", "head_dim"):
        val = params.get(meta_key)
        if val is not None:
            result[meta_key] = int(val)

    return result


def _normalize_predictor_mode(mode: str) -> str:
    aliases = {
        "rope": "rope_affine",
        "rope_rotation": "rope_affine",
        "rope-rotation": "rope_affine",
        "rope_rotation_affine": "rope_affine",
        "rope-affine": "rope_affine",
        # S2-Opt v2
        "blockvar": "block_var",
        "block-var": "block_var",
        "bv": "block_var",
        "errorfeedback": "error_feedback",
        "error-feedback": "error_feedback",
        "ef": "error_feedback",
        "delta_sigma": "error_feedback",
        "ar2": "ar2",
        "k_to_v": "cross_kv",
        "k2v": "cross_kv",
        "k-to-v": "cross_kv",
        "k-v": "cross_kv",
        "cross": "cross_kv",
        "cross-modal": "cross_kv",
        "cross_modal": "cross_kv",
        "cross-kv": "cross_kv",
        "kv_cross": "cross_kv",
    }
    return aliases.get(str(mode), str(mode))


def _resolve_scale_precision(scale_precision: torch.dtype | str) -> torch.dtype:
    if isinstance(scale_precision, torch.dtype):
        return scale_precision
    if scale_precision in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if scale_precision in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"unsupported scale precision: {scale_precision}")


def _validate_bits(bits: int) -> int:
    bits = int(bits)
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in [2, 8], got {bits}")
    return bits


def _validate_residual_quant_mode(mode: str) -> str:
    aliases = {
        "asym": "asym_zero_point",
        "asymmetric": "asym_zero_point",
        "asymmetric_zero_point": "asym_zero_point",
        "zero_point": "asym_zero_point",
        "zp": "asym_zero_point",
        "sym": "symmetric",
    }
    mode = aliases.get(str(mode), str(mode))
    if mode not in {"symmetric", "asym_zero_point"}:
        raise ValueError(f"unsupported residual quant mode: {mode}")
    return mode


def _validate_block_size(block_size: int, head_dim: int) -> int:
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if head_dim % block_size != 0:
        raise ValueError(f"block_size={block_size} must divide head_dim={head_dim}")
    return block_size


def _validate_stride(predictor_stride: int, seq_len: int) -> int:
    predictor_stride = int(predictor_stride)
    if predictor_stride <= 0:
        raise ValueError(f"predictor_stride must be positive, got {predictor_stride}")
    return min(predictor_stride, seq_len)
