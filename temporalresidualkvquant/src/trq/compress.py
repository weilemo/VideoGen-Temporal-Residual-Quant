import os
import numpy as np
import torch
from enum import Enum
import re
from .sim.quant.quantize_config import QuantizeConfig
from .real.trq import normalize_trq_predictor_mode, trq_quantize_tensor

# Per-process predictor parameter cache.
_TRQ_PREDICTOR_CACHE: dict[str, dict] = {}


def _config_value(
    quant_config,
    canonical: str,
    legacy: str | tuple[str, ...] | None,
    default=None,
):
    value = getattr(quant_config, canonical, None)
    if value is not None and value != "":
        return value
    if legacy is not None:
        names = (legacy,) if isinstance(legacy, str) else legacy
        for name in names:
            value = getattr(quant_config, name, None)
            if value is not None and value != "":
                return value
    return default


def _resolve_trq_params_path(quant_config, mode: str, kv_key: str) -> str:
    role_attr = "trq_v_predictor_params_path" if kv_key == "V" else "trq_k_predictor_params_path"
    explicit = getattr(quant_config, role_attr, None)
    if not explicit:
        explicit = _config_value(
            quant_config,
            "trq_predictor_params_path",
            (
                "hrq_predictor_params_path",
                "s2pp_v_affine_path" if kv_key == "V" else "s2pp_affine_path",
            ),
            None,
        )
    if explicit:
        return explicit
    params_dir = os.environ.get(
        "TRQ_PREDICTOR_PARAMS_DIR",
        os.environ.get("HRQ_PREDICTOR_PARAMS_DIR", "assets/trq_predictors"),
    )
    return os.path.join(params_dir, f"{mode}_self_forcing_dmd.pt")


def _load_trq_predictor_params_lazy(quant_config, mode: str, kv_key: str) -> dict | None:
    """Lazy-load predictor params and fail closed for non-identity modes."""
    if mode == "identity":
        return None
    path = _resolve_trq_params_path(quant_config, mode, kv_key)
    if path in _TRQ_PREDICTOR_CACHE:
        return _TRQ_PREDICTOR_CACHE[path]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"TRQ predictor_mode={mode!r} requires parameters, but {path!r} does not exist"
        )
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as data:
            if "alpha" not in data or "beta" not in data:
                raise ValueError(f"TRQ affine .npz must contain alpha and beta: {path}")
            params = {
                "alpha": torch.from_numpy(data["alpha"].astype(np.float32)),
                "beta": torch.from_numpy(data["beta"].astype(np.float32)),
            }
    else:
        params = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(params, dict):
        raise ValueError(f"TRQ predictor file must contain a dictionary: {path}")
    _TRQ_PREDICTOR_CACHE[path] = params
    print(f"[TRQ] Loaded predictor params ({mode}) from {path!r}")
    return params


def _get_layer_predictor_params(
    quant_config,
    mode: str,
    layer_idx: int | None,
    kv_key: str,
    head_ids=None,
) -> dict | None:
    """Return K/V predictor parameters, preserving supported shape variants."""
    full = _load_trq_predictor_params_lazy(quant_config, mode, kv_key)
    if full is None:
        return None
    if kv_key in full:
        if layer_idx is None:
            raise ValueError(f"TRQ predictor file is layer-specific for {kv_key}; layer_idx is required")
        layer_table = full[kv_key]
        params = layer_table.get(layer_idx, layer_table.get(str(layer_idx)))
        if params is None:
            raise KeyError(f"TRQ predictor file has no {kv_key} parameters for layer {layer_idx}")
    else:
        params = full
    if "alpha" not in params or "beta" not in params:
        raise ValueError(f"TRQ predictor parameters for {kv_key} require alpha and beta")
    if head_ids is not None:
        idx = torch.tensor(list(head_ids), dtype=torch.long)
        alpha = params["alpha"]
        beta = params["beta"]
        head_dim = 0 if alpha.ndim == 2 else 1 if alpha.ndim == 3 else None
        if head_dim is not None:
            alpha = alpha.index_select(head_dim, idx)
            beta = beta.index_select(head_dim, idx)
        params = {
            "alpha": alpha,
            "beta": beta,
        }
    return params


########################################################
# Entrypoints
########################################################

# Make it a enum
class QuantizeFunctions(Enum):
    NAIVE = "naive"
    KMEANS = "kmeans"
    NSTAGE_KMEANS = "prq"
    NSTAGE_KMEANS_CLIP = "prq_clip"
    TRITON_PRQ = "triton_prq"
    TRITON_PRQ_CLIP = "triton_prq_clip"
    PACKED_NAIVE = "packed_naive"
    TRQ = "trq"
    HRQ = "trq"


def get_quantize_fn(quant_type: str, quant_config: QuantizeConfig):
    if quant_type in ["naive-fp4", "kmeans-fp4", "nstages-kmeans-fp4", "nstages-kmeans-fp4-clip"]:
        from .sim.quant.lowbit_quantize import nvfp4_e2m1_quantize_triton

        def quantize_fn(x):
            """Quantization function - replace this to use different methods."""
            return nvfp4_e2m1_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
            )

    elif quant_type == "kmeans-fp4-clip":
        from .sim.quant.lowbit_quantize import nvfp4_e2m1_quantize_triton

        def quantize_fn(x):
            """Quantization function with percentile clipping."""
            return nvfp4_e2m1_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
                use_percentile_clipping=True,
                percentile=99.0,
            )

    elif quant_type in ["naive-int4", "kmeans-int4", "nstages-kmeans-int4", "nstages-kmeans-int4-clip"]:
        from .sim.quant.lowbit_quantize import blockwise_int4_quantize_triton

        def quantize_fn(x):
            """Quantization function - replace this to use different methods."""
            return blockwise_int4_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
            )

    elif quant_type == "kmeans-int4-clip":
        from .sim.quant.lowbit_quantize import blockwise_int4_quantize_triton

        def quantize_fn(x):
            """Quantization function with percentile clipping."""
            return blockwise_int4_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
                use_percentile_clipping=True,
                percentile=99.0,
            )

    elif quant_type in ["naive-int3", "kmeans-int3", "nstages-kmeans-int3", "nstages-kmeans-int3-clip"]:
        from .sim.quant.lowbit_quantize import blockwise_int3_quantize_triton

        def quantize_fn(x):
            """Quantization function - replace this to use different methods."""
            return blockwise_int3_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
            )

    elif quant_type == "kmeans-int3-clip":
        from .sim.quant.lowbit_quantize import blockwise_int3_quantize_triton

        def quantize_fn(x):
            """Quantization function with percentile clipping."""
            return blockwise_int3_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
                use_percentile_clipping=True,
                percentile=99.0,
            )

    elif quant_type in ["naive-int2", "kmeans-int2", "nstages-kmeans-int2", "nstages-kmeans-int2-clip"]:
        from .sim.quant.lowbit_quantize import blockwise_int2_quantize_triton

        def quantize_fn(x):
            """Quantization function - replace this to use different methods."""
            return blockwise_int2_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
            )

    elif quant_type == "kmeans-int2-clip":
        from .sim.quant.lowbit_quantize import blockwise_int2_quantize_triton

        def quantize_fn(x):
            """Quantization function with percentile clipping."""
            return blockwise_int2_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
                use_percentile_clipping=True,
                percentile=99.0,
            )
    elif quant_type in ["triton-nstages-kmeans-int2", "triton-nstages-kmeans-int2-clip", "triton-nstages-kmeans-int4", "triton-nstages-kmeans-int4-clip"]:
        """Do not quantize here"""
        def quantize_fn(x):
            m = re.search(r'int(\d+)', quant_config.quant_type)
            if m is None:
                raise ValueError(f"Cannot identify num_bits from {quant_config.quant_type}")
            num_bits = int(m.group(1))
            return num_bits
    elif quant_type in ["packed-naive-int2", "packed-naive-int4", "packed-naive-int8"] or quant_type.startswith(("trq", "hrq", "s2pp")):
        """Packed/real quantization is handled in compress_kv_cache."""
        def quantize_fn(x):
            m = re.search(r'int(\d+)', quant_config.quant_type)
            if m is None:
                raise ValueError(f"Cannot identify num_bits from {quant_config.quant_type}")
            return int(m.group(1))
        
    else:
        raise ValueError(
            f"Unsupported quant type: {quant_type}"
        )
        
    return quantize_fn


def get_quantize_type(quant_type: str):
    # ==========================================================
    # Determine preprocessing mode
    # ==========================================================
    if quant_type in [
        "kmeans-fp4",
        "kmeans-fp4-clip",
        "kmeans-int4",
        "kmeans-int4-clip",
        "kmeans-int3",
        "kmeans-int3-clip",
        "kmeans-int2",
        "kmeans-int2-clip",
    ]:
        quantize_type = QuantizeFunctions.KMEANS

    elif quant_type in [
        "nstages-kmeans-fp4",
        "nstages-kmeans-int4",
        "nstages-kmeans-int3",
        "nstages-kmeans-int2",
    ]:
        quantize_type = QuantizeFunctions.NSTAGE_KMEANS

    elif quant_type in [
        "nstages-kmeans-fp4-clip",
        "nstages-kmeans-int4-clip",
        "nstages-kmeans-int3-clip",
        "nstages-kmeans-int2-clip",
    ]:
        quantize_type = QuantizeFunctions.NSTAGE_KMEANS_CLIP
    elif quant_type in [
        "triton-nstages-kmeans-int2",
        "triton-nstages-kmeans-int4",
    ]:
        quantize_type = QuantizeFunctions.TRITON_PRQ
    elif quant_type in [
        "triton-nstages-kmeans-int2-clip",
        "triton-nstages-kmeans-int4-clip",
    ]:
        quantize_type = QuantizeFunctions.TRITON_PRQ_CLIP
    elif quant_type in [
        "packed-naive-int2",
        "packed-naive-int4",
        "packed-naive-int8",
    ]:
        quantize_type = QuantizeFunctions.PACKED_NAIVE
    elif quant_type.startswith(("trq", "hrq", "s2pp")):
        quantize_type = QuantizeFunctions.TRQ
    else:
        quantize_type = QuantizeFunctions.NAIVE

    return quantize_type


def compress_kv_cache(k: torch.Tensor, v: torch.Tensor, quant_type: str, quant_config: QuantizeConfig, quantize_fn: callable,
                      layer_idx: int | None = None, head_ids=None):
    quantize_type = get_quantize_type(quant_type)

    if quantize_type == QuantizeFunctions.NSTAGE_KMEANS:
        from .functions import prq_quantize_tensor

        # Apply PRQ (multi-stage K-Means) based quantization
        k_quant = prq_quantize_tensor(
            k,
            num_stages=quant_config.num_prq_stages,
            codebook_size=quant_config.cache_num_k_centroids,
            kmeans_max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
        )
        v_quant = prq_quantize_tensor(
            v,
            num_stages=quant_config.num_prq_stages,
            codebook_size=quant_config.cache_num_v_centroids,
            kmeans_max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
        )
    elif quantize_type == QuantizeFunctions.NSTAGE_KMEANS_CLIP:
        from .functions import prq_quantize_tensor

        # Apply PRQ (multi-stage K-Means) based quantization
        k_quant = prq_quantize_tensor(
            k,
            num_stages=quant_config.num_prq_stages,
            codebook_size=quant_config.cache_num_k_centroids,
            kmeans_max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
            use_percentile_clipping=True,
        )
        v_quant = prq_quantize_tensor(
            v,
            num_stages=quant_config.num_prq_stages,
            codebook_size=quant_config.cache_num_v_centroids,
            kmeans_max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
            use_percentile_clipping=True,
        )
    elif quantize_type == QuantizeFunctions.KMEANS:
        from .functions import kmeans_quantize_tensor

        # Apply K-Means based quantization
        k_quant = kmeans_quantize_tensor(
            k,
            num_centroids=quant_config.cache_num_k_centroids,
            kmeans_max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
        )
        v_quant = kmeans_quantize_tensor(
            v,
            num_centroids=quant_config.cache_num_v_centroids,
            kmeans_max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
        )
    elif quantize_type == QuantizeFunctions.TRITON_PRQ:
        from .functions import triton_prq_quantize_tensor

        # Apply Triton N-Stage K-Means based quantization
        k_quant = triton_prq_quantize_tensor(
            k,
            num_stages=quant_config.num_prq_stages,
            num_clusters=quant_config.cache_num_k_centroids,
            block_size=quant_config.quant_block_size,
            max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
        )
        v_quant = triton_prq_quantize_tensor(
            v,
            num_stages=quant_config.num_prq_stages,
            num_clusters=quant_config.cache_num_v_centroids,
            block_size=quant_config.quant_block_size,
            max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
        )
    elif quantize_type == QuantizeFunctions.TRITON_PRQ_CLIP:
        from .functions import triton_prq_quantize_tensor

        # Apply Triton N-Stage K-Means based quantization
        k_quant = triton_prq_quantize_tensor(
            k,
            num_stages=quant_config.num_prq_stages,
            num_clusters=quant_config.cache_num_k_centroids,
            block_size=quant_config.quant_block_size,
            max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
            use_percentile_clipping=True,
        )
        v_quant = triton_prq_quantize_tensor(
            v,
            num_stages=quant_config.num_prq_stages,
            num_clusters=quant_config.cache_num_v_centroids,
            block_size=quant_config.quant_block_size,
            max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
            use_percentile_clipping=True,
        )
    elif quantize_type == QuantizeFunctions.NAIVE:
        # ==========================================================
        # Direct quantization (no preprocessing)
        # ==========================================================
        k_quant = quantize_fn(k)
        v_quant = quantize_fn(v)
    elif quantize_type == QuantizeFunctions.PACKED_NAIVE:
        from .packed_naive import packed_naive_quantize_tensor

        num_bits = quantize_fn(k)
        k_quant = packed_naive_quantize_tensor(
            k,
            num_bits=num_bits,
            block_size=quant_config.quant_block_size,
        )
        v_quant = packed_naive_quantize_tensor(
            v,
            num_bits=num_bits,
            block_size=quant_config.quant_block_size,
        )
    elif quantize_type == QuantizeFunctions.TRQ:
        inferred_bits = quantize_fn(k)
        k_bits = int(_config_value(quant_config, "trq_k_bits", "s2pp_k_bits", 0) or inferred_bits)
        v_bits = int(_config_value(quant_config, "trq_v_bits", "s2pp_v_bits", 0) or inferred_bits)
        predictor_mode = _config_value(
            quant_config,
            "trq_predictor_mode",
            ("hrq_predictor_mode", "s2pp_predictor_mode"),
            "identity",
        )
        k_mode = _config_value(quant_config, "trq_k_predictor_mode", None, predictor_mode)
        v_mode = _config_value(quant_config, "trq_v_predictor_mode", "s2pp_v_predictor_mode", predictor_mode)
        if k_mode == "auto":
            k_mode = "identity" if "identity" in quant_type else "affine_channel"
        if v_mode == "auto":
            v_mode = k_mode
        cross_modes = {
            "cross_kv", "hybrid_kv_innovation", "k_to_v", "k2v",
            "k-to-v", "cross", "cross-kv", "kv_cross",
        }
        use_s2pp_cross_codec = quant_type.startswith("s2pp") and str(v_mode) in cross_modes
        if use_s2pp_cross_codec:
            from .real.s2pp import s2pp_quantize_tensor

            group_size = _config_value(
                quant_config,
                "trq_group_size",
                ("hrq_group_size", "s2pp_group_size"),
                getattr(quant_config, "quant_block_size", 16),
            )
            anchor_bits = _config_value(
                quant_config, "trq_anchor_bits", ("hrq_anchor_bits", "s2pp_anchor_bits"), 4
            )
            predictor_stride = _config_value(
                quant_config,
                "trq_predictor_stride",
                ("hrq_predictor_stride", "s2pp_predictor_stride"),
                1560,
            )
            scale_precision = _config_value(
                quant_config,
                "trq_scale_precision",
                ("hrq_scale_precision", "s2pp_scale_precision"),
                torch.bfloat16,
            )
            residual_quant_mode = _config_value(
                quant_config,
                "trq_residual_quant_mode",
                ("hrq_residual_quant_mode", "s2pp_residual_quant_mode"),
                "asym_zero_point",
            )
            v_params_path = _resolve_trq_params_path(quant_config, str(v_mode), "V")
            k_quant = s2pp_quantize_tensor(
                k,
                num_bits=k_bits,
                block_size=group_size,
                anchor_bits=anchor_bits,
                predictor_stride=predictor_stride,
                mode="identity",
                scale_precision=scale_precision,
                residual_quant_mode=residual_quant_mode,
                layer_idx=layer_idx,
                parameter_role="K",
            )
            v_quant = s2pp_quantize_tensor(
                v,
                num_bits=v_bits,
                block_size=group_size,
                anchor_bits=anchor_bits,
                predictor_stride=predictor_stride,
                affine_path=v_params_path,
                mode="cross_kv",
                scale_precision=scale_precision,
                residual_quant_mode=residual_quant_mode,
                source_state=k_quant,
                layer_idx=layer_idx,
                parameter_role="V",
            )
            # K and V live in separate ChunkedKVCache objects. Persist the
            # shared packed K object so V can decode without an external cache
            # pairing API; tensor storage remains shared in-process.
            v_quant["cross_source_state"] = k_quant
            return k_quant, v_quant
        k_mode = normalize_trq_predictor_mode(k_mode)
        v_mode = normalize_trq_predictor_mode(v_mode)
        k_params = _get_layer_predictor_params(quant_config, k_mode, layer_idx, "K", head_ids)
        v_params = _get_layer_predictor_params(quant_config, v_mode, layer_idx, "V", head_ids)
        group_size = _config_value(
            quant_config,
            "trq_group_size",
            ("hrq_group_size", "s2pp_group_size"),
            getattr(quant_config, "quant_block_size", 16),
        )
        anchor_bits = _config_value(
            quant_config, "trq_anchor_bits", ("hrq_anchor_bits", "s2pp_anchor_bits"), 4
        )
        predictor_stride = _config_value(
            quant_config,
            "trq_predictor_stride",
            ("hrq_predictor_stride", "s2pp_predictor_stride"),
            1560,
        )
        scale_precision = _config_value(
            quant_config,
            "trq_scale_precision",
            ("hrq_scale_precision", "s2pp_scale_precision"),
            torch.bfloat16,
        )
        residual_quant_mode = _config_value(
            quant_config,
            "trq_residual_quant_mode",
            ("hrq_residual_quant_mode", "s2pp_residual_quant_mode"),
            "asym_zero_point",
        )
        k_quant = trq_quantize_tensor(
            k,
            num_bits=k_bits,
            block_size=group_size,
            anchor_bits=anchor_bits,
            predictor_stride=predictor_stride,
            predictor_mode=k_mode,
            predictor_params=k_params,
            layer_idx=layer_idx,
            scale_precision=scale_precision,
            residual_quant_mode=residual_quant_mode,
        )
        v_quant = trq_quantize_tensor(
            v,
            num_bits=v_bits,
            block_size=group_size,
            anchor_bits=anchor_bits,
            predictor_stride=predictor_stride,
            predictor_mode=v_mode,
            predictor_params=v_params,
            layer_idx=layer_idx,
            scale_precision=scale_precision,
            residual_quant_mode=residual_quant_mode,
        )
    else:
        raise ValueError(f"Unsupported quant type: {quant_type}")

    return k_quant, v_quant
