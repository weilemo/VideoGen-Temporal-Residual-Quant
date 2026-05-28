from functools import lru_cache
import os
import torch
from enum import Enum
import re
import math
from .sim.quant.lowbit_quantize import (
    nvfp4_e2m1_quantize_triton,
    blockwise_int4_quantize_triton,
    blockwise_int3_quantize_triton,
    blockwise_int2_quantize_triton,
)
from .sim.quant.quantize_config import QuantizeConfig

from .functions import (
    kmeans_quantize_tensor,
    prq_quantize_tensor,
    triton_prq_quantize_tensor,
)
from .packed_naive import packed_naive_quantize_tensor
from .real.hrq import hrq_quantize_tensor

# ── Per-process predictor param cache ──────────────────────────────
_HRQ_PREDICTOR_CACHE: dict[str, dict | None] = {}


def _resolve_hrq_params_path(quant_config, mode: str) -> str | None:
    explicit = getattr(quant_config, "hrq_predictor_params_path", None)
    if explicit:
        return explicit
    params_dir = os.environ.get("HRQ_PREDICTOR_PARAMS_DIR", "assets/hrq_predictors")
    return os.path.join(params_dir, f"{mode}_self_forcing_dmd.pt")


def _load_hrq_predictor_params_lazy(quant_config) -> dict | None:
    """Lazy-load and cache predictor params from disk; returns None for identity."""
    mode = getattr(quant_config, "hrq_predictor_mode", "identity")
    if mode == "identity":
        return None
    path = _resolve_hrq_params_path(quant_config, mode)
    if path is None:
        return None
    if path in _HRQ_PREDICTOR_CACHE:
        return _HRQ_PREDICTOR_CACHE[path]
    if not os.path.exists(path):
        print(f"[HRQ] WARNING: predictor params not found at {path!r}; "
              "falling back to identity predictor.")
        _HRQ_PREDICTOR_CACHE[path] = None
        return None
    params = torch.load(path, map_location="cpu", weights_only=False)
    _HRQ_PREDICTOR_CACHE[path] = params
    print(f"[HRQ] Loaded predictor params ({mode}) from {path!r}")
    return params


def _get_layer_predictor_params(quant_config, layer_idx: int | None,
                                 kv_key: str, head_ids=None) -> dict | None:
    """Return predictor_params for one hrq_quantize_tensor call (layer + K or V).

    For affine_channel, slices alpha/beta by *head_ids* when provided.
    For tiny_mlp, the MLP is head-agnostic so no slicing is needed.
    """
    if layer_idx is None:
        return None
    full = _load_hrq_predictor_params_lazy(quant_config)
    if full is None:
        return None
    params = full.get(kv_key, {}).get(layer_idx)
    if params is None:
        return None
    if head_ids is not None and "alpha" in params:
        idx = torch.tensor(list(head_ids), dtype=torch.long)
        params = {
            "alpha": params["alpha"][idx],
            "beta": params["beta"][idx],
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
    HRQ = "hrq"


def get_quantize_fn(quant_type: str, quant_config: QuantizeConfig):
    if quant_type in ["naive-fp4", "kmeans-fp4", "nstages-kmeans-fp4", "nstages-kmeans-fp4-clip"]:
        def quantize_fn(x):
            """Quantization function - replace this to use different methods."""
            return nvfp4_e2m1_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
            )

    elif quant_type == "kmeans-fp4-clip":
        def quantize_fn(x):
            """Quantization function with percentile clipping."""
            return nvfp4_e2m1_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
                use_percentile_clipping=True,
                percentile=99.0,
            )

    elif quant_type in ["naive-int4", "kmeans-int4", "nstages-kmeans-int4", "nstages-kmeans-int4-clip"]:
        def quantize_fn(x):
            """Quantization function - replace this to use different methods."""
            return blockwise_int4_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
            )

    elif quant_type == "kmeans-int4-clip":
        def quantize_fn(x):
            """Quantization function with percentile clipping."""
            return blockwise_int4_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
                use_percentile_clipping=True,
                percentile=99.0,
            )

    elif quant_type in ["naive-int3", "kmeans-int3", "nstages-kmeans-int3", "nstages-kmeans-int3-clip"]:
        def quantize_fn(x):
            """Quantization function - replace this to use different methods."""
            return blockwise_int3_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
            )

    elif quant_type == "kmeans-int3-clip":
        def quantize_fn(x):
            """Quantization function with percentile clipping."""
            return blockwise_int3_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
                use_percentile_clipping=True,
                percentile=99.0,
            )

    elif quant_type in ["naive-int2", "kmeans-int2", "nstages-kmeans-int2", "nstages-kmeans-int2-clip"]:
        def quantize_fn(x):
            """Quantization function - replace this to use different methods."""
            return blockwise_int2_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
            )

    elif quant_type == "kmeans-int2-clip":
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
    elif quant_type in ["packed-naive-int2", "packed-naive-int4", "packed-naive-int8"] or quant_type.startswith("hrq"):
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
    elif quant_type.startswith("hrq"):
        quantize_type = QuantizeFunctions.HRQ
    else:
        quantize_type = QuantizeFunctions.NAIVE

    return quantize_type


def compress_kv_cache(k: torch.Tensor, v: torch.Tensor, quant_type: str, quant_config: QuantizeConfig, quantize_fn: callable,
                      layer_idx: int | None = None, head_ids=None):
    quantize_type = get_quantize_type(quant_type)

    if quantize_type == QuantizeFunctions.NSTAGE_KMEANS:
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
    elif quantize_type == QuantizeFunctions.HRQ:
        num_bits = quantize_fn(k)
        _predictor_mode = getattr(quant_config, "hrq_predictor_mode", "identity")
        _k_params = _get_layer_predictor_params(quant_config, layer_idx, "K", head_ids)
        _v_params = _get_layer_predictor_params(quant_config, layer_idx, "V", head_ids)
        k_quant = hrq_quantize_tensor(
            k,
            num_bits=num_bits,
            block_size=getattr(quant_config, "hrq_group_size", quant_config.quant_block_size),
            anchor_bits=getattr(quant_config, "hrq_anchor_bits", 4),
            predictor_stride=getattr(quant_config, "hrq_predictor_stride", 1560),
            predictor_mode=_predictor_mode,
            predictor_params=_k_params,
            scale_precision=getattr(quant_config, "hrq_scale_precision", torch.bfloat16),
            residual_quant_mode=getattr(quant_config, "hrq_residual_quant_mode", "asym_zero_point"),
        )
        v_quant = hrq_quantize_tensor(
            v,
            num_bits=num_bits,
            block_size=getattr(quant_config, "hrq_group_size", quant_config.quant_block_size),
            anchor_bits=getattr(quant_config, "hrq_anchor_bits", 4),
            predictor_stride=getattr(quant_config, "hrq_predictor_stride", 1560),
            predictor_mode=_predictor_mode,
            predictor_params=_v_params,
            scale_precision=getattr(quant_config, "hrq_scale_precision", torch.bfloat16),
            residual_quant_mode=getattr(quant_config, "hrq_residual_quant_mode", "asym_zero_point"),
        )
    else:
        raise ValueError(f"Unsupported quant type: {quant_type}")

    return k_quant, v_quant
