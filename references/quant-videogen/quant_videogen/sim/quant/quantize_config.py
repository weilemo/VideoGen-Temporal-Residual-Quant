from dataclasses import dataclass


@dataclass
class QuantizeConfig:
    """Configuration for model quantization settings."""

    quant_type: str = "none"
    """Quantization type: 'none', 'naive-fp4', 'kmeans-fp4'."""

    # KV cache quantization parameters
    cache_num_k_centroids: int = 256
    """Number of K-Means centroids for K tensor (used in kmeans and nstages-kmeans)."""

    cache_num_v_centroids: int = 256
    """Number of K-Means centroids for V tensor (used in kmeans and nstages-kmeans)."""

    kmeans_max_iters: int = 100
    """Maximum iterations for K-Means clustering."""

    quant_block_size: int = 16
    """Block size for quantization."""

    # PRQ (nstages-kmeans) specific parameters
    num_prq_stages: int = 4
    """Number of PRQ stages for nstages-kmeans quantization."""
