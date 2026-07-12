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

    # TRQ (temporal residual quantization) parameters. ``None`` means use the
    # corresponding legacy HRQ field below.
    trq_group_size: int | None = None
    """Group size along head_dim for TRQ."""

    trq_anchor_bits: int | None = None
    """Bit width for the first TRQ predictor unit anchor."""

    trq_predictor_stride: int | None = None
    """Number of sequence tokens in one TRQ predictor unit."""

    trq_predictor_mode: str | None = None
    """Stable TRQ v1 predictor: identity or affine_channel."""

    trq_k_predictor_mode: str | None = None
    """Optional K-specific predictor override."""

    trq_v_predictor_mode: str | None = None
    """Optional V-specific predictor override."""

    trq_predictor_params_path: str | None = None
    """Shared .pt/.npz predictor path."""

    trq_k_predictor_params_path: str | None = None
    """Optional K-specific .pt/.npz predictor path."""

    trq_v_predictor_params_path: str | None = None
    """Optional V-specific .pt/.npz predictor path."""

    trq_k_bits: int = 0
    """Optional K residual bit override; zero uses quant_type."""

    trq_v_bits: int = 0
    """Optional V residual bit override; zero uses quant_type."""

    trq_scale_precision: str | None = None
    """Scale dtype used by TRQ."""

    trq_residual_quant_mode: str | None = None
    """TRQ v1 residual mode; currently asym_zero_point only."""

    # Legacy HRQ names are retained for old scripts and serialized configs.
    hrq_group_size: int = 64
    """Legacy alias for trq_group_size."""

    hrq_anchor_bits: int = 4
    """Legacy alias for trq_anchor_bits."""

    hrq_predictor_stride: int = 1560
    """Legacy alias for trq_predictor_stride."""

    hrq_predictor_mode: str = "identity"
    """Legacy alias for trq_predictor_mode."""

    hrq_predictor_params_path: str = ""
    """Legacy path to .pt file with fitted affine_channel predictor params.
    When empty, defaults to assets/trq_predictors/{mode}_self_forcing_dmd.pt
    (or overridden by TRQ_PREDICTOR_PARAMS_DIR; HRQ_PREDICTOR_PARAMS_DIR is a
    compatibility alias)."""

    hrq_scale_precision: str = "bf16"
    """Legacy alias for trq_scale_precision."""

    hrq_residual_quant_mode: str = "asym_zero_point"
    """Legacy alias for trq_residual_quant_mode."""
