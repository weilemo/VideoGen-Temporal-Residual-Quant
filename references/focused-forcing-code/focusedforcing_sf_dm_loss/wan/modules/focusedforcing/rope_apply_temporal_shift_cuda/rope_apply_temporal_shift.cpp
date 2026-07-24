#include <torch/extension.h>

void rope_apply_temporal_shift_cuda(
    torch::Tensor k_chunk,
    torch::Tensor mult_re,
    torch::Tensor mult_im
);

void rope_apply_temporal_shift(
    torch::Tensor k_chunk,
    torch::Tensor mult_re,
    torch::Tensor mult_im
) {
    TORCH_CHECK(k_chunk.is_cuda(), "k_chunk must be CUDA");
    TORCH_CHECK(mult_re.is_cuda(), "mult_re must be CUDA");
    TORCH_CHECK(mult_im.is_cuda(), "mult_im must be CUDA");

    TORCH_CHECK(k_chunk.is_contiguous(), "k_chunk must be contiguous");
    TORCH_CHECK(mult_re.is_contiguous(), "mult_re must be contiguous");
    TORCH_CHECK(mult_im.is_contiguous(), "mult_im must be contiguous");

    TORCH_CHECK(k_chunk.dim() == 4, "k_chunk must be [B, L, H, D]");
    TORCH_CHECK(k_chunk.size(3) == 128, "fixed-shape kernel requires D=128");

    TORCH_CHECK(mult_re.dim() == 1 && mult_im.dim() == 1, "mult must be 1D");
    TORCH_CHECK(mult_re.numel() == 22 && mult_im.numel() == 22, "fixed-shape kernel requires t_c=22");

    TORCH_CHECK(mult_re.scalar_type() == torch::kFloat64, "mult_re must be float64");
    TORCH_CHECK(mult_im.scalar_type() == torch::kFloat64, "mult_im must be float64");

    auto st = k_chunk.scalar_type();
    TORCH_CHECK(
        st == torch::kBFloat16 || st == torch::kFloat16 || st == torch::kFloat32,
        "k_chunk must be bf16/fp16/fp32"
    );

    rope_apply_temporal_shift_cuda(k_chunk, mult_re, mult_im);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rope_apply_temporal_shift",
        &rope_apply_temporal_shift,
        "RoPE temporal shift inplace (fixed shape, fp64 compute)"
    );
}
