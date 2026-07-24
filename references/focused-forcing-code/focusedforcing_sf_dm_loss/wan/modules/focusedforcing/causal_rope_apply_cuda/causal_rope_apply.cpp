#include <torch/extension.h>

void causal_rope_apply_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor grid_sizes,
    torch::Tensor freqs0_re,
    torch::Tensor freqs0_im,
    torch::Tensor freqs1_re,
    torch::Tensor freqs1_im,
    torch::Tensor freqs2_re,
    torch::Tensor freqs2_im,
    torch::Tensor out_q,
    torch::Tensor out_k,
    int start_frame
);

void causal_rope_apply(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor grid_sizes,
    torch::Tensor freqs0_re,
    torch::Tensor freqs0_im,
    torch::Tensor freqs1_re,
    torch::Tensor freqs1_im,
    torch::Tensor freqs2_re,
    torch::Tensor freqs2_im,
    torch::Tensor out_q,
    torch::Tensor out_k,
    int start_frame
) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(grid_sizes.is_cuda(), "grid_sizes must be CUDA");
    TORCH_CHECK(freqs0_re.is_cuda(), "freqs0_re must be CUDA");
    TORCH_CHECK(freqs0_im.is_cuda(), "freqs0_im must be CUDA");
    TORCH_CHECK(freqs1_re.is_cuda(), "freqs1_re must be CUDA");
    TORCH_CHECK(freqs1_im.is_cuda(), "freqs1_im must be CUDA");
    TORCH_CHECK(freqs2_re.is_cuda(), "freqs2_re must be CUDA");
    TORCH_CHECK(freqs2_im.is_cuda(), "freqs2_im must be CUDA");
    TORCH_CHECK(out_q.is_cuda(), "out_q must be CUDA");
    TORCH_CHECK(out_k.is_cuda(), "out_k must be CUDA");

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(grid_sizes.is_contiguous(), "grid_sizes must be contiguous");
    TORCH_CHECK(out_q.is_contiguous(), "out_q must be contiguous");
    TORCH_CHECK(out_k.is_contiguous(), "out_k must be contiguous");

    TORCH_CHECK(q.sizes() == k.sizes(), "q and k shapes must match");
    TORCH_CHECK(q.dim() == 4, "q/k must be [1,4680,12,128]");
    TORCH_CHECK(q.size(0) == 1, "B must be 1");
    TORCH_CHECK(q.size(1) == 4680, "L must be 4680");
    TORCH_CHECK(q.size(2) == 12, "N must be 12");
    TORCH_CHECK(q.size(3) == 128, "D must be 128");

    TORCH_CHECK(out_q.sizes() == q.sizes(), "out_q shape mismatch");
    TORCH_CHECK(out_k.sizes() == q.sizes(), "out_k shape mismatch");

    TORCH_CHECK(grid_sizes.dim() == 2 && grid_sizes.size(0) == 1 && grid_sizes.size(1) == 3,
                "grid_sizes must be [1,3]");
    TORCH_CHECK(grid_sizes.scalar_type() == torch::kInt32, "grid_sizes must be int32");

    TORCH_CHECK(freqs0_re.scalar_type() == torch::kFloat32, "freqs0_re must be float32");
    TORCH_CHECK(freqs0_im.scalar_type() == torch::kFloat32, "freqs0_im must be float32");
    TORCH_CHECK(freqs1_re.scalar_type() == torch::kFloat32, "freqs1_re must be float32");
    TORCH_CHECK(freqs1_im.scalar_type() == torch::kFloat32, "freqs1_im must be float32");
    TORCH_CHECK(freqs2_re.scalar_type() == torch::kFloat32, "freqs2_re must be float32");
    TORCH_CHECK(freqs2_im.scalar_type() == torch::kFloat32, "freqs2_im must be float32");

    TORCH_CHECK(
        q.scalar_type() == torch::kFloat32 ||
        q.scalar_type() == torch::kFloat16 ||
        q.scalar_type() == torch::kBFloat16,
        "q/k must be float32/float16/bfloat16"
    );
    TORCH_CHECK(k.scalar_type() == q.scalar_type(), "k dtype must match q");
    TORCH_CHECK(out_q.scalar_type() == q.scalar_type(), "out_q dtype must match q");
    TORCH_CHECK(out_k.scalar_type() == q.scalar_type(), "out_k dtype must match q");

    causal_rope_apply_cuda(
        q, k, grid_sizes,
        freqs0_re, freqs0_im,
        freqs1_re, freqs1_im,
        freqs2_re, freqs2_im,
        out_q, out_k,
        start_frame
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("causal_rope_apply", &causal_rope_apply, "Causal RoPE CUDA");
}
