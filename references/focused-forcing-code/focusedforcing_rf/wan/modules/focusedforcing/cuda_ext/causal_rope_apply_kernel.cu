#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int B_FIXED = 1;
constexpr int T_FIXED = 1560;
constexpr int N_FIXED = 12;
constexpr int D_FIXED = 128;
constexpr int C_HALF = 64;
constexpr int C0 = 22;
constexpr int C1 = 21;
constexpr int C2 = 21;

// blockDim = (64, TILE_T), one thread handles one (token, complex_pair)
template <typename scalar_t, int TILE_T>
__global__ void causal_rope_apply_kernel(
    const scalar_t* __restrict__ q,         // [1,L,12,128], contiguous, nullable if !do_q
    const scalar_t* __restrict__ k,         // [1,L,12,128], contiguous, nullable if !do_k
    const int32_t* __restrict__ grid_sizes, // [1,3], int32 on CUDA
    const float* __restrict__ freqs0_re,    // flattened [Tmax * 22]
    const float* __restrict__ freqs0_im,
    const float* __restrict__ freqs1_re,    // flattened [gh_max * 21]
    const float* __restrict__ freqs1_im,
    const float* __restrict__ freqs2_re,    // flattened [gw_max * 21]
    const float* __restrict__ freqs2_im,
    scalar_t* __restrict__ out_q,           // [1,L,12,128], nullable if !do_q
    scalar_t* __restrict__ out_k,           // [1,L,12,128], nullable if !do_k
    int start_frame,
    int L,
    bool do_q,
    bool do_k
) {
    const int pair_idx = threadIdx.x;   // 0..63
    const int tok_lane = threadIdx.y;   // 0..TILE_T-1
    const int head_idx = blockIdx.y;    // 0..11
    const int tok_idx = blockIdx.x * TILE_T + tok_lane;

    if (pair_idx >= C_HALF || tok_idx >= L || head_idx >= N_FIXED) {
        return;
    }

    const int32_t f  = grid_sizes[0];
    const int32_t gh = grid_sizes[1];
    const int32_t gw = grid_sizes[2];

    const int32_t seq_len = f * gh * gw;
    const bool in_seq = tok_idx < seq_len;

    float freq_re = 1.0f;
    float freq_im = 0.0f;

    // IMPORTANT:
    // Only access frequency tables for tokens inside valid seq_len.
    // This matches the Python reference:
    //   x_i = rope(x[i, :seq_len])
    //   x_i = torch.cat([x_i, x[i, seq_len:]])
    if (in_seq) {
        const int32_t hw = gh * gw;
        const int32_t frame_idx = tok_idx / hw;
        const int32_t hw_idx = tok_idx % hw;
        const int32_t h_idx = hw_idx / gw;
        const int32_t w_idx = hw_idx % gw;

        if (pair_idx < C0) {
            const int32_t t = start_frame + frame_idx;
            const int idx = t * C0 + pair_idx;
            freq_re = freqs0_re[idx];
            freq_im = freqs0_im[idx];
        } else if (pair_idx < C0 + C1) {
            const int local = pair_idx - C0;
            const int idx = h_idx * C1 + local;
            freq_re = freqs1_re[idx];
            freq_im = freqs1_im[idx];
        } else {
            const int local = pair_idx - C0 - C1;
            const int idx = w_idx * C2 + local;
            freq_re = freqs2_re[idx];
            freq_im = freqs2_im[idx];
        }
    }

    const int d_re = 2 * pair_idx;
    const int d_im = d_re + 1;

    // contiguous offset for [1, L, N, D]
    const int base = ((tok_idx * N_FIXED) + head_idx) * D_FIXED;

    if (do_q) {
        const float q_re = static_cast<float>(q[base + d_re]);
        const float q_im = static_cast<float>(q[base + d_im]);

        float oq_re, oq_im;
        if (in_seq) {
            oq_re = q_re * freq_re - q_im * freq_im;
            oq_im = q_re * freq_im + q_im * freq_re;
        } else {
            oq_re = q_re;
            oq_im = q_im;
        }

        out_q[base + d_re] = static_cast<scalar_t>(oq_re);
        out_q[base + d_im] = static_cast<scalar_t>(oq_im);
    }

    if (do_k) {
        const float k_re = static_cast<float>(k[base + d_re]);
        const float k_im = static_cast<float>(k[base + d_im]);

        float ok_re, ok_im;
        if (in_seq) {
            ok_re = k_re * freq_re - k_im * freq_im;
            ok_im = k_re * freq_im + k_im * freq_re;
        } else {
            ok_re = k_re;
            ok_im = k_im;
        }

        out_k[base + d_re] = static_cast<scalar_t>(ok_re);
        out_k[base + d_im] = static_cast<scalar_t>(ok_im);
    }
}

template <typename scalar_t>
void launch_impl(
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
    int start_frame,
    bool do_q,
    bool do_k
) {
    constexpr int TILE_T = 4;
    const int L = do_q ? static_cast<int>(q.size(1))
                       : static_cast<int>(k.size(1));

    dim3 block(64, TILE_T);
    dim3 grid((L + TILE_T - 1) / TILE_T, N_FIXED);

    auto stream = at::cuda::getCurrentCUDAStream();

    const scalar_t* q_ptr = do_q ? q.data_ptr<scalar_t>() : nullptr;
    const scalar_t* k_ptr = do_k ? k.data_ptr<scalar_t>() : nullptr;
    scalar_t* out_q_ptr = do_q ? out_q.data_ptr<scalar_t>() : nullptr;
    scalar_t* out_k_ptr = do_k ? out_k.data_ptr<scalar_t>() : nullptr;

    causal_rope_apply_kernel<scalar_t, TILE_T><<<grid, block, 0, stream>>>(
        q_ptr,
        k_ptr,
        grid_sizes.data_ptr<int32_t>(),
        freqs0_re.data_ptr<float>(),
        freqs0_im.data_ptr<float>(),
        freqs1_re.data_ptr<float>(),
        freqs1_im.data_ptr<float>(),
        freqs2_re.data_ptr<float>(),
        freqs2_im.data_ptr<float>(),
        out_q_ptr,
        out_k_ptr,
        start_frame,
        L,
        do_q,
        do_k
    );
}

} // namespace

void causal_rope_apply_cuda(
    torch::Tensor q,          // can be undefined if do_q == false
    torch::Tensor k,          // can be undefined if do_k == false
    torch::Tensor grid_sizes, // must be [1,3], int32, CUDA
    torch::Tensor freqs0_re,
    torch::Tensor freqs0_im,
    torch::Tensor freqs1_re,
    torch::Tensor freqs1_im,
    torch::Tensor freqs2_re,
    torch::Tensor freqs2_im,
    torch::Tensor out_q,      // can be undefined if do_q == false
    torch::Tensor out_k,      // can be undefined if do_k == false
    int start_frame,
    bool do_q,
    bool do_k
) {
    TORCH_CHECK(do_q || do_k, "At least one of do_q or do_k must be true");

    // common checks
    TORCH_CHECK(grid_sizes.is_cuda(), "grid_sizes must be CUDA tensor");
    TORCH_CHECK(freqs0_re.is_cuda() && freqs0_im.is_cuda(), "freqs0 must be CUDA tensors");
    TORCH_CHECK(freqs1_re.is_cuda() && freqs1_im.is_cuda(), "freqs1 must be CUDA tensors");
    TORCH_CHECK(freqs2_re.is_cuda() && freqs2_im.is_cuda(), "freqs2 must be CUDA tensors");

    TORCH_CHECK(grid_sizes.scalar_type() == torch::kInt32, "grid_sizes must be int32");
    TORCH_CHECK(freqs0_re.scalar_type() == torch::kFloat32 && freqs0_im.scalar_type() == torch::kFloat32,
                "freqs0 must be float32");
    TORCH_CHECK(freqs1_re.scalar_type() == torch::kFloat32 && freqs1_im.scalar_type() == torch::kFloat32,
                "freqs1 must be float32");
    TORCH_CHECK(freqs2_re.scalar_type() == torch::kFloat32 && freqs2_im.scalar_type() == torch::kFloat32,
                "freqs2 must be float32");

    TORCH_CHECK(grid_sizes.is_contiguous(), "grid_sizes must be contiguous");
    TORCH_CHECK(freqs0_re.is_contiguous() && freqs0_im.is_contiguous(), "freqs0 must be contiguous");
    TORCH_CHECK(freqs1_re.is_contiguous() && freqs1_im.is_contiguous(), "freqs1 must be contiguous");
    TORCH_CHECK(freqs2_re.is_contiguous() && freqs2_im.is_contiguous(), "freqs2 must be contiguous");

    TORCH_CHECK(grid_sizes.dim() == 2, "grid_sizes must be [1, 3]");
    TORCH_CHECK(grid_sizes.size(0) == 1 && grid_sizes.size(1) == 3,
                "grid_sizes must have shape [1, 3]");

    int L = -1;
    c10::ScalarType dtype = c10::ScalarType::Undefined;
    c10::Device device = grid_sizes.device();

    if (do_q) {
        TORCH_CHECK(q.defined(), "q must be defined when do_q is true");
        TORCH_CHECK(out_q.defined(), "out_q must be defined when do_q is true");
        TORCH_CHECK(q.is_cuda(), "q must be CUDA tensor");
        TORCH_CHECK(out_q.is_cuda(), "out_q must be CUDA tensor");
        TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
        TORCH_CHECK(out_q.is_contiguous(), "out_q must be contiguous");
        TORCH_CHECK(q.dim() == 4, "q must be [B, L, N, D]");
        TORCH_CHECK(q.size(0) == B_FIXED, "Only batch size 1 is supported for q");
        TORCH_CHECK(q.size(2) == N_FIXED, "q head dim N must be 12");
        TORCH_CHECK(q.size(3) == D_FIXED, "q head size D must be 128");
        TORCH_CHECK(out_q.sizes() == q.sizes(), "out_q must have same shape as q");

        L = static_cast<int>(q.size(1));
        dtype = q.scalar_type();
        device = q.device();
    }

    if (do_k) {
        TORCH_CHECK(k.defined(), "k must be defined when do_k is true");
        TORCH_CHECK(out_k.defined(), "out_k must be defined when do_k is true");
        TORCH_CHECK(k.is_cuda(), "k must be CUDA tensor");
        TORCH_CHECK(out_k.is_cuda(), "out_k must be CUDA tensor");
        TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
        TORCH_CHECK(out_k.is_contiguous(), "out_k must be contiguous");
        TORCH_CHECK(k.dim() == 4, "k must be [B, L, N, D]");
        TORCH_CHECK(k.size(0) == B_FIXED, "Only batch size 1 is supported for k");
        TORCH_CHECK(k.size(2) == N_FIXED, "k head dim N must be 12");
        TORCH_CHECK(k.size(3) == D_FIXED, "k head size D must be 128");
        TORCH_CHECK(out_k.sizes() == k.sizes(), "out_k must have same shape as k");

        if (L == -1) {
            L = static_cast<int>(k.size(1));
            dtype = k.scalar_type();
            device = k.device();
        } else {
            TORCH_CHECK(k.size(1) == L, "q and k must have the same sequence length when both are used");
            TORCH_CHECK(k.scalar_type() == dtype, "q and k must have the same dtype when both are used");
            TORCH_CHECK(k.device() == device, "q and k must be on the same device when both are used");
        }
    }

    TORCH_CHECK(L > 0, "L must be positive");
    TORCH_CHECK(L % T_FIXED == 0, "sequence length L must be a multiple of 1560, got ", L);

    // NOTE:
    // We intentionally do NOT read grid_sizes contents on host side here,
    // because grid_sizes is a CUDA tensor. Reading grid_sizes.data_ptr<int32_t>()
    // on CPU side would dereference a device pointer and may segfault.

    if (do_q) {
        TORCH_CHECK(out_q.scalar_type() == dtype, "out_q dtype must match q dtype");
        TORCH_CHECK(out_q.device() == device, "out_q device must match q device");
    }
    if (do_k) {
        TORCH_CHECK(out_k.scalar_type() == dtype, "out_k dtype must match k dtype");
        TORCH_CHECK(out_k.device() == device, "out_k device must match k device");
    }

    switch (dtype) {
        case torch::kFloat32:
            launch_impl<float>(
                q, k, grid_sizes,
                freqs0_re, freqs0_im,
                freqs1_re, freqs1_im,
                freqs2_re, freqs2_im,
                out_q, out_k, start_frame,
                do_q, do_k
            );
            break;
        case torch::kFloat16:
            launch_impl<c10::Half>(
                q, k, grid_sizes,
                freqs0_re, freqs0_im,
                freqs1_re, freqs1_im,
                freqs2_re, freqs2_im,
                out_q, out_k, start_frame,
                do_q, do_k
            );
            break;
        case torch::kBFloat16:
            launch_impl<c10::BFloat16>(
                q, k, grid_sizes,
                freqs0_re, freqs0_im,
                freqs1_re, freqs1_im,
                freqs2_re, freqs2_im,
                out_q, out_k, start_frame,
                do_q, do_k
            );
            break;
        default:
            TORCH_CHECK(false, "Only float32 / float16 / bfloat16 are supported");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
