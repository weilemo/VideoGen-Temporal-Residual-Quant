#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int B_FIXED = 1;
constexpr int L_FIXED = 4680;
constexpr int N_FIXED = 12;
constexpr int D_FIXED = 128;
constexpr int C_HALF = 64;
constexpr int C0 = 22;
constexpr int C1 = 21;
constexpr int C2 = 21;

// blockDim = (64, TILE_T), one thread handles one (token, complex_pair)
template <typename scalar_t, int TILE_T>
__global__ void causal_rope_apply_kernel(
    const scalar_t* __restrict__ q,      // [1,4680,12,128], contiguous
    const scalar_t* __restrict__ k,      // [1,4680,12,128], contiguous
    const int32_t* __restrict__ grid_sizes, // [1,3], int32 on CUDA
    const float* __restrict__ freqs0_re, // [1024 * 22]
    const float* __restrict__ freqs0_im,
    const float* __restrict__ freqs1_re, // [gh * 21] flattened
    const float* __restrict__ freqs1_im,
    const float* __restrict__ freqs2_re, // [gw * 21] flattened
    const float* __restrict__ freqs2_im,
    scalar_t* __restrict__ out_q,        // [1,4680,12,128]
    scalar_t* __restrict__ out_k,        // [1,4680,12,128]
    int start_frame
) {
    const int pair_idx = threadIdx.x;         // 0..63
    const int tok_lane = threadIdx.y;         // 0..TILE_T-1
    const int head_idx = blockIdx.y;          // 0..11
    const int tok_idx = blockIdx.x * TILE_T + tok_lane;

    if (pair_idx >= C_HALF || tok_idx >= L_FIXED || head_idx >= N_FIXED) {
        return;
    }

    const int32_t f  = grid_sizes[0];
    const int32_t gh = grid_sizes[1];
    const int32_t gw = grid_sizes[2];

    const int32_t seq_len = f * gh * gw;
    const bool in_seq = tok_idx < seq_len;

    const int32_t hw = gh * gw;
    const int32_t frame_idx = tok_idx / hw;
    const int32_t hw_idx = tok_idx % hw;
    const int32_t h_idx = hw_idx / gw;
    const int32_t w_idx = hw_idx % gw;

    float freq_re = 1.0f;
    float freq_im = 0.0f;

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

    const int d_re = 2 * pair_idx;
    const int d_im = d_re + 1;

    // contiguous offset for [1, L, N, D]
    const int base = ((tok_idx * N_FIXED) + head_idx) * D_FIXED;

    const float q_re = static_cast<float>(q[base + d_re]);
    const float q_im = static_cast<float>(q[base + d_im]);
    const float k_re = static_cast<float>(k[base + d_re]);
    const float k_im = static_cast<float>(k[base + d_im]);

    float oq_re, oq_im, ok_re, ok_im;

    if (in_seq) {
        oq_re = q_re * freq_re - q_im * freq_im;
        oq_im = q_re * freq_im + q_im * freq_re;
        ok_re = k_re * freq_re - k_im * freq_im;
        ok_im = k_re * freq_im + k_im * freq_re;
    } else {
        oq_re = q_re;
        oq_im = q_im;
        ok_re = k_re;
        ok_im = k_im;
    }

    out_q[base + d_re] = static_cast<scalar_t>(oq_re);
    out_q[base + d_im] = static_cast<scalar_t>(oq_im);
    out_k[base + d_re] = static_cast<scalar_t>(ok_re);
    out_k[base + d_im] = static_cast<scalar_t>(ok_im);
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
    int start_frame
) {
    constexpr int TILE_T = 4;
    dim3 block(64, TILE_T);
    dim3 grid((L_FIXED + TILE_T - 1) / TILE_T, N_FIXED);

    auto stream = at::cuda::getDefaultCUDAStream();

    causal_rope_apply_kernel<scalar_t, TILE_T><<<grid, block, 0, stream>>>(
        q.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        grid_sizes.data_ptr<int32_t>(),
        freqs0_re.data_ptr<float>(),
        freqs0_im.data_ptr<float>(),
        freqs1_re.data_ptr<float>(),
        freqs1_im.data_ptr<float>(),
        freqs2_re.data_ptr<float>(),
        freqs2_im.data_ptr<float>(),
        out_q.data_ptr<scalar_t>(),
        out_k.data_ptr<scalar_t>(),
        start_frame
    );
}

} // namespace

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
) {
    switch (q.scalar_type()) {
        case torch::kFloat32:
            launch_impl<float>(
                q, k, grid_sizes,
                freqs0_re, freqs0_im,
                freqs1_re, freqs1_im,
                freqs2_re, freqs2_im,
                out_q, out_k, start_frame
            );
            break;
        case torch::kFloat16:
            launch_impl<c10::Half>(
                q, k, grid_sizes,
                freqs0_re, freqs0_im,
                freqs1_re, freqs1_im,
                freqs2_re, freqs2_im,
                out_q, out_k, start_frame
            );
            break;
        case torch::kBFloat16:
            launch_impl<c10::BFloat16>(
                q, k, grid_sizes,
                freqs0_re, freqs0_im,
                freqs1_re, freqs1_im,
                freqs2_re, freqs2_im,
                out_q, out_k, start_frame
            );
            break;
        default:
            TORCH_CHECK(false, "Only float32 / float16 / bfloat16 are supported");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
