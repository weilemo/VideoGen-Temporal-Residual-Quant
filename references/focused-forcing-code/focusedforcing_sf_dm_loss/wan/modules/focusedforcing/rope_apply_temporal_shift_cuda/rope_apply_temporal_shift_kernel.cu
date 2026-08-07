#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int D_FIXED = 128;
constexpr int T_C_FIXED = 22;

template <typename T>
__device__ __forceinline__ double load_as_double(const T* ptr);

template <>
__device__ __forceinline__ double load_as_double<float>(const float* ptr) {
    return static_cast<double>(*ptr);
}

template <>
__device__ __forceinline__ double load_as_double<half>(const half* ptr) {
    return static_cast<double>(__half2float(*ptr));
}

template <>
__device__ __forceinline__ double load_as_double<nv_bfloat16>(const nv_bfloat16* ptr) {
    return static_cast<double>(__bfloat162float(*ptr));
}

template <typename T>
__device__ __forceinline__ T cast_from_double(double x);

template <>
__device__ __forceinline__ float cast_from_double<float>(double x) {
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__ half cast_from_double<half>(double x) {
    return __float2half_rn(static_cast<float>(x));
}

template <>
__device__ __forceinline__ nv_bfloat16 cast_from_double<nv_bfloat16>(double x) {
    return __float2bfloat16(static_cast<float>(x));
}

template <typename scalar_t>
__global__ void rope_apply_temporal_shift_kernel(
    scalar_t* __restrict__ k,
    const double* __restrict__ mult_re,
    const double* __restrict__ mult_im,
    int64_t n_tokens
) {
    int64_t token_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= n_tokens) return;

    scalar_t* base = k + token_idx * D_FIXED;

    #pragma unroll
    for (int t = 0; t < T_C_FIXED; ++t) {
        int re_idx = 2 * t;
        int im_idx = re_idx + 1;

        double kre = load_as_double(base + re_idx);
        double kim = load_as_double(base + im_idx);

        double mre = mult_re[t];
        double mim = mult_im[t];

        double out_re = kre * mre - kim * mim;
        double out_im = kre * mim + kim * mre;

        base[re_idx] = cast_from_double<scalar_t>(out_re);
        base[im_idx] = cast_from_double<scalar_t>(out_im);
    }
}

} // namespace

void rope_apply_temporal_shift_cuda(
    torch::Tensor k_chunk,
    torch::Tensor mult_re,
    torch::Tensor mult_im
) {
    const auto B = k_chunk.size(0);
    const auto L = k_chunk.size(1);
    const auto H = k_chunk.size(2);
    const int64_t n_tokens = B * L * H;

    auto stream = at::cuda::getDefaultCUDAStream();
    constexpr int threads = 256;
    const int blocks = (int)((n_tokens + threads - 1) / threads);

    auto dtype = k_chunk.scalar_type();

    if (dtype == torch::kFloat32) {
        rope_apply_temporal_shift_kernel<float><<<blocks, threads, 0, stream>>>(
            k_chunk.data_ptr<float>(),
            mult_re.data_ptr<double>(),
            mult_im.data_ptr<double>(),
            n_tokens
        );
    } else if (dtype == torch::kFloat16) {
        rope_apply_temporal_shift_kernel<half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<half*>(k_chunk.data_ptr<at::Half>()),
            mult_re.data_ptr<double>(),
            mult_im.data_ptr<double>(),
            n_tokens
        );
    } else if (dtype == torch::kBFloat16) {
        rope_apply_temporal_shift_kernel<nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<nv_bfloat16*>(k_chunk.data_ptr<at::BFloat16>()),
            mult_re.data_ptr<double>(),
            mult_im.data_ptr<double>(),
            n_tokens
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
