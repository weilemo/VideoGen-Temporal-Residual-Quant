#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

constexpr int NUM_BLOCKS = 30;
constexpr int B_FIXED    = 1;
constexpr int H_FIXED    = 12;
constexpr int D_FIXED    = 128;
constexpr int T_C_FIXED  = 22;

template <typename T>
__device__ __forceinline__ float to_float_device(T x);

template <>
__device__ __forceinline__ float to_float_device<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ float to_float_device<half>(half x) {
    return __half2float(x);
}

template <>
__device__ __forceinline__ float to_float_device<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T from_float_device(float x);

template <>
__device__ __forceinline__ float from_float_device<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ half from_float_device<half>(float x) {
    return __float2half_rn(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float_device<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

// offset computed from actual tensor strides (in elements)
__device__ __forceinline__ long k_offset_5d_stride(
    int blk, int b, int l, int h, int d,
    long s0, long s1, long s2, long s3, long s4
) {
    return (long)blk * s0 + (long)b * s1 + (long)l * s2 + (long)h * s3 + (long)d * s4;
}

// one CTA = (token, head, block), 32 threads, first 22 lanes used
template <typename scalar_t>
__global__ void rope_apply_temporal_shift_all_kernel_stride(
    scalar_t* __restrict__ k,
    const double* __restrict__ mult_re,
    const double* __restrict__ mult_im,
    int L,
    long s0, long s1, long s2, long s3, long s4
) {
    int tok = blockIdx.x;
    int h   = blockIdx.y;
    int blk = blockIdx.z;

    if (tok >= L) return;

    int c = threadIdx.x;
    if (c < T_C_FIXED) {
        int d0 = 2 * c;
        int d1 = d0 + 1;

        long off0 = k_offset_5d_stride(blk, 0, tok, h, d0, s0, s1, s2, s3, s4);
        long off1 = k_offset_5d_stride(blk, 0, tok, h, d1, s0, s1, s2, s3, s4);

        float xr_f = to_float_device<scalar_t>(k[off0]);
        float xi_f = to_float_device<scalar_t>(k[off1]);

        double xr = static_cast<double>(xr_f);
        double xi = static_cast<double>(xi_f);
        double mr = mult_re[c];
        double mi = mult_im[c];

        double yr = xr * mr - xi * mi;
        double yi = xr * mi + xi * mr;

        k[off0] = from_float_device<scalar_t>(static_cast<float>(yr));
        k[off1] = from_float_device<scalar_t>(static_cast<float>(yi));
    }
}

void rope_apply_temporal_shift_cuda(
    torch::Tensor k_all,
    torch::Tensor mult_re,
    torch::Tensor mult_im
) {
    const int L = static_cast<int>(k_all.size(2));

    const long s0 = k_all.stride(0);
    const long s1 = k_all.stride(1);
    const long s2 = k_all.stride(2);
    const long s3 = k_all.stride(3);
    const long s4 = k_all.stride(4);

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 block(32);
    dim3 grid((unsigned int)L, H_FIXED, NUM_BLOCKS);

    switch (k_all.scalar_type()) {
        case at::ScalarType::Float: {
            rope_apply_temporal_shift_all_kernel_stride<float><<<grid, block, 0, stream>>>(
                k_all.data_ptr<float>(),
                mult_re.data_ptr<double>(),
                mult_im.data_ptr<double>(),
                L, s0, s1, s2, s3, s4
            );
            break;
        }
        case at::ScalarType::Half: {
            rope_apply_temporal_shift_all_kernel_stride<half><<<grid, block, 0, stream>>>(
                reinterpret_cast<half*>(k_all.data_ptr<at::Half>()),
                mult_re.data_ptr<double>(),
                mult_im.data_ptr<double>(),
                L, s0, s1, s2, s3, s4
            );
            break;
        }
        case at::ScalarType::BFloat16: {
            rope_apply_temporal_shift_all_kernel_stride<__nv_bfloat16><<<grid, block, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(k_all.data_ptr<at::BFloat16>()),
                mult_re.data_ptr<double>(),
                mult_im.data_ptr<double>(),
                L, s0, s1, s2, s3, s4
            );
            break;
        }
        default:
            TORCH_CHECK(false, "Unsupported dtype for k_all");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
