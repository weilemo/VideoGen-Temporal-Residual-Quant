#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

constexpr int NUM_BLOCKS = 30;
constexpr int B_FIXED    = 1;
constexpr int F_FIXED    = 18;
constexpr int T_FIXED    = 1560;
constexpr int H_FIXED    = 12;
constexpr int D_FIXED    = 128;
constexpr int L_FIXED    = F_FIXED * T_FIXED;
constexpr float EPS      = 1e-6f;
constexpr int T_TILE     = 32;

// ------------------------------------------------------------
// helpers
// ------------------------------------------------------------
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

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_sum_128(float v) {
    __shared__ float shared[4];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    v = warp_sum(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    float out = 0.0f;
    if (wid == 0) {
        out = (lane < 4) ? shared[lane] : 0.0f;
        out = warp_sum(out);
    }
    return out;
}

// k layout: [30,1,28080,12,128]
__device__ __forceinline__ long k_offset_5d(int blk, int l, int h, int d) {
    return ((((long)blk * B_FIXED + 0) * L_FIXED + l) * H_FIXED + h) * D_FIXED + d;
}

// mean_pos layout: [30,1560,12,128]
__device__ __forceinline__ long mp_offset(int blk, int t, int h, int d) {
    return (((long)blk * T_FIXED + t) * H_FIXED + h) * D_FIXED + d;
}

// out_div logical layout: [30,1,12,18]
__device__ __forceinline__ long out_offset(int blk, int h, int f) {
    return ((long)blk * H_FIXED + h) * F_FIXED + f;
}

// ------------------------------------------------------------
// build mean_pos over frames, then normalize over D
// one CTA = (blk, t, h), 128 threads over d
// logic identical to original
// ------------------------------------------------------------
template <typename scalar_t>
__global__ void build_mean_pos_all_kernel(
    const scalar_t* __restrict__ k,
    scalar_t* __restrict__ mean_pos
) {
    int t   = blockIdx.x;
    int h   = blockIdx.y;
    int blk = blockIdx.z;
    int d   = threadIdx.x;

    float sum = 0.0f;
#pragma unroll
    for (int f = 0; f < F_FIXED; ++f) {
        int l = f * T_FIXED + t;
        sum += to_float_device<scalar_t>(k[k_offset_5d(blk, l, h, d)]);
    }

    float mean_v = sum / float(F_FIXED);
    float sq = mean_v * mean_v;
    float norm2 = block_sum_128(sq);

    __shared__ float inv_norm;
    if (threadIdx.x == 0) {
        inv_norm = rsqrtf(norm2 + EPS);
    }
    __syncthreads();

    mean_pos[mp_offset(blk, t, h, d)] =
        from_float_device<scalar_t>(mean_v * inv_norm);
}

// ------------------------------------------------------------
// optimized div + zscore kernels
// logic identical to original, but:
// - block = (32, 18), one warp handles one frame f
// - mean_pos is tiled into shared memory by t
// ------------------------------------------------------------

__global__ void compute_div_zscore_all_kernel_fp32_vec4_tiled(
    const float* __restrict__ k,
    const float* __restrict__ mean_pos,
    float* __restrict__ out_div
) {
    int h       = blockIdx.x;
    int blk     = blockIdx.y;
    int lane    = threadIdx.x;  // 0..31
    int warp_id = threadIdx.y;  // 0..17
    int f       = warp_id;

    __shared__ float div_vals[F_FIXED];
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    __shared__ float4 smem_mean[T_TILE][D_FIXED / 4];

    float warp_frame_sum = 0.0f;

    for (int t0 = 0; t0 < T_FIXED; t0 += T_TILE) {
        int valid_t = T_FIXED - t0;
        if (valid_t > T_TILE) valid_t = T_TILE;

        int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
        int nthreads   = blockDim.x * blockDim.y;
        int nvec       = valid_t * (D_FIXED / 4);

        for (int idx = linear_tid; idx < nvec; idx += nthreads) {
            int tt = idx / (D_FIXED / 4);
            int vi = idx % (D_FIXED / 4);
            long m_base = mp_offset(blk, t0 + tt, h, 0);
            const float4* m4 = reinterpret_cast<const float4*>(mean_pos + m_base);
            smem_mean[tt][vi] = m4[vi];
        }
        __syncthreads();

        if (lane < valid_t) {
            int t = t0 + lane;
            long k_base = k_offset_5d(blk, f * T_FIXED + t, h, 0);

            const float4* k4 = reinterpret_cast<const float4*>(k + k_base);
            float dot = 0.0f;
            float norm2 = 0.0f;

#pragma unroll
            for (int i = 0; i < D_FIXED / 4; ++i) {
                float4 a = k4[i];
                float4 b = smem_mean[lane][i];
                dot   += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                norm2 += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
            }

            warp_frame_sum += dot * rsqrtf(norm2 + EPS);
        }
        __syncthreads();
    }

    warp_frame_sum = warp_sum(warp_frame_sum);
    if (lane == 0) {
        float sim = warp_frame_sum / float(T_FIXED);
        div_vals[f] = -sim;
    }

    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < F_FIXED) ? div_vals[lane] : 0.0f;
        float sum = warp_sum(v);

        if (lane == 0) {
            s_mean = sum / float(F_FIXED);
        }
        __syncthreads();

        float diff = (lane < F_FIXED) ? (div_vals[lane] - s_mean) : 0.0f;
        float sq = diff * diff;
        float sq_sum = warp_sum(sq);

        if (lane == 0) {
            float var = sq_sum / float(F_FIXED);  // correction=0
            float std = sqrtf(var);
            s_inv_std = 1.0f / (std + 1e-6f);
        }
        __syncthreads();

        if (lane < F_FIXED) {
            out_div[out_offset(blk, h, lane)] = (div_vals[lane] - s_mean) * s_inv_std;
        }
    }
}

__global__ void compute_div_zscore_all_kernel_fp16_half2_tiled(
    const half* __restrict__ k,
    const half* __restrict__ mean_pos,
    float* __restrict__ out_div
) {
    int h       = blockIdx.x;
    int blk     = blockIdx.y;
    int lane    = threadIdx.x;  // 0..31
    int warp_id = threadIdx.y;  // 0..17
    int f       = warp_id;

    __shared__ float div_vals[F_FIXED];
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    __shared__ half2 smem_mean[T_TILE][D_FIXED / 2];

    float warp_frame_sum = 0.0f;

    for (int t0 = 0; t0 < T_FIXED; t0 += T_TILE) {
        int valid_t = T_FIXED - t0;
        if (valid_t > T_TILE) valid_t = T_TILE;

        int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
        int nthreads   = blockDim.x * blockDim.y;
        int nvec       = valid_t * (D_FIXED / 2);

        for (int idx = linear_tid; idx < nvec; idx += nthreads) {
            int tt = idx / (D_FIXED / 2);
            int vi = idx % (D_FIXED / 2);
            long m_base = mp_offset(blk, t0 + tt, h, 0);
            const half2* m2 = reinterpret_cast<const half2*>(mean_pos + m_base);
            smem_mean[tt][vi] = m2[vi];
        }
        __syncthreads();

        if (lane < valid_t) {
            int t = t0 + lane;
            long k_base = k_offset_5d(blk, f * T_FIXED + t, h, 0);

            const half2* k2 = reinterpret_cast<const half2*>(k + k_base);
            float dot = 0.0f;
            float norm2 = 0.0f;

#pragma unroll
            for (int i = 0; i < D_FIXED / 2; ++i) {
                float2 af = __half22float2(k2[i]);
                float2 bf = __half22float2(smem_mean[lane][i]);
                dot   += af.x * bf.x + af.y * bf.y;
                norm2 += af.x * af.x + af.y * af.y;
            }

            warp_frame_sum += dot * rsqrtf(norm2 + EPS);
        }
        __syncthreads();
    }

    warp_frame_sum = warp_sum(warp_frame_sum);
    if (lane == 0) {
        float sim = warp_frame_sum / float(T_FIXED);
        div_vals[f] = -sim;
    }

    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < F_FIXED) ? div_vals[lane] : 0.0f;
        float sum = warp_sum(v);

        if (lane == 0) {
            s_mean = sum / float(F_FIXED);
        }
        __syncthreads();

        float diff = (lane < F_FIXED) ? (div_vals[lane] - s_mean) : 0.0f;
        float sq = diff * diff;
        float sq_sum = warp_sum(sq);

        if (lane == 0) {
            float var = sq_sum / float(F_FIXED);
            float std = sqrtf(var);
            s_inv_std = 1.0f / (std + 1e-6f);
        }
        __syncthreads();

        if (lane < F_FIXED) {
            out_div[out_offset(blk, h, lane)] = (div_vals[lane] - s_mean) * s_inv_std;
        }
    }
}

__global__ void compute_div_zscore_all_kernel_bf16_bf162_tiled(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ mean_pos,
    float* __restrict__ out_div
) {
    int h       = blockIdx.x;
    int blk     = blockIdx.y;
    int lane    = threadIdx.x;  // 0..31
    int warp_id = threadIdx.y;  // 0..17
    int f       = warp_id;

    __shared__ float div_vals[F_FIXED];
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    __shared__ __nv_bfloat162 smem_mean[T_TILE][D_FIXED / 2];

    float warp_frame_sum = 0.0f;

    for (int t0 = 0; t0 < T_FIXED; t0 += T_TILE) {
        int valid_t = T_FIXED - t0;
        if (valid_t > T_TILE) valid_t = T_TILE;

        int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
        int nthreads   = blockDim.x * blockDim.y;
        int nvec       = valid_t * (D_FIXED / 2);

        for (int idx = linear_tid; idx < nvec; idx += nthreads) {
            int tt = idx / (D_FIXED / 2);
            int vi = idx % (D_FIXED / 2);
            long m_base = mp_offset(blk, t0 + tt, h, 0);
            const __nv_bfloat162* m2 = reinterpret_cast<const __nv_bfloat162*>(mean_pos + m_base);
            smem_mean[tt][vi] = m2[vi];
        }
        __syncthreads();

        if (lane < valid_t) {
            int t = t0 + lane;
            long k_base = k_offset_5d(blk, f * T_FIXED + t, h, 0);

            const __nv_bfloat162* k2 = reinterpret_cast<const __nv_bfloat162*>(k + k_base);
            float dot = 0.0f;
            float norm2 = 0.0f;

#pragma unroll
            for (int i = 0; i < D_FIXED / 2; ++i) {
                float2 af = __bfloat1622float2(k2[i]);
                float2 bf = __bfloat1622float2(smem_mean[lane][i]);
                dot   += af.x * bf.x + af.y * bf.y;
                norm2 += af.x * af.x + af.y * af.y;
            }

            warp_frame_sum += dot * rsqrtf(norm2 + EPS);
        }
        __syncthreads();
    }

    warp_frame_sum = warp_sum(warp_frame_sum);
    if (lane == 0) {
        float sim = warp_frame_sum / float(T_FIXED);
        div_vals[f] = -sim;
    }

    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < F_FIXED) ? div_vals[lane] : 0.0f;
        float sum = warp_sum(v);

        if (lane == 0) {
            s_mean = sum / float(F_FIXED);
        }
        __syncthreads();

        float diff = (lane < F_FIXED) ? (div_vals[lane] - s_mean) : 0.0f;
        float sq = diff * diff;
        float sq_sum = warp_sum(sq);

        if (lane == 0) {
            float var = sq_sum / float(F_FIXED);
            float std = sqrtf(var);
            s_inv_std = 1.0f / (std + 1e-6f);
        }
        __syncthreads();

        if (lane < F_FIXED) {
            out_div[out_offset(blk, h, lane)] = (div_vals[lane] - s_mean) * s_inv_std;
        }
    }
}

// ------------------------------------------------------------
// host entry
// ------------------------------------------------------------
void compute_key_diversity_cuda(
    torch::Tensor k,
    torch::Tensor mean_pos,
    torch::Tensor out_div
) {
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 block_mean(D_FIXED);                // 128
    dim3 grid_mean(T_FIXED, H_FIXED, NUM_BLOCKS);

    dim3 block_div(32, F_FIXED);             // 32 x 18
    dim3 grid_div(H_FIXED, NUM_BLOCKS);

    switch (k.scalar_type()) {
        case at::ScalarType::Float: {
            build_mean_pos_all_kernel<float><<<grid_mean, block_mean, 0, stream>>>(
                k.data_ptr<float>(),
                mean_pos.data_ptr<float>()
            );

            compute_div_zscore_all_kernel_fp32_vec4_tiled<<<grid_div, block_div, 0, stream>>>(
                k.data_ptr<float>(),
                mean_pos.data_ptr<float>(),
                out_div.data_ptr<float>()
            );
            break;
        }
        case at::ScalarType::Half: {
            build_mean_pos_all_kernel<half><<<grid_mean, block_mean, 0, stream>>>(
                reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
                reinterpret_cast<half*>(mean_pos.data_ptr<at::Half>())
            );

            compute_div_zscore_all_kernel_fp16_half2_tiled<<<grid_div, block_div, 0, stream>>>(
                reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(mean_pos.data_ptr<at::Half>()),
                out_div.data_ptr<float>()
            );
            break;
        }
        case at::ScalarType::BFloat16: {
            build_mean_pos_all_kernel<__nv_bfloat16><<<grid_mean, block_mean, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(mean_pos.data_ptr<at::BFloat16>())
            );

            compute_div_zscore_all_kernel_bf16_bf162_tiled<<<grid_div, block_div, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(mean_pos.data_ptr<at::BFloat16>()),
                out_div.data_ptr<float>()
            );
            break;
        }
        default:
            TORCH_CHECK(false, "Unsupported dtype for k");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
