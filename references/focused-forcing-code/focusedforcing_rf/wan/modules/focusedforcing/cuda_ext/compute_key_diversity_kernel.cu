#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

constexpr int T_FIXED = 1560;
constexpr int H_FIXED = 12;
constexpr int D_FIXED = 128;
constexpr int F_MAX   = 21;
constexpr float EPS   = 1e-6f;
constexpr int T_TILE  = 32;

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

// fixed: broadcast the block sum to all 128 threads
__device__ __forceinline__ float block_sum_128(float v) {
    __shared__ float shared[4];
    __shared__ float block_out;

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    v = warp_sum(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    if (wid == 0) {
        float out = (lane < 4) ? shared[lane] : 0.0f;
        out = warp_sum(out);
        if (lane == 0) block_out = out;
    }
    __syncthreads();

    return block_out;
}

// ------------------------------------------------------------
// layout helpers
// k layout:       [B, L, H, D]
// mean_pos:       [B, T, H, D]
// partial_sums:   [B, H, F, NTILES]
// out_div:        [B, H, F]
// ------------------------------------------------------------
__device__ __forceinline__ long k_offset_4d(
    int b, int l, int h, int d,
    int L
) {
    return (((long)b * L + l) * H_FIXED + h) * D_FIXED + d;
}

__device__ __forceinline__ long mp_offset_4d(
    int b, int t, int h, int d
) {
    return (((long)b * T_FIXED + t) * H_FIXED + h) * D_FIXED + d;
}

__device__ __forceinline__ long partial_offset_4d(
    int b, int h, int f, int tile,
    int F, int NTILES
) {
    return ((((long)b * H_FIXED + h) * F + f) * NTILES + tile);
}

__device__ __forceinline__ long out_offset_3d(
    int b, int h, int f,
    int F
) {
    return ((long)b * H_FIXED + h) * F + f;
}

// ------------------------------------------------------------
// stage 1: build mean_pos over frames, then normalize over D
// one CTA = (b, t, h), 128 threads over d
// ------------------------------------------------------------
template <typename scalar_t>
__global__ void build_mean_pos_kernel(
    const scalar_t* __restrict__ k,        // [B, L, H, D]
    scalar_t* __restrict__ mean_pos,       // [B, T, H, D]
    int L,
    int F
) {
    int t = blockIdx.x;    // 0..T-1
    int h = blockIdx.y;    // 0..H-1
    int b = blockIdx.z;    // 0..B-1
    int d = threadIdx.x;   // 0..127

    float sum = 0.0f;
#pragma unroll
    for (int f = 0; f < F; ++f) {
        int l = f * T_FIXED + t;
        sum += to_float_device<scalar_t>(k[k_offset_4d(b, l, h, d, L)]);
    }

    float mean_v = sum / float(F);
    float sq = mean_v * mean_v;
    float norm2 = block_sum_128(sq);

    __shared__ float inv_norm;
    if (threadIdx.x == 0) {
        inv_norm = rsqrtf(norm2 + EPS);
    }
    __syncthreads();

    mean_pos[mp_offset_4d(b, t, h, d)] =
        from_float_device<scalar_t>(mean_v * inv_norm);
}

// ------------------------------------------------------------
// stage 2: partial similarity
//
// grid:
//   x = h
//   y = b
//   z = f * NTILES + tile_id
//
// block:
//   32 threads, one warp
//
// each block computes one (b, h, f, tile_id)
// and writes a partial sum over one T tile
// ------------------------------------------------------------
__global__ void compute_div_partial_kernel_fp32_vec4(
    const float* __restrict__ k,
    const float* __restrict__ mean_pos,
    float* __restrict__ partial_sums,
    int L,
    int F,
    int NTILES
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    int z = blockIdx.z;

    int f = z / NTILES;
    int tile_id = z % NTILES;

    int lane = threadIdx.x;   // 0..31

    int t0 = tile_id * T_TILE;
    int t  = t0 + lane;

    float v = 0.0f;

    if (f < F && t < T_FIXED) {
        long k_base = k_offset_4d(b, f * T_FIXED + t, h, 0, L);
        long m_base = mp_offset_4d(b, t, h, 0);

        const float4* k4 = reinterpret_cast<const float4*>(k + k_base);
        const float4* m4 = reinterpret_cast<const float4*>(mean_pos + m_base);

        float dot = 0.0f;
        float norm2 = 0.0f;

#pragma unroll
        for (int i = 0; i < D_FIXED / 4; ++i) {
            float4 a = k4[i];
            float4 b4 = m4[i];
            dot   += a.x * b4.x + a.y * b4.y + a.z * b4.z + a.w * b4.w;
            norm2 += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
        }

        v = dot * rsqrtf(norm2 + EPS);
    }

    float tile_sum = warp_sum(v);
    if (lane == 0) {
        partial_sums[partial_offset_4d(b, h, f, tile_id, F, NTILES)] = tile_sum;
    }
}

__global__ void compute_div_partial_kernel_fp16_half2(
    const half* __restrict__ k,
    const half* __restrict__ mean_pos,
    float* __restrict__ partial_sums,
    int L,
    int F,
    int NTILES
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    int z = blockIdx.z;

    int f = z / NTILES;
    int tile_id = z % NTILES;

    int lane = threadIdx.x;   // 0..31

    int t0 = tile_id * T_TILE;
    int t  = t0 + lane;

    float v = 0.0f;

    if (f < F && t < T_FIXED) {
        long k_base = k_offset_4d(b, f * T_FIXED + t, h, 0, L);
        long m_base = mp_offset_4d(b, t, h, 0);

        const half2* k2 = reinterpret_cast<const half2*>(k + k_base);
        const half2* m2 = reinterpret_cast<const half2*>(mean_pos + m_base);

        float dot = 0.0f;
        float norm2 = 0.0f;

#pragma unroll
        for (int i = 0; i < D_FIXED / 2; ++i) {
            float2 af = __half22float2(k2[i]);
            float2 bf = __half22float2(m2[i]);
            dot   += af.x * bf.x + af.y * bf.y;
            norm2 += af.x * af.x + af.y * af.y;
        }

        v = dot * rsqrtf(norm2 + EPS);
    }

    float tile_sum = warp_sum(v);
    if (lane == 0) {
        partial_sums[partial_offset_4d(b, h, f, tile_id, F, NTILES)] = tile_sum;
    }
}

__global__ void compute_div_partial_kernel_bf16_bf162(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ mean_pos,
    float* __restrict__ partial_sums,
    int L,
    int F,
    int NTILES
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    int z = blockIdx.z;

    int f = z / NTILES;
    int tile_id = z % NTILES;

    int lane = threadIdx.x;   // 0..31

    int t0 = tile_id * T_TILE;
    int t  = t0 + lane;

    float v = 0.0f;

    if (f < F && t < T_FIXED) {
        long k_base = k_offset_4d(b, f * T_FIXED + t, h, 0, L);
        long m_base = mp_offset_4d(b, t, h, 0);

        const __nv_bfloat162* k2 =
            reinterpret_cast<const __nv_bfloat162*>(k + k_base);
        const __nv_bfloat162* m2 =
            reinterpret_cast<const __nv_bfloat162*>(mean_pos + m_base);

        float dot = 0.0f;
        float norm2 = 0.0f;

#pragma unroll
        for (int i = 0; i < D_FIXED / 2; ++i) {
            float2 af = __bfloat1622float2(k2[i]);
            float2 bf = __bfloat1622float2(m2[i]);
            dot   += af.x * bf.x + af.y * bf.y;
            norm2 += af.x * af.x + af.y * af.y;
        }

        v = dot * rsqrtf(norm2 + EPS);
    }

    float tile_sum = warp_sum(v);
    if (lane == 0) {
        partial_sums[partial_offset_4d(b, h, f, tile_id, F, NTILES)] = tile_sum;
    }
}

// ------------------------------------------------------------
// stage 3: reduce partial sums over tiles, then z-score over F
//
// grid:  (h, b)
// block: 32 threads
//
// lane < F handles one frame
// ------------------------------------------------------------
__global__ void reduce_div_zscore_kernel(
    const float* __restrict__ partial_sums,   // [B, H, F, NTILES]
    float* __restrict__ out_div,              // [B, H, F]
    int F,
    int NTILES
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    int lane = threadIdx.x;

    __shared__ float div_vals[F_MAX];
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    if (lane < F) {
        float s = 0.0f;
#pragma unroll
        for (int tile = 0; tile < 49; ++tile) {
            if (tile < NTILES) {
                s += partial_sums[partial_offset_4d(b, h, lane, tile, F, NTILES)];
            }
        }

        float sim = s / float(T_FIXED);
        div_vals[lane] = -sim;
    }
    __syncthreads();

    float v = (lane < F) ? div_vals[lane] : 0.0f;
    float sum = warp_sum(v);

    if (lane == 0) {
        s_mean = sum / float(F);
    }
    __syncthreads();

    float diff = (lane < F) ? (div_vals[lane] - s_mean) : 0.0f;
    float sq = diff * diff;
    float sq_sum = warp_sum(sq);

    if (lane == 0) {
        float var = sq_sum / float(F);
        float std = sqrtf(var);
        s_inv_std = 1.0f / (std + 1e-6f);
    }
    __syncthreads();

    if (lane < F) {
        out_div[out_offset_3d(b, h, lane, F)] =
            (div_vals[lane] - s_mean) * s_inv_std;
    }
}

// ------------------------------------------------------------
// host entry
// k:        [B, L, H, D]
// mean_pos: [B, T, H, D]
// out_div:  [B, H, F]
// ------------------------------------------------------------
void compute_key_diversity_cuda(
    torch::Tensor k,
    torch::Tensor mean_pos,
    torch::Tensor out_div
) {
    TORCH_CHECK(k.is_cuda(), "k must be CUDA tensor");
    TORCH_CHECK(mean_pos.is_cuda(), "mean_pos must be CUDA tensor");
    TORCH_CHECK(out_div.is_cuda(), "out_div must be CUDA tensor");

    TORCH_CHECK(k.dim() == 4, "k must be [B, L, H, D]");
    TORCH_CHECK(k.size(2) == H_FIXED, "k.size(2) must be 12");
    TORCH_CHECK(k.size(3) == D_FIXED, "k.size(3) must be 128");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(mean_pos.is_contiguous(), "mean_pos must be contiguous");
    TORCH_CHECK(out_div.is_contiguous(), "out_div must be contiguous");

    const int B = (int)k.size(0);
    const int L = (int)k.size(1);

    TORCH_CHECK(L % T_FIXED == 0,
                "L must be divisible by T_FIXED=", T_FIXED);

    const int F = L / T_FIXED;

    TORCH_CHECK(F >= 1, "F must be >= 1");
    TORCH_CHECK(F <= F_MAX, "F too large, expected F <= ", F_MAX, ", got ", F);

    TORCH_CHECK(mean_pos.size(0) == B &&
                mean_pos.size(1) == T_FIXED &&
                mean_pos.size(2) == H_FIXED &&
                mean_pos.size(3) == D_FIXED,
                "mean_pos must have shape [B, 1560, 12, 128]");

    TORCH_CHECK(out_div.size(0) == B &&
                out_div.size(1) == H_FIXED &&
                out_div.size(2) == F,
                "out_div must have shape [B, 12, F]");

    TORCH_CHECK(k.scalar_type() == mean_pos.scalar_type(),
                "k and mean_pos dtype must match");
    TORCH_CHECK(out_div.scalar_type() == at::ScalarType::Float,
                "out_div must be float32");

    const int NTILES = (T_FIXED + T_TILE - 1) / T_TILE;  // 49

    auto partial_sums = torch::empty(
        {B, H_FIXED, F, NTILES},
        torch::TensorOptions().device(k.device()).dtype(torch::kFloat)
    );

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 block_mean(D_FIXED);              // 128
    dim3 grid_mean(T_FIXED, H_FIXED, B);   // (t, h, b)

    dim3 block_partial(32);                // one warp
    dim3 grid_partial(H_FIXED, B, F * NTILES);

    dim3 block_reduce(32);
    dim3 grid_reduce(H_FIXED, B);

    switch (k.scalar_type()) {
        case at::ScalarType::Float: {
            build_mean_pos_kernel<float><<<grid_mean, block_mean, 0, stream>>>(
                k.data_ptr<float>(),
                mean_pos.data_ptr<float>(),
                L, F
            );

            compute_div_partial_kernel_fp32_vec4<<<grid_partial, block_partial, 0, stream>>>(
                k.data_ptr<float>(),
                mean_pos.data_ptr<float>(),
                partial_sums.data_ptr<float>(),
                L, F, NTILES
            );
            break;
        }

        case at::ScalarType::Half: {
            build_mean_pos_kernel<half><<<grid_mean, block_mean, 0, stream>>>(
                reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
                reinterpret_cast<half*>(mean_pos.data_ptr<at::Half>()),
                L, F
            );

            compute_div_partial_kernel_fp16_half2<<<grid_partial, block_partial, 0, stream>>>(
                reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(mean_pos.data_ptr<at::Half>()),
                partial_sums.data_ptr<float>(),
                L, F, NTILES
            );
            break;
        }

        case at::ScalarType::BFloat16: {
            build_mean_pos_kernel<__nv_bfloat16><<<grid_mean, block_mean, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(mean_pos.data_ptr<at::BFloat16>()),
                L, F
            );

            compute_div_partial_kernel_bf16_bf162<<<grid_partial, block_partial, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(mean_pos.data_ptr<at::BFloat16>()),
                partial_sums.data_ptr<float>(),
                L, F, NTILES
            );
            break;
        }

        default:
            TORCH_CHECK(false, "Unsupported dtype for k");
    }

    reduce_div_zscore_kernel<<<grid_reduce, block_reduce, 0, stream>>>(
        partial_sums.data_ptr<float>(),
        out_div.data_ptr<float>(),
        F, NTILES
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
