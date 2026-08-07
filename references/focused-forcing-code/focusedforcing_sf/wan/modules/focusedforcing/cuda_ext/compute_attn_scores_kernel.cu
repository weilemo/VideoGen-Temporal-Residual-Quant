#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cfloat>
#include <type_traits>

#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

constexpr int B_ = 1;
constexpr int H_ = 12;
constexpr int D_ = 128;
constexpr int QF_ = 3;
constexpr int KF_ = 18;
constexpr int P_ = 30;
constexpr int BLOCK_ = 52;
constexpr int QP_ = QF_ * P_;   // 90
constexpr int KP_ = KF_ * P_;   // 540
constexpr int T_ = 1560;
constexpr float kSoftmaxScale = 0.08838834764831845f;  // 1/sqrt(128)

template <typename T>
__device__ __forceinline__ float to_float(T x) {
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float to_float<c10::Half>(c10::Half x) {
    return __half2float(reinterpret_cast<__half&>(x));
}

template <>
__device__ __forceinline__ float to_float<c10::BFloat16>(c10::BFloat16 x) {
#if __CUDA_ARCH__ >= 800
    return __bfloat162float(reinterpret_cast<__nv_bfloat16&>(x));
#else
    return static_cast<float>(x);
#endif
}

template <typename scalar_t>
__global__ void fused_pool_qk_kernel(
    const scalar_t* __restrict__ query,   // [4680,12,128]
    const scalar_t* __restrict__ key,     // [32760,12,128]
    scalar_t* __restrict__ pooled_q,      // [12,90,128]
    scalar_t* __restrict__ pooled_k       // [12,540,128]
) {
    const int d = threadIdx.x;  // 0..127
    const int task = blockIdx.x;
    if (d >= D_) return;

    const int num_q_tasks = H_ * QP_;     // 1080
    const int num_k_tasks = H_ * KP_;     // 6480

    if (task < num_q_tasks) {
        const int h  = task / QP_;
        const int qp = task % QP_;
        const int qf = qp / P_;
        const int p  = qp % P_;

        const int base_l = qf * T_ + p * BLOCK_;

        float sum = 0.f;
        float mx = -FLT_MAX;

#pragma unroll
        for (int i = 0; i < BLOCK_; ++i) {
            const int l = base_l + i;
            const int idx = ((l * H_ + h) * D_) + d;
            const float v = to_float(query[idx]);
            sum += v;
            mx = fmaxf(mx, v);
        }

        const float mean = sum * (1.f / 52.f);
        const float outv = 0.5f * (mean + mx);   // alpha = 0.5
        pooled_q[(h * QP_ + qp) * D_ + d] = static_cast<scalar_t>(outv);
    } else {
        const int ktask = task - num_q_tasks;
        if (ktask >= num_k_tasks) return;

        const int h  = ktask / KP_;
        const int kp = ktask % KP_;
        const int kf = kp / P_;
        const int p  = kp % P_;

        const int base_l = kf * T_ + p * BLOCK_;

        float sum = 0.f;
        float mx = -FLT_MAX;

#pragma unroll
        for (int i = 0; i < BLOCK_; ++i) {
            const int l = base_l + i;
            const int idx = ((l * H_ + h) * D_) + d;
            const float v = to_float(key[idx]);
            sum += v;
            mx = fmaxf(mx, v);
        }

        const float mean = sum * (1.f / 52.f);
        const float outv = 0.5f * (mean + mx);
        pooled_k[(h * KP_ + kp) * D_ + d] = static_cast<scalar_t>(outv);
    }
}

__global__ void reduce_norm_kernel(
    const float* __restrict__ gemm_out,   // [12,90,540], row-major
    float* __restrict__ out               // [1,3,12,18]
) {
    const int qf = blockIdx.x;  // 0..2
    const int h  = blockIdx.y;  // 0..11
    const int kf = threadIdx.x; // 0..17

    __shared__ float vals[KF_];
    __shared__ float mean_s;
    __shared__ float inv_std_s;

    if (kf < KF_) {
        float acc = 0.f;

#pragma unroll
        for (int p1 = 0; p1 < P_; ++p1) {
            const int qpi = qf * P_ + p1;
#pragma unroll
            for (int p2 = 0; p2 < P_; ++p2) {
                const int kpi = kf * P_ + p2;
                const int idx = (h * QP_ + qpi) * KP_ + kpi;  // row-major [12,90,540]
                acc += gemm_out[idx];
            }
        }

        vals[kf] = acc * (kSoftmaxScale / 900.f);
    }

    __syncthreads();

    if (kf == 0) {
        float mean = 0.f;
#pragma unroll
        for (int i = 0; i < KF_; ++i) mean += vals[i];
        mean *= (1.f / 18.f);

        float var = 0.f;
#pragma unroll
        for (int i = 0; i < KF_; ++i) {
            float t = vals[i] - mean;
            var += t * t;
        }
        var *= (1.f / 18.f);

        mean_s = mean;
        inv_std_s = rsqrtf(var + 1e-6f);
    }

    __syncthreads();

    if (kf < KF_) {
        const float z = (vals[kf] - mean_s) * inv_std_s;
        const int out_idx = ((qf * H_) + h) * KF_ + kf;  // [1,3,12,18]
        out[out_idx] = z;
    }
}

template <typename scalar_t>
void run_geem(
    torch::Tensor pooled_q,   // [12,90,128], row-major
    torch::Tensor pooled_k,   // [12,540,128], row-major
    torch::Tensor gemm_out,   // [12,90,540], row-major memory
    cudaStream_t stream
) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, stream);

    cudaDataType_t in_type;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

    if constexpr (std::is_same_v<scalar_t, float>) {
        in_type = CUDA_R_32F;
        algo = CUBLAS_GEMM_DEFAULT;
    } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
        in_type = CUDA_R_16F;
        algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    } else {
        in_type = CUDA_R_16BF;
        algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }

    // Want C_row = Q_row @ K_row^T  => shape [90,540]
    // Compute C_col = C_row^T = K_row @ Q_row^T => shape [540,90]
    // K_row memory == K_col(128,540), use OP_T
    // Q_row memory == Q_col(128,90),  use OP_N
    const int m = 540;
    const int n = 90;
    const int k = 128;

    const long long strideA = 128LL * 540; // K
    const long long strideB = 128LL * 90;  // Q
    const long long strideC = 540LL * 90;  // C_col, same memory as row-major [90,540]

    const float alpha = 1.f;
    const float beta = 0.f;

    cublasStatus_t stat = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T,   // K: col(128,540) -> (540,128)
        CUBLAS_OP_N,   // Q: col(128,90)  -> (128,90)
        m, n, k,
        &alpha,
        pooled_k.data_ptr<scalar_t>(), in_type, 128, strideA,
        pooled_q.data_ptr<scalar_t>(), in_type, 128, strideB,
        &beta,
        gemm_out.data_ptr<float>(), CUDA_R_32F, 540, strideC,
        12,
        compute_type,
        algo
    );

    TORCH_CHECK(stat == CUBLAS_STATUS_SUCCESS, "cublasGemmStridedBatchedEx failed");
}

} // namespace

void compute_attn_scores_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor pooled_q,
    torch::Tensor pooled_k,
    torch::Tensor gemm_out,
    torch::Tensor out
) {
    auto stream = at::cuda::getCurrentCUDAStream();

    constexpr int total_tasks = H_ * QP_ + H_ * KP_; // 1080 + 6480 = 7560
    dim3 grid(total_tasks);
    dim3 block(D_);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        query.scalar_type(),
        "fused_pool_qk_kernel",
        [&] {
            fused_pool_qk_kernel<scalar_t><<<grid, block, 0, stream>>>(
                query.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(),
                pooled_q.data_ptr<scalar_t>(),
                pooled_k.data_ptr<scalar_t>()
            );
        }
    );

    switch (query.scalar_type()) {
        case torch::kFloat:
            run_geem<float>(pooled_q, pooled_k, gemm_out, stream);
            break;
        case torch::kHalf:
            run_geem<c10::Half>(pooled_q, pooled_k, gemm_out, stream);
            break;
        case torch::kBFloat16:
            run_geem<c10::BFloat16>(pooled_q, pooled_k, gemm_out, stream);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype");
    }

    dim3 grid2(QF_, H_);
    dim3 block2(32);
    reduce_norm_kernel<<<grid2, block2, 0, stream>>>(
        gemm_out.data_ptr<float>(),
        out.data_ptr<float>()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
