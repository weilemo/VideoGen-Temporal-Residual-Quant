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

constexpr int P_ = 30;
constexpr int BLOCK_ = 52;
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

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

template <typename scalar_t>
__global__ void fused_pool_qk_kernel(
    const scalar_t* __restrict__ query,   // [qf*T_, H_, D_]
    const scalar_t* __restrict__ key,     // [kf*T_, H_, D_]
    scalar_t* __restrict__ pooled_q,      // [H_, qf*P_, D_]
    scalar_t* __restrict__ pooled_k,      // [H_, kf*P_, D_]
    int qf,
    int kf,
    int qp,
    int kp
) {
    const int d = threadIdx.x;   // 0..127
    const int task = blockIdx.x;
    if (d >= D_) return;

    const int num_q_tasks = H_ * qp;
    const int num_k_tasks = H_ * kp;

    if (task < num_q_tasks) {
        const int h     = task / qp;
        const int qp_idx = task % qp;
        const int qf_idx = qp_idx / P_;
        const int p      = qp_idx % P_;

        const int base_l = qf_idx * T_ + p * BLOCK_;

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

        const float mean = sum * (1.f / BLOCK_);
        const float outv = 0.5f * (mean + mx);

        pooled_q[(h * qp + qp_idx) * D_ + d] = static_cast<scalar_t>(outv);
    } else {
        const int ktask = task - num_q_tasks;
        if (ktask >= num_k_tasks) return;

        const int h     = ktask / kp;
        const int kp_idx = ktask % kp;
        const int kf_idx = kp_idx / P_;
        const int p      = kp_idx % P_;

        const int base_l = kf_idx * T_ + p * BLOCK_;

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

        const float mean = sum * (1.f / BLOCK_);
        const float outv = 0.5f * (mean + mx);

        pooled_k[(h * kp + kp_idx) * D_ + d] = static_cast<scalar_t>(outv);
    }
}

__global__ void reduce_norm_kernel(
    const float* __restrict__ gemm_out,   // [H_, qp, kp], row-major
    float* __restrict__ out,              // [1, qf, H_, kf]
    int qf,
    int kf,
    int qp,
    int kp
) {
    const int qf_idx = blockIdx.x;   // 0..qf-1
    const int h      = blockIdx.y;   // 0..H_-1
    const int kf_idx = threadIdx.x;  // 0..kf-1

    extern __shared__ float smem[];
    float* vals = smem;          // kf floats
    float* mean_s_ptr = vals + kf;
    float* inv_std_s_ptr = vals + kf + 1;

    if (kf_idx < kf) {
        float acc = 0.f;

#pragma unroll
        for (int p1 = 0; p1 < P_; ++p1) {
            const int qpi = qf_idx * P_ + p1;
#pragma unroll
            for (int p2 = 0; p2 < P_; ++p2) {
                const int kpi = kf_idx * P_ + p2;
                const int idx = (h * qp + qpi) * kp + kpi;  // row-major [H_, qp, kp]
                acc += gemm_out[idx];
            }
        }

        vals[kf_idx] = acc * (kSoftmaxScale / float(P_ * P_));
    }

    __syncthreads();

    if (kf_idx == 0) {
        float mean = 0.f;
        for (int i = 0; i < kf; ++i) mean += vals[i];
        mean *= (1.f / float(kf));

        float var = 0.f;
        for (int i = 0; i < kf; ++i) {
            float t = vals[i] - mean;
            var += t * t;
        }
        var *= (1.f / float(kf));

        *mean_s_ptr = mean;
        *inv_std_s_ptr = rsqrtf(var + 1e-6f);
    }

    __syncthreads();

    if (kf_idx < kf) {
        const float z = (vals[kf_idx] - *mean_s_ptr) * (*inv_std_s_ptr);
        const int out_idx = ((qf_idx * H_) + h) * kf + kf_idx;  // [1, qf, H_, kf]
        out[out_idx] = z;
    }
}

template <typename scalar_t>
void run_gemm(
    torch::Tensor pooled_q,   // [H_, qp, D_], row-major
    torch::Tensor pooled_k,   // [H_, kp, D_], row-major
    torch::Tensor gemm_out,   // [H_, qp, kp], row-major memory
    cudaStream_t stream,
    int qp,
    int kp
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

    // Want C_row = Q_row @ K_row^T => [qp, kp]
    // Compute C_col = C_row^T = K_row @ Q_row^T => [kp, qp]
    //

    const int m = kp;
    const int n = qp;
    const int k = D_;

    const long long strideA = 1LL * kp * D_; // K
    const long long strideB = 1LL * qp * D_; // Q
    const long long strideC = 1LL * kp * qp; // C_col == same memory as row-major [qp, kp]

    const float alpha = 1.f;
    const float beta = 0.f;

    cublasStatus_t stat = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T,   // K: col(D, kp) -> (kp, D)
        CUBLAS_OP_N,   // Q: col(D, qp) -> (D, qp)
        m, n, k,
        &alpha,
        pooled_k.data_ptr<scalar_t>(), in_type, D_, strideA,
        pooled_q.data_ptr<scalar_t>(), in_type, D_, strideB,
        &beta,
        gemm_out.data_ptr<float>(), CUDA_R_32F, kp, strideC,
        H_,
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
    TORCH_CHECK(query.is_cuda(), "query must be CUDA tensor");
    TORCH_CHECK(key.is_cuda(), "key must be CUDA tensor");
    TORCH_CHECK(pooled_q.is_cuda(), "pooled_q must be CUDA tensor");
    TORCH_CHECK(pooled_k.is_cuda(), "pooled_k must be CUDA tensor");
    TORCH_CHECK(gemm_out.is_cuda(), "gemm_out must be CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA tensor");

    TORCH_CHECK(query.dim() == 4, "query must be [1, Lq, 12, 128]");
    TORCH_CHECK(key.dim() == 4, "key must be [1, Lk, 12, 128]");

    TORCH_CHECK(query.size(0) == 1, "query batch must be 1");
    TORCH_CHECK(key.size(0) == 1, "key batch must be 1");

    TORCH_CHECK(query.size(2) == H_ && query.size(3) == D_,
                "query shape must be [1, Lq, 12, 128]");
    TORCH_CHECK(key.size(2) == H_ && key.size(3) == D_,
                "key shape must be [1, Lk, 12, 128]");

    TORCH_CHECK(query.scalar_type() == key.scalar_type(),
                "query/key dtype must match");

    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(pooled_q.is_contiguous(), "pooled_q must be contiguous");
    TORCH_CHECK(pooled_k.is_contiguous(), "pooled_k must be contiguous");
    TORCH_CHECK(gemm_out.is_contiguous(), "gemm_out must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    TORCH_CHECK(query.size(1) % T_ == 0,
                "query.size(1) must be divisible by T_=", T_);
    TORCH_CHECK(key.size(1) % T_ == 0,
                "key.size(1) must be divisible by T_=", T_);

    const int qf = static_cast<int>(query.size(1) / T_);
    const int kf = static_cast<int>(key.size(1) / T_);
    const int qp = qf * P_;
    const int kp = kf * P_;

    TORCH_CHECK(pooled_q.sizes() == torch::IntArrayRef({H_, qp, D_}),
                "pooled_q shape must be [12, qf*30, 128]");
    TORCH_CHECK(pooled_k.sizes() == torch::IntArrayRef({H_, kp, D_}),
                "pooled_k shape must be [12, kf*30, 128]");
    TORCH_CHECK(gemm_out.sizes() == torch::IntArrayRef({H_, qp, kp}),
                "gemm_out shape must be [12, qf*30, kf*30]");
    TORCH_CHECK(out.sizes() == torch::IntArrayRef({1, qf, H_, kf}),
                "out shape must be [1, qf, 12, kf]");

    auto stream = at::cuda::getCurrentCUDAStream();

    const int total_tasks = H_ * qp + H_ * kp;
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
                pooled_k.data_ptr<scalar_t>(),
                qf, kf, qp, kp
            );
        }
    );

    switch (query.scalar_type()) {
        case torch::kFloat:
            run_gemm<float>(pooled_q, pooled_k, gemm_out, stream, qp, kp);
            break;
        case torch::kHalf:
            run_gemm<c10::Half>(pooled_q, pooled_k, gemm_out, stream, qp, kp);
            break;
        case torch::kBFloat16:
            run_gemm<c10::BFloat16>(pooled_q, pooled_k, gemm_out, stream, qp, kp);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype");
    }

    TORCH_CHECK(kf <= 1024, "kf too large for one CUDA block: ", kf);

    dim3 grid2(qf, H_);
    dim3 block2(kf);
    size_t smem_bytes = (kf + 2) * sizeof(float);

    reduce_norm_kernel<<<grid2, block2, smem_bytes, stream>>>(
        gemm_out.data_ptr<float>(),
        out.data_ptr<float>(),
        qf, kf, qp, kp
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

