#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int T_FIXED  = 1560;
constexpr int H_FIXED  = 12;
constexpr int D_FIXED  = 128;

constexpr int D_U32  = D_FIXED / 2;        // 64 uint32 per fp16/bf16 row
constexpr int WARP_SIZE_ = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS = WARP_SIZE_ * WARPS_PER_BLOCK;   // 128
constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK;         // 4 rows per block

// ------------------------------------------------------------
// q pack
//
// input logical shape:
//   roped_query: [1, Lq, 12, 128]
//
// output logical packed order:
//   q_out rows ordered as:
//     (qf=0,h=0,t=0..1559),
//     (qf=0,h=1,t=0..1559),
//     ...
//     (qf=QF_total-1,h=11,t=0..1559)
//
// so each (qf, h) becomes one contiguous varlen sequence.
// output shape:
//   q_out: [Lq*12, 128]
// ------------------------------------------------------------
__global__ void pack_q_kernel(
    const uint32_t* __restrict__ q_src_u32,   // flattened [Lq*H, 64]
    uint32_t* __restrict__ q_out_u32,         // [Lq*H, 64]
    int q_rows
) {
    const int warp_id = threadIdx.x >> 5;   // 0..3
    const int lane    = threadIdx.x & 31;   // 0..31

    const int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;
    if (row >= q_rows) return;

    // q_rows = QF_total * T_FIXED * H_FIXED
    // packed row ordering:
    //   group = (qf, h), token varies fastest
    const int group = row / T_FIXED;            // 0 .. QF_total*H_FIXED-1
    const int token = row - group * T_FIXED;    // 0 .. 1559

    const int qf = group / H_FIXED;             // 0 .. QF_total-1
    const int h  = group - qf * H_FIXED;        // 0 .. 11

    const int src_l   = qf * T_FIXED + token;   // 0 .. Lq-1
    const int src_row = src_l * H_FIXED + h;    // flattened [Lq*H]

    const uint32_t* src = q_src_u32 + src_row * D_U32;
    uint32_t* dst       = q_out_u32 + row * D_U32;

    #pragma unroll
    for (int i = lane; i < D_U32; i += WARP_SIZE_) {
        dst[i] = src[i];
    }
}

// ------------------------------------------------------------
// kv pack
//
// key/value logical shape: [1, Lk, H, D]
// kv_rows stores flattened row indices in [0, Lk*H)
//
// output rows follow kv_rows order directly.
// ------------------------------------------------------------
__global__ void pack_kv_kernel(
    const uint32_t* __restrict__ k_src_u32,   // flattened [Lk*H, 64]
    const uint32_t* __restrict__ v_src_u32,   // flattened [Lk*H, 64]
    const int32_t* __restrict__ kv_rows,      // [N]
    uint32_t* __restrict__ k_out_u32,         // [N, 64]
    uint32_t* __restrict__ v_out_u32,         // [N, 64]
    int n_rows
) {
    const int warp_id = threadIdx.x >> 5;   // 0..3
    const int lane    = threadIdx.x & 31;   // 0..31

    const int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;
    if (row >= n_rows) return;

    const int src_row = kv_rows[row];

    const uint32_t* src_k = k_src_u32 + src_row * D_U32;
    const uint32_t* src_v = v_src_u32 + src_row * D_U32;
    uint32_t* dst_k       = k_out_u32 + row * D_U32;
    uint32_t* dst_v       = v_out_u32 + row * D_U32;

    #pragma unroll
    for (int i = lane; i < D_U32; i += WARP_SIZE_) {
        dst_k[i] = src_k[i];
        dst_v[i] = src_v[i];
    }
}

} // namespace

void pack_qkv_cuda(
    torch::Tensor roped_query,   // [1, Lq, 12, 128], fp16/bf16
    torch::Tensor key,           // [1, Lk, 12, 128], fp16/bf16
    torch::Tensor value,         // [1, Lk, 12, 128], fp16/bf16
    torch::Tensor kv_rows,       // [N], int32
    torch::Tensor q_out,         // [Lq*12, 128]
    torch::Tensor k_out,         // [N, 128]
    torch::Tensor v_out          // [N, 128]
) {
    TORCH_CHECK(roped_query.is_cuda(), "roped_query must be CUDA");
    TORCH_CHECK(key.is_cuda(), "key must be CUDA");
    TORCH_CHECK(value.is_cuda(), "value must be CUDA");
    TORCH_CHECK(kv_rows.is_cuda(), "kv_rows must be CUDA");
    TORCH_CHECK(q_out.is_cuda(), "q_out must be CUDA");
    TORCH_CHECK(k_out.is_cuda(), "k_out must be CUDA");
    TORCH_CHECK(v_out.is_cuda(), "v_out must be CUDA");

    TORCH_CHECK(roped_query.dim() == 4, "roped_query must be [1, Lq, H, D]");
    TORCH_CHECK(key.dim() == 4, "key must be [1, Lk, H, D]");
    TORCH_CHECK(value.dim() == 4, "value must be [1, Lk, H, D]");

    TORCH_CHECK(roped_query.size(0) == 1, "roped_query batch must be 1");
    TORCH_CHECK(key.size(0) == 1, "key batch must be 1");
    TORCH_CHECK(value.size(0) == 1, "value batch must be 1");

    TORCH_CHECK(roped_query.size(2) == H_FIXED && roped_query.size(3) == D_FIXED,
                "roped_query must have shape [1, Lq, 12, 128]");
    TORCH_CHECK(key.size(2) == H_FIXED && key.size(3) == D_FIXED,
                "key must have shape [1, Lk, 12, 128]");
    TORCH_CHECK(value.size(2) == H_FIXED && value.size(3) == D_FIXED,
                "value must have shape [1, Lk, 12, 128]");

    TORCH_CHECK(key.sizes() == value.sizes(), "key/value shapes must match");

    TORCH_CHECK(
        roped_query.scalar_type() == at::kHalf ||
        roped_query.scalar_type() == at::kBFloat16,
        "roped_query/key/value must be fp16 or bf16"
    );
    TORCH_CHECK(key.scalar_type() == roped_query.scalar_type(),
                "roped_query/key dtype must match");
    TORCH_CHECK(value.scalar_type() == roped_query.scalar_type(),
                "roped_query/value dtype must match");

    TORCH_CHECK(q_out.scalar_type() == roped_query.scalar_type(),
                "q_out dtype must match roped_query");
    TORCH_CHECK(k_out.scalar_type() == key.scalar_type(),
                "k_out dtype must match key");
    TORCH_CHECK(v_out.scalar_type() == value.scalar_type(),
                "v_out dtype must match value");

    TORCH_CHECK(kv_rows.scalar_type() == at::kInt, "kv_rows must be int32");

    TORCH_CHECK(roped_query.is_contiguous(), "roped_query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(kv_rows.is_contiguous(), "kv_rows must be contiguous");
    TORCH_CHECK(q_out.is_contiguous(), "q_out must be contiguous");
    TORCH_CHECK(k_out.is_contiguous(), "k_out must be contiguous");
    TORCH_CHECK(v_out.is_contiguous(), "v_out must be contiguous");

    const int Lq = static_cast<int>(roped_query.size(1));
    const int Lk = static_cast<int>(key.size(1));
    const int q_rows = Lq * H_FIXED;
    const int n_rows = static_cast<int>(kv_rows.size(0));

    TORCH_CHECK(Lq % T_FIXED == 0,
                "roped_query.size(1) must be divisible by T_FIXED=", T_FIXED);
    TORCH_CHECK(Lk % T_FIXED == 0,
                "key.size(1) must be divisible by T_FIXED=", T_FIXED);

    TORCH_CHECK(q_out.dim() == 2 &&
                q_out.size(0) == q_rows &&
                q_out.size(1) == D_FIXED,
                "q_out must have shape [Lq*12, 128]");

    TORCH_CHECK(k_out.dim() == 2 &&
                k_out.size(0) == n_rows &&
                k_out.size(1) == D_FIXED,
                "k_out must have shape [N, 128]");

    TORCH_CHECK(v_out.dim() == 2 &&
                v_out.size(0) == n_rows &&
                v_out.size(1) == D_FIXED,
                "v_out must have shape [N, 128]");

    if (n_rows > 0) {
        TORCH_CHECK(kv_rows.min().item<int>() >= 0, "kv_rows must be non-negative");
        TORCH_CHECK(kv_rows.max().item<int>() < Lk * H_FIXED,
                    "kv_rows contains out-of-range indices");
    }

    const c10::cuda::CUDAGuard device_guard(roped_query.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const uint32_t* q_src_u32 = reinterpret_cast<const uint32_t*>(roped_query.data_ptr());
    const uint32_t* k_src_u32 = reinterpret_cast<const uint32_t*>(key.data_ptr());
    const uint32_t* v_src_u32 = reinterpret_cast<const uint32_t*>(value.data_ptr());

    uint32_t* q_out_u32 = reinterpret_cast<uint32_t*>(q_out.data_ptr());
    uint32_t* k_out_u32 = reinterpret_cast<uint32_t*>(k_out.data_ptr());
    uint32_t* v_out_u32 = reinterpret_cast<uint32_t*>(v_out.data_ptr());

    const int32_t* kv_rows_ptr = kv_rows.data_ptr<int32_t>();

    const int q_blocks  = (q_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const int kv_blocks = (n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    pack_q_kernel<<<q_blocks, THREADS, 0, stream>>>(
        q_src_u32,
        q_out_u32,
        q_rows
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (n_rows > 0) {
        pack_kv_kernel<<<kv_blocks, THREADS, 0, stream>>>(
            k_src_u32,
            v_src_u32,
            kv_rows_ptr,
            k_out_u32,
            v_out_u32,
            n_rows
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

