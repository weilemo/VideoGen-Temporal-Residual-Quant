#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int QF_FIXED = 3;
constexpr int T_FIXED  = 1560;
constexpr int H_FIXED  = 12;
constexpr int D_FIXED  = 128;
constexpr int LQ_FIXED = 4680;
constexpr int LK_FIXED = 32760;

constexpr int Q_ROWS = QF_FIXED * T_FIXED * H_FIXED; // 56160
constexpr int D_U32  = D_FIXED / 2;                  // 64 uint32 per bf16 row

constexpr int WARP_SIZE_ = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS = WARP_SIZE_ * WARPS_PER_BLOCK;   // 128
constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK;         // 4 rows per block

// q_out[row] = roped_query[0, qf*T + token, h, :]
__global__ void pack_q_kernel(
    const uint32_t* __restrict__ q_src_u32,   // flattened [4680*12, 64]
    uint32_t* __restrict__ q_out_u32          // [56160, 64]
) {
    const int warp_id = threadIdx.x >> 5;       // 0..3
    const int lane    = threadIdx.x & 31;       // 0..31

    const int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;
    if (row >= Q_ROWS) return;

    const int group = row / T_FIXED;            // 0..35
    const int token = row - group * T_FIXED;    // 0..1559

    const int qf = group / H_FIXED;             // 0..2
    const int h  = group - qf * H_FIXED;        // 0..11

    const int src_l = qf * T_FIXED + token;     // 0..4679
    const int src_row = src_l * H_FIXED + h;    // flattened [Lq*H]

    const uint32_t* src = q_src_u32 + src_row * D_U32;
    uint32_t* dst = q_out_u32 + row * D_U32;

    #pragma unroll
    for (int i = lane; i < D_U32; i += WARP_SIZE_) {
        dst[i] = src[i];
    }
}

// k_out[row] = key_flat[kv_rows[row]]
// v_out[row] = value_flat[kv_rows[row]]
__global__ void pack_kv_kernel(
    const uint32_t* __restrict__ k_src_u32,   // flattened [32760*12, 64]
    const uint32_t* __restrict__ v_src_u32,   // flattened [32760*12, 64]
    const int32_t* __restrict__ kv_rows,      // [N]
    uint32_t* __restrict__ k_out_u32,         // [N, 64]
    uint32_t* __restrict__ v_out_u32,         // [N, 64]
    int n_rows
) {
    const int warp_id = threadIdx.x >> 5;      // 0..3
    const int lane    = threadIdx.x & 31;      // 0..31

    const int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;
    if (row >= n_rows) return;

    const int src_row = kv_rows[row];

    const uint32_t* src_k = k_src_u32 + src_row * D_U32;
    const uint32_t* src_v = v_src_u32 + src_row * D_U32;
    uint32_t* dst_k = k_out_u32 + row * D_U32;
    uint32_t* dst_v = v_out_u32 + row * D_U32;

    #pragma unroll
    for (int i = lane; i < D_U32; i += WARP_SIZE_) {
        dst_k[i] = src_k[i];
        dst_v[i] = src_v[i];
    }
}

} // namespace

void pack_qkv_cuda(
    torch::Tensor roped_query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor kv_rows,
    torch::Tensor q_out,
    torch::Tensor k_out,
    torch::Tensor v_out
) {
    const c10::cuda::CUDAGuard device_guard(roped_query.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const uint32_t* q_src_u32 = reinterpret_cast<const uint32_t*>(roped_query.data_ptr());
    const uint32_t* k_src_u32 = reinterpret_cast<const uint32_t*>(key.data_ptr());
    const uint32_t* v_src_u32 = reinterpret_cast<const uint32_t*>(value.data_ptr());

    uint32_t* q_out_u32 = reinterpret_cast<uint32_t*>(q_out.data_ptr());
    uint32_t* k_out_u32 = reinterpret_cast<uint32_t*>(k_out.data_ptr());
    uint32_t* v_out_u32 = reinterpret_cast<uint32_t*>(v_out.data_ptr());

    const int32_t* kv_rows_ptr = kv_rows.data_ptr<int32_t>();
    const int n_rows = static_cast<int>(kv_rows.size(0));

    const int q_blocks  = (Q_ROWS + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const int kv_blocks = (n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    pack_q_kernel<<<q_blocks, THREADS, 0, stream>>>(
        q_src_u32,
        q_out_u32
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

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
