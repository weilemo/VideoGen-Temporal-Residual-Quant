#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int B_FIXED = 1;
constexpr int H_FIXED = 12;
constexpr int D_FIXED = 128;

constexpr int WARP_SIZE_ = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS = WARP_SIZE_ * WARPS_PER_BLOCK;   // 128
constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK;         // 4 rows per block

template <typename scalar_t>
__global__ void concat2_fixed_b1_kernel(
    const scalar_t* __restrict__ a,   // [1, La, 12, 128]
    const scalar_t* __restrict__ b,   // [1, Lb, 12, 128]
    scalar_t* __restrict__ out,       // [1, Lo, 12, 128]
    int La,
    int Lb
) {
    const int warp_id = threadIdx.x >> 5;   // 0..3
    const int lane    = threadIdx.x & 31;   // 0..31

    const int Lo = La + Lb;
    const int total_rows = Lo * H_FIXED;

    const int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;
    if (row >= total_rows) return;

    const int l = row / H_FIXED;         // 0..Lo-1
    const int h = row - l * H_FIXED;     // 0..11

    const scalar_t* src_ptr = nullptr;
    int src_l = 0;

    if (l < La) {
        src_ptr = a;
        src_l = l;
    } else {
        src_ptr = b;
        src_l = l - La;
    }

    const int src_row = src_l * H_FIXED + h;
    const int dst_row = l * H_FIXED + h;

    const scalar_t* src = src_ptr + src_row * D_FIXED;
    scalar_t* dst = out + dst_row * D_FIXED;

    #pragma unroll
    for (int i = lane; i < D_FIXED; i += WARP_SIZE_) {
        dst[i] = src[i];
    }
}

} // namespace

void concat_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor out
) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
                "all tensors must be contiguous");

    TORCH_CHECK(a.dim() == 4 && b.dim() == 4 && out.dim() == 4,
                "all tensors must be [1, L, 12, 128]");
    TORCH_CHECK(a.size(0) == B_FIXED && b.size(0) == B_FIXED && out.size(0) == B_FIXED,
                "batch must be 1");
    TORCH_CHECK(a.size(2) == H_FIXED && b.size(2) == H_FIXED && out.size(2) == H_FIXED,
                "head dim must be 12");
    TORCH_CHECK(a.size(3) == D_FIXED && b.size(3) == D_FIXED && out.size(3) == D_FIXED,
                "last dim must be 128");

    TORCH_CHECK(a.scalar_type() == b.scalar_type() &&
                a.scalar_type() == out.scalar_type(),
                "all tensors must have same dtype");

    const int La = static_cast<int>(a.size(1));
    const int Lb = static_cast<int>(b.size(1));
    const int Lo = static_cast<int>(out.size(1));

    TORCH_CHECK(Lo == La + Lb,
                "out.size(1) must equal La + Lb");

    const int total_rows = Lo * H_FIXED;
    const int blocks = (total_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    const c10::cuda::CUDAGuard device_guard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    switch (a.scalar_type()) {
        case torch::kFloat32:
            concat2_fixed_b1_kernel<float><<<blocks, THREADS, 0, stream>>>(
                a.data_ptr<float>(),
                b.data_ptr<float>(),
                out.data_ptr<float>(),
                La, Lb
            );
            break;
        case torch::kFloat16:
            concat2_fixed_b1_kernel<c10::Half><<<blocks, THREADS, 0, stream>>>(
                a.data_ptr<c10::Half>(),
                b.data_ptr<c10::Half>(),
                out.data_ptr<c10::Half>(),
                La, Lb
            );
            break;
        case torch::kBFloat16:
            concat2_fixed_b1_kernel<c10::BFloat16><<<blocks, THREADS, 0, stream>>>(
                a.data_ptr<c10::BFloat16>(),
                b.data_ptr<c10::BFloat16>(),
                out.data_ptr<c10::BFloat16>(),
                La, Lb
            );
            break;
        default:
            TORCH_CHECK(false, "only float32 / float16 / bfloat16 are supported");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
