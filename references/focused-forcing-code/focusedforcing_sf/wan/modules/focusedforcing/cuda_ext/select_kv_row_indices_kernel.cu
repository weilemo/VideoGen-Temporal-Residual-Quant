#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cfloat>

namespace {

constexpr int QF_FIXED = 3;
constexpr int H_FIXED = 12;
constexpr int FH_FIXED = 18;
constexpr int TOTAL_FRAMES = 21;
constexpr int T_FIXED = 1560;

// one block per (qf, h)
__global__ void select_kv_row_indices_kernel(
    const float* __restrict__ scores,      // [3,12,18]
    const int32_t* __restrict__ kv_budget, // [3,12]
    const int32_t* __restrict__ offsets,   // [3,12], in frame units
    int32_t* __restrict__ out              // [sum(kv_budget)*1560]
) {
    const int qh = blockIdx.x;   // 0..35
    const int tid = threadIdx.x;

    const int qf = qh / H_FIXED;
    const int h  = qh % H_FIXED;

    const int score_base = (qf * H_FIXED + h) * FH_FIXED;
    const int budget_idx = qf * H_FIXED + h;

    __shared__ int selected_mask[TOTAL_FRAMES];
    __shared__ int selected_list[TOTAL_FRAMES];
    __shared__ int selected_count;
    __shared__ int out_frame_base;

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < TOTAL_FRAMES; ++i) {
            selected_mask[i] = 0;
        }

        int front_best = 0;
        float best_front = scores[score_base + 0];
        for (int i = 1; i < 3; ++i) {
            float v = scores[score_base + i];
            if (v > best_front) {
                best_front = v;
                front_best = i;
            }
        }

        selected_mask[front_best] = 1;
        selected_mask[18] = 1;
        selected_mask[19] = 1;
        selected_mask[20] = 1;

        int need = kv_budget[budget_idx] - 4;
        if (need < 0) need = 0;
        if (need > 17) need = 17;

        int cand_idx[17];
        float cand_val[17];
        int n_cand = 0;

        #pragma unroll
        for (int f = 0; f < FH_FIXED; ++f) {
            if (f != front_best) {
                cand_idx[n_cand] = f;
                cand_val[n_cand] = scores[score_base + f];
                ++n_cand;
            }
        }

        for (int i = 1; i < 17; ++i) {
            int key_idx = cand_idx[i];
            float key_val = cand_val[i];
            int j = i - 1;

            while (j >= 0) {
                bool should_shift = false;
                if (cand_val[j] < key_val) {
                    should_shift = true;
                } else if (cand_val[j] == key_val && cand_idx[j] > key_idx) {
                    should_shift = true;
                }

                if (!should_shift) break;

                cand_idx[j + 1] = cand_idx[j];
                cand_val[j + 1] = cand_val[j];
                --j;
            }
            cand_idx[j + 1] = key_idx;
            cand_val[j + 1] = key_val;
        }

        for (int i = 0; i < need; ++i) {
            selected_mask[cand_idx[i]] = 1;
        }

        int cnt = 0;
        #pragma unroll
        for (int f = 0; f < TOTAL_FRAMES; ++f) {
            if (selected_mask[f]) {
                selected_list[cnt] = f;
                ++cnt;
            }
        }

        selected_count = cnt;
        out_frame_base = offsets[budget_idx];
    }

    __syncthreads();

    const int total_rows = selected_count * T_FIXED;
    const int base_out = out_frame_base * T_FIXED;

    for (int linear = tid; linear < total_rows; linear += blockDim.x) {
        const int frame_rank = linear / T_FIXED;
        const int token = linear - frame_rank * T_FIXED;
        const int frame = selected_list[frame_rank];

        out[base_out + linear] = ((frame * T_FIXED + token) * H_FIXED + h);
    }
}

} // namespace

void select_kv_row_indices_cuda(
    torch::Tensor scores,
    torch::Tensor kv_budget,
    torch::Tensor offsets,
    torch::Tensor out
) {
    const c10::cuda::CUDAGuard device_guard(scores.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    constexpr int blocks = QF_FIXED * H_FIXED;  // 36
    constexpr int threads = 256;

    select_kv_row_indices_kernel<<<blocks, threads, 0, stream>>>(
        scores.data_ptr<float>(),
        kv_budget.data_ptr<int32_t>(),
        offsets.data_ptr<int32_t>(),
        out.data_ptr<int32_t>()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
