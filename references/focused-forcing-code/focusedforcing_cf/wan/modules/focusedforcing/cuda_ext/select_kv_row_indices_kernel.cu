#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cfloat>

namespace {

constexpr int H_FIXED = 12;
constexpr int TOTAL_FRAMES = 21;
constexpr int FRONT_FIXED = 3;
constexpr int T_FIXED = 1560;
constexpr int QF_MAX = TOTAL_FRAMES - FRONT_FIXED; // 18

__device__ __forceinline__ void sort_desc_score_tie_small_idx(
    int* idxs,
    float* vals,
    int n
) {
    for (int i = 1; i < n; ++i) {
        int key_idx = idxs[i];
        float key_val = vals[i];
        int j = i - 1;

        while (j >= 0) {
            bool should_shift = false;
            if (vals[j] < key_val) {
                should_shift = true;
            } else if (vals[j] == key_val && idxs[j] > key_idx) {
                should_shift = true;
            }

            if (!should_shift) break;

            idxs[j + 1] = idxs[j];
            vals[j + 1] = vals[j];
            --j;
        }

        idxs[j + 1] = key_idx;
        vals[j + 1] = key_val;
    }
}

// one block per (qf_idx, h)
__global__ void select_kv_row_indices_kernel(
    const float* __restrict__ scores,      // [QF_total, 12, 21]
    const int32_t* __restrict__ kv_budget, // [QF_total, 12]
    const int32_t* __restrict__ offsets,   // [QF_total, 12], in frame units
    int32_t* __restrict__ out,             // [sum(kv_budget) * 1560]
    int QF_total,
    bool update
) {
    const int qh = blockIdx.x;
    const int tid = threadIdx.x;

    const int qf_idx = qh / H_FIXED;
    const int h      = qh % H_FIXED;

    if (qf_idx >= QF_total) return;

    const int score_base = (qf_idx * H_FIXED + h) * TOTAL_FRAMES;
    const int budget_idx = qf_idx * H_FIXED + h;

    __shared__ int selected_mask[TOTAL_FRAMES];
    __shared__ int selected_list[TOTAL_FRAMES];
    __shared__ int selected_count;
    __shared__ int out_frame_base;

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < TOTAL_FRAMES; ++i) {
            selected_mask[i] = 0;
        }

        if (update) {
            // legacy rule:
            // 1) best from front 3: 0,1,2
            // 2) always keep 18,19,20
            // 3) remaining from 0..17 except front_best
            int front_best = 0;
            float best_front = scores[score_base + 0];
            for (int i = 1; i < FRONT_FIXED; ++i) {
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

            for (int f = 0; f < TOTAL_FRAMES - FRONT_FIXED; ++f) {
                if (f != front_best) {
                    cand_idx[n_cand] = f;
                    cand_val[n_cand] = scores[score_base + f];
                    ++n_cand;
                }
            }

            sort_desc_score_tie_small_idx(cand_idx, cand_val, n_cand);

            for (int i = 0; i < need; ++i) {
                selected_mask[cand_idx[i]] = 1;
            }

        } else {
            // new rule:
            // 1) best from front 3: 0,1,2
            // 2) tail has QF_total frames, keep matched tail frame
            // 3) remaining budget split proportionally between:
            //    middle [3, tail_start)
            //    tail_rest [tail_start, 21) except tail_match
            const int tail_start = TOTAL_FRAMES - QF_total;
            const int tail_match = tail_start + qf_idx;

            int front_best = 0;
            float best_front = scores[score_base + 0];
            for (int i = 1; i < FRONT_FIXED; ++i) {
                float v = scores[score_base + i];
                if (v > best_front) {
                    best_front = v;
                    front_best = i;
                }
            }

            selected_mask[front_best] = 1;
            selected_mask[tail_match] = 1;

            int remain = kv_budget[budget_idx] - 2;
            if (remain < 0) remain = 0;

            int mid_idx[TOTAL_FRAMES];
            float mid_val[TOTAL_FRAMES];
            int n_mid = 0;

            for (int f = FRONT_FIXED; f < tail_start; ++f) {
                mid_idx[n_mid] = f;
                mid_val[n_mid] = scores[score_base + f];
                ++n_mid;
            }

            int tail_idx[TOTAL_FRAMES];
            float tail_val[TOTAL_FRAMES];
            int n_tail = 0;

            for (int f = tail_start; f < TOTAL_FRAMES; ++f) {
                if (f != tail_match) {
                    tail_idx[n_tail] = f;
                    tail_val[n_tail] = scores[score_base + f];
                    ++n_tail;
                }
            }

            int mid_quota = 0;
            int tail_quota = 0;

            if (remain > 0 && (n_mid + n_tail) > 0) {
                mid_quota = (remain * n_mid) / (n_mid + n_tail);
                tail_quota = remain - mid_quota;

                if (mid_quota > n_mid) mid_quota = n_mid;
                if (tail_quota > n_tail) tail_quota = n_tail;

                int leftover = remain - mid_quota - tail_quota;

                if (leftover > 0) {
                    int give_mid = n_mid - mid_quota;
                    if (give_mid > leftover) give_mid = leftover;
                    mid_quota += give_mid;
                    leftover -= give_mid;
                }

                if (leftover > 0) {
                    int give_tail = n_tail - tail_quota;
                    if (give_tail > leftover) give_tail = leftover;
                    tail_quota += give_tail;
                    leftover -= give_tail;
                }

                sort_desc_score_tie_small_idx(mid_idx, mid_val, n_mid);
                sort_desc_score_tie_small_idx(tail_idx, tail_val, n_tail);

                for (int i = 0; i < mid_quota; ++i) {
                    selected_mask[mid_idx[i]] = 1;
                }
                for (int i = 0; i < tail_quota; ++i) {
                    selected_mask[tail_idx[i]] = 1;
                }
            }
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
    torch::Tensor scores,    // [QF_total, 12, 21], float32
    torch::Tensor kv_budget, // [QF_total, 12], int32
    torch::Tensor offsets,   // [QF_total, 12], int32, in frame units
    torch::Tensor out,       // [sum(kv_budget) * 1560], int32
    bool update
) {
    TORCH_CHECK(scores.is_cuda(), "scores must be CUDA");
    TORCH_CHECK(kv_budget.is_cuda(), "kv_budget must be CUDA");
    TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");

    TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
    TORCH_CHECK(kv_budget.scalar_type() == at::kInt, "kv_budget must be int32");
    TORCH_CHECK(offsets.scalar_type() == at::kInt, "offsets must be int32");
    TORCH_CHECK(out.scalar_type() == at::kInt, "out must be int32");

    TORCH_CHECK(scores.dim() == 3, "scores must be [QF_total, 12, 21]");
    TORCH_CHECK(kv_budget.dim() == 2, "kv_budget must be [QF_total, 12]");
    TORCH_CHECK(offsets.dim() == 2, "offsets must be [QF_total, 12]");
    TORCH_CHECK(out.dim() == 1, "out must be 1D");

    const int QF_total = static_cast<int>(scores.size(0));

    TORCH_CHECK(QF_total >= 1 && QF_total <= QF_MAX,
                "QF_total must be in [1, ", QF_MAX, "]");
    TORCH_CHECK(scores.size(1) == H_FIXED, "scores.size(1) must be 12");
    TORCH_CHECK(scores.size(2) == TOTAL_FRAMES, "scores.size(2) must be 21");
    TORCH_CHECK(kv_budget.size(0) == QF_total && kv_budget.size(1) == H_FIXED,
                "kv_budget must be [QF_total, 12]");
    TORCH_CHECK(offsets.size(0) == QF_total && offsets.size(1) == H_FIXED,
                "offsets must be [QF_total, 12]");

    const c10::cuda::CUDAGuard device_guard(scores.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int blocks = QF_total * H_FIXED;
    constexpr int threads = 256;

    select_kv_row_indices_kernel<<<blocks, threads, 0, stream>>>(
        scores.data_ptr<float>(),
        kv_budget.data_ptr<int32_t>(),
        offsets.data_ptr<int32_t>(),
        out.data_ptr<int32_t>(),
        QF_total,
        update
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
