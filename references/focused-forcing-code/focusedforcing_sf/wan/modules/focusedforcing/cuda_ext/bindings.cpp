#include <torch/extension.h>

// ============================================================
// CUDA function declarations
// ============================================================

void compute_key_diversity_cuda(
    torch::Tensor k,          // [30,1,28080,12,128], bf16/fp16/fp32
    torch::Tensor mean_pos,   // [30,1560,12,128], same dtype as k
    torch::Tensor out_div     // [30,1,12,18], float32
);

void rope_apply_temporal_shift_cuda(
    torch::Tensor k_all,     // [30,1,L,12,128], can be non-contiguous
    torch::Tensor mult_re,   // [22], float64
    torch::Tensor mult_im    // [22], float64
);

void compute_attn_scores_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor pooled_q,
    torch::Tensor pooled_k,
    torch::Tensor gemm_out,
    torch::Tensor out
);

void causal_rope_apply_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor grid_sizes,
    torch::Tensor freqs0_re,
    torch::Tensor freqs0_im,
    torch::Tensor freqs1_re,
    torch::Tensor freqs1_im,
    torch::Tensor freqs2_re,
    torch::Tensor freqs2_im,
    torch::Tensor out_q,
    torch::Tensor out_k,
    int start_frame
);

void pack_qkv_cuda(
    torch::Tensor roped_query,   // [1,4680,12,128], bf16
    torch::Tensor key,           // [1,32760,12,128], bf16
    torch::Tensor value,         // [1,32760,12,128], bf16
    torch::Tensor kv_rows,       // [N], int32
    torch::Tensor q_out,         // [56160,128], bf16
    torch::Tensor k_out,         // [N,128], bf16
    torch::Tensor v_out          // [N,128], bf16
);

void select_kv_row_indices_cuda(
    torch::Tensor scores,     // [3,12,18], float32 CUDA contiguous
    torch::Tensor kv_budget,  // [3,12], int32 CUDA contiguous
    torch::Tensor offsets,    // [3,12], int32 CUDA contiguous
    torch::Tensor out         // [sum(kv_budget)*1560], int32 CUDA contiguous
);

// ============================================================
// C++ wrappers with checks
// ============================================================

void compute_key_diversity(
    torch::Tensor k,
    torch::Tensor mean_pos,
    torch::Tensor out_div
) {
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(mean_pos.is_cuda(), "mean_pos must be CUDA");
    TORCH_CHECK(out_div.is_cuda(), "out_div must be CUDA");

    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(mean_pos.is_contiguous(), "mean_pos must be contiguous");
    TORCH_CHECK(out_div.is_contiguous(), "out_div must be contiguous");

    TORCH_CHECK(k.dim() == 5, "k must be [30,1,28080,12,128]");
    TORCH_CHECK(k.size(0) == 30, "num_blocks must be 30");
    TORCH_CHECK(k.size(1) == 1, "B must be 1");
    TORCH_CHECK(k.size(2) == 28080, "L must be 28080 (=18*1560)");
    TORCH_CHECK(k.size(3) == 12, "H must be 12");
    TORCH_CHECK(k.size(4) == 128, "D must be 128");

    TORCH_CHECK(mean_pos.dim() == 4, "mean_pos must be [30,1560,12,128]");
    TORCH_CHECK(mean_pos.size(0) == 30, "mean_pos.size(0) must be 30");
    TORCH_CHECK(mean_pos.size(1) == 1560, "mean_pos.size(1) must be 1560");
    TORCH_CHECK(mean_pos.size(2) == 12, "mean_pos.size(2) must be 12");
    TORCH_CHECK(mean_pos.size(3) == 128, "mean_pos.size(3) must be 128");

    TORCH_CHECK(out_div.dim() == 4, "out_div must be [30,1,12,18]");
    TORCH_CHECK(out_div.size(0) == 30, "out_div.size(0) must be 30");
    TORCH_CHECK(out_div.size(1) == 1, "out_div.size(1) must be 1");
    TORCH_CHECK(out_div.size(2) == 12, "out_div.size(2) must be 12");
    TORCH_CHECK(out_div.size(3) == 18, "out_div.size(3) must be 18");
    TORCH_CHECK(out_div.scalar_type() == torch::kFloat32, "out_div must be float32");

    TORCH_CHECK(
        k.scalar_type() == torch::kFloat32 ||
        k.scalar_type() == torch::kFloat16 ||
        k.scalar_type() == torch::kBFloat16,
        "k must be float32/float16/bfloat16"
    );

    TORCH_CHECK(mean_pos.scalar_type() == k.scalar_type(), "mean_pos dtype must match k");

    compute_key_diversity_cuda(k, mean_pos, out_div);
}

void rope_apply_temporal_shift(
    torch::Tensor k_all,
    torch::Tensor mult_re,
    torch::Tensor mult_im
) {
    TORCH_CHECK(k_all.is_cuda(), "k_all must be CUDA");
    TORCH_CHECK(mult_re.is_cuda(), "mult_re must be CUDA");
    TORCH_CHECK(mult_im.is_cuda(), "mult_im must be CUDA");

    TORCH_CHECK(mult_re.is_contiguous(), "mult_re must be contiguous");
    TORCH_CHECK(mult_im.is_contiguous(), "mult_im must be contiguous");

    TORCH_CHECK(k_all.dim() == 5, "k_all must be [30,1,L,H,D]");
    TORCH_CHECK(k_all.size(0) == 30, "fixed-shape kernel requires num_blocks=30");
    TORCH_CHECK(k_all.size(1) == 1, "fixed-shape kernel requires B=1");
    TORCH_CHECK(k_all.size(3) == 12, "fixed-shape kernel requires H=12");
    TORCH_CHECK(k_all.size(4) == 128, "fixed-shape kernel requires D=128");

    TORCH_CHECK(mult_re.dim() == 1 && mult_im.dim() == 1, "mult must be 1D");
    TORCH_CHECK(mult_re.numel() == 22 && mult_im.numel() == 22, "fixed-shape kernel requires t_c=22");

    TORCH_CHECK(mult_re.scalar_type() == torch::kFloat64, "mult_re must be float64");
    TORCH_CHECK(mult_im.scalar_type() == torch::kFloat64, "mult_im must be float64");

    auto st = k_all.scalar_type();
    TORCH_CHECK(
        st == torch::kBFloat16 || st == torch::kFloat16 || st == torch::kFloat32,
        "k_all must be bf16/fp16/fp32"
    );

    rope_apply_temporal_shift_cuda(k_all, mult_re, mult_im);
}

void compute_attn_scores(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor pooled_q,
    torch::Tensor pooled_k,
    torch::Tensor gemm_out,
    torch::Tensor out
) {
    TORCH_CHECK(query.is_cuda(), "query must be CUDA");
    TORCH_CHECK(key.is_cuda(), "key must be CUDA");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");

    TORCH_CHECK(query.dim() == 4, "query must be 4D");
    TORCH_CHECK(key.dim() == 4, "key must be 4D");

    TORCH_CHECK(query.size(0) == 1 && query.size(1) == 4680 &&
                query.size(2) == 12 && query.size(3) == 128,
                "query shape must be [1,4680,12,128]");

    TORCH_CHECK(key.size(0) == 1 && key.size(1) == 32760 &&
                key.size(2) == 12 && key.size(3) == 128,
                "key shape must be [1,32760,12,128]");

    TORCH_CHECK(
        query.scalar_type() == torch::kFloat ||
        query.scalar_type() == torch::kHalf ||
        query.scalar_type() == torch::kBFloat16,
        "query dtype must be fp32/fp16/bf16"
    );
    TORCH_CHECK(key.scalar_type() == query.scalar_type(), "dtype mismatch");

    TORCH_CHECK(pooled_q.sizes() == torch::IntArrayRef({12, 90, 128}), "pooled_q shape mismatch");
    TORCH_CHECK(pooled_k.sizes() == torch::IntArrayRef({12, 540, 128}), "pooled_k shape mismatch");
    TORCH_CHECK(gemm_out.sizes() == torch::IntArrayRef({12, 90, 540}), "gemm_out shape mismatch");
    TORCH_CHECK(out.sizes() == torch::IntArrayRef({1, 3, 12, 18}), "out shape mismatch");

    TORCH_CHECK(pooled_q.scalar_type() == query.scalar_type(), "pooled_q dtype mismatch");
    TORCH_CHECK(pooled_k.scalar_type() == query.scalar_type(), "pooled_k dtype mismatch");
    TORCH_CHECK(gemm_out.scalar_type() == torch::kFloat, "gemm_out must be fp32");
    TORCH_CHECK(out.scalar_type() == torch::kFloat, "out must be fp32");

    compute_attn_scores_cuda(query, key, pooled_q, pooled_k, gemm_out, out);
}

void causal_rope_apply(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor grid_sizes,
    torch::Tensor freqs0_re,
    torch::Tensor freqs0_im,
    torch::Tensor freqs1_re,
    torch::Tensor freqs1_im,
    torch::Tensor freqs2_re,
    torch::Tensor freqs2_im,
    torch::Tensor out_q,
    torch::Tensor out_k,
    int start_frame
) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(grid_sizes.is_cuda(), "grid_sizes must be CUDA");
    TORCH_CHECK(freqs0_re.is_cuda(), "freqs0_re must be CUDA");
    TORCH_CHECK(freqs0_im.is_cuda(), "freqs0_im must be CUDA");
    TORCH_CHECK(freqs1_re.is_cuda(), "freqs1_re must be CUDA");
    TORCH_CHECK(freqs1_im.is_cuda(), "freqs1_im must be CUDA");
    TORCH_CHECK(freqs2_re.is_cuda(), "freqs2_re must be CUDA");
    TORCH_CHECK(freqs2_im.is_cuda(), "freqs2_im must be CUDA");
    TORCH_CHECK(out_q.is_cuda(), "out_q must be CUDA");
    TORCH_CHECK(out_k.is_cuda(), "out_k must be CUDA");

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(grid_sizes.is_contiguous(), "grid_sizes must be contiguous");
    TORCH_CHECK(out_q.is_contiguous(), "out_q must be contiguous");
    TORCH_CHECK(out_k.is_contiguous(), "out_k must be contiguous");

    TORCH_CHECK(q.sizes() == k.sizes(), "q and k shapes must match");
    TORCH_CHECK(q.dim() == 4, "q/k must be [1,4680,12,128]");
    TORCH_CHECK(q.size(0) == 1, "B must be 1");
    TORCH_CHECK(q.size(1) == 4680, "L must be 4680");
    TORCH_CHECK(q.size(2) == 12, "N must be 12");
    TORCH_CHECK(q.size(3) == 128, "D must be 128");

    TORCH_CHECK(out_q.sizes() == q.sizes(), "out_q shape mismatch");
    TORCH_CHECK(out_k.sizes() == q.sizes(), "out_k shape mismatch");

    TORCH_CHECK(grid_sizes.dim() == 2 && grid_sizes.size(0) == 1 && grid_sizes.size(1) == 3,
                "grid_sizes must be [1,3]");
    TORCH_CHECK(grid_sizes.scalar_type() == torch::kInt32, "grid_sizes must be int32");

    TORCH_CHECK(freqs0_re.scalar_type() == torch::kFloat32, "freqs0_re must be float32");
    TORCH_CHECK(freqs0_im.scalar_type() == torch::kFloat32, "freqs0_im must be float32");
    TORCH_CHECK(freqs1_re.scalar_type() == torch::kFloat32, "freqs1_re must be float32");
    TORCH_CHECK(freqs1_im.scalar_type() == torch::kFloat32, "freqs1_im must be float32");
    TORCH_CHECK(freqs2_re.scalar_type() == torch::kFloat32, "freqs2_re must be float32");
    TORCH_CHECK(freqs2_im.scalar_type() == torch::kFloat32, "freqs2_im must be float32");

    TORCH_CHECK(
        q.scalar_type() == torch::kFloat32 ||
        q.scalar_type() == torch::kFloat16 ||
        q.scalar_type() == torch::kBFloat16,
        "q/k must be float32/float16/bfloat16"
    );
    TORCH_CHECK(k.scalar_type() == q.scalar_type(), "k dtype must match q");
    TORCH_CHECK(out_q.scalar_type() == q.scalar_type(), "out_q dtype must match q");
    TORCH_CHECK(out_k.scalar_type() == q.scalar_type(), "out_k dtype must match q");

    causal_rope_apply_cuda(
        q, k, grid_sizes,
        freqs0_re, freqs0_im,
        freqs1_re, freqs1_im,
        freqs2_re, freqs2_im,
        out_q, out_k,
        start_frame
    );
}

void pack_qkv(
    torch::Tensor roped_query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor kv_rows,
    torch::Tensor q_out,
    torch::Tensor k_out,
    torch::Tensor v_out
) {
    TORCH_CHECK(roped_query.is_cuda(), "roped_query must be CUDA");
    TORCH_CHECK(key.is_cuda(), "key must be CUDA");
    TORCH_CHECK(value.is_cuda(), "value must be CUDA");
    TORCH_CHECK(kv_rows.is_cuda(), "kv_rows must be CUDA");
    TORCH_CHECK(q_out.is_cuda(), "q_out must be CUDA");
    TORCH_CHECK(k_out.is_cuda(), "k_out must be CUDA");
    TORCH_CHECK(v_out.is_cuda(), "v_out must be CUDA");

    TORCH_CHECK(roped_query.is_contiguous(), "roped_query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(kv_rows.is_contiguous(), "kv_rows must be contiguous");
    TORCH_CHECK(q_out.is_contiguous(), "q_out must be contiguous");
    TORCH_CHECK(k_out.is_contiguous(), "k_out must be contiguous");
    TORCH_CHECK(v_out.is_contiguous(), "v_out must be contiguous");

    TORCH_CHECK(roped_query.scalar_type() == torch::kBFloat16, "roped_query must be bf16");
    TORCH_CHECK(key.scalar_type() == torch::kBFloat16, "key must be bf16");
    TORCH_CHECK(value.scalar_type() == torch::kBFloat16, "value must be bf16");
    TORCH_CHECK(kv_rows.scalar_type() == torch::kInt32, "kv_rows must be int32");
    TORCH_CHECK(q_out.scalar_type() == torch::kBFloat16, "q_out must be bf16");
    TORCH_CHECK(k_out.scalar_type() == torch::kBFloat16, "k_out must be bf16");
    TORCH_CHECK(v_out.scalar_type() == torch::kBFloat16, "v_out must be bf16");

    TORCH_CHECK(
        roped_query.sizes() == torch::IntArrayRef({1, 4680, 12, 128}),
        "roped_query must be [1,4680,12,128]"
    );
    TORCH_CHECK(
        key.sizes() == torch::IntArrayRef({1, 32760, 12, 128}),
        "key must be [1,32760,12,128]"
    );
    TORCH_CHECK(
        value.sizes() == torch::IntArrayRef({1, 32760, 12, 128}),
        "value must be [1,32760,12,128]"
    );

    TORCH_CHECK(q_out.dim() == 2 && q_out.size(0) == 56160 && q_out.size(1) == 128,
                "q_out must be [56160,128]");
    TORCH_CHECK(k_out.dim() == 2 && k_out.size(1) == 128,
                "k_out must be [N,128]");
    TORCH_CHECK(v_out.dim() == 2 && v_out.size(1) == 128,
                "v_out must be [N,128]");
    TORCH_CHECK(k_out.size(0) == kv_rows.size(0), "k_out size mismatch");
    TORCH_CHECK(v_out.size(0) == kv_rows.size(0), "v_out size mismatch");

    pack_qkv_cuda(roped_query, key, value, kv_rows, q_out, k_out, v_out);
}

void select_kv_row_indices(
    torch::Tensor scores,
    torch::Tensor kv_budget,
    torch::Tensor offsets,
    torch::Tensor out
) {
    TORCH_CHECK(scores.is_cuda(), "scores must be CUDA");
    TORCH_CHECK(kv_budget.is_cuda(), "kv_budget must be CUDA");
    TORCH_CHECK(offsets.is_cuda(), "offsets must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");

    TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
    TORCH_CHECK(kv_budget.is_contiguous(), "kv_budget must be contiguous");
    TORCH_CHECK(offsets.is_contiguous(), "offsets must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    TORCH_CHECK(scores.scalar_type() == torch::kFloat32, "scores must be float32");
    TORCH_CHECK(kv_budget.scalar_type() == torch::kInt32, "kv_budget must be int32");
    TORCH_CHECK(offsets.scalar_type() == torch::kInt32, "offsets must be int32");
    TORCH_CHECK(out.scalar_type() == torch::kInt32, "out must be int32");

    TORCH_CHECK(scores.dim() == 3, "scores must be [3,12,18]");
    TORCH_CHECK(kv_budget.dim() == 2, "kv_budget must be [3,12]");
    TORCH_CHECK(offsets.dim() == 2, "offsets must be [3,12]");

    TORCH_CHECK(scores.size(0) == 3 && scores.size(1) == 12 && scores.size(2) == 18,
                "scores must be [3,12,18]");
    TORCH_CHECK(kv_budget.size(0) == 3 && kv_budget.size(1) == 12,
                "kv_budget must be [3,12]");
    TORCH_CHECK(offsets.size(0) == 3 && offsets.size(1) == 12,
                "offsets must be [3,12]");

    select_kv_row_indices_cuda(scores, kv_budget, offsets, out);
}

// ============================================================
// Single module export
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_key_diversity", &compute_key_diversity, "Compute key diversity CUDA for all 30 blocks");
    m.def("rope_apply_temporal_shift", &rope_apply_temporal_shift, "RoPE temporal shift inplace for all 30 blocks [30,1,L,12,128], supports non-contiguous input");
    m.def("compute_attn_scores", &compute_attn_scores, "compute_attn_scores CUDA");
    m.def("causal_rope_apply", &causal_rope_apply, "Causal RoPE CUDA");
    m.def("pack_qkv", &pack_qkv, "Fixed-shape pack q/kv CUDA");
    m.def("select_kv_row_indices", &select_kv_row_indices, "select_kv_row_indices fixed-shape CUDA");
}
