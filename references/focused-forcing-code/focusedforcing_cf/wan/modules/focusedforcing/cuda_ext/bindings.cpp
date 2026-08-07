#include <pybind11/pybind11.h>
#include <torch/extension.h>

// CUDA entry points
void compute_key_diversity_cuda(
    torch::Tensor k,
    torch::Tensor mean_pos,
    torch::Tensor out_div
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
    int start_frame,
    bool do_q,
    bool do_k
);

void select_kv_row_indices_cuda(
    torch::Tensor scores,
    torch::Tensor kv_budget,
    torch::Tensor offsets,
    torch::Tensor out,
    bool update
);

void pack_qkv_cuda(
    torch::Tensor roped_query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor kv_rows,
    torch::Tensor q_out,
    torch::Tensor k_out,
    torch::Tensor v_out
);

void concat_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor out
);

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA kernels for focusedforcing";

    m.def(
        "compute_key_diversity",
        &compute_key_diversity_cuda,
        py::arg("k"),
        py::arg("mean_pos"),
        py::arg("out_div"),
        "Compute key diversity"
    );

    m.def(
        "compute_attn_scores",
        &compute_attn_scores_cuda,
        py::arg("query"),
        py::arg("key"),
        py::arg("pooled_q"),
        py::arg("pooled_k"),
        py::arg("gemm_out"),
        py::arg("out"),
        "Compute attention scores"
    );

    m.def(
        "causal_rope_apply",
        &causal_rope_apply_cuda,
        py::arg("q"),
        py::arg("k"),
        py::arg("grid_sizes"),
        py::arg("freqs0_re"),
        py::arg("freqs0_im"),
        py::arg("freqs1_re"),
        py::arg("freqs1_im"),
        py::arg("freqs2_re"),
        py::arg("freqs2_im"),
        py::arg("out_q"),
        py::arg("out_k"),
        py::arg("start_frame"),
        py::arg("do_q"),
        py::arg("do_k"),
        "Apply causal RoPE to q and/or k"
    );

    m.def(
        "select_kv_row_indices",
        &select_kv_row_indices_cuda,
        py::arg("scores"),
        py::arg("kv_budget"),
        py::arg("offsets"),
        py::arg("out"),
        py::arg("update"),
        "Select KV row indices"
    );

    m.def(
        "pack_qkv",
        &pack_qkv_cuda,
        py::arg("roped_query"),
        py::arg("key"),
        py::arg("value"),
        py::arg("kv_rows"),
        py::arg("q_out"),
        py::arg("k_out"),
        py::arg("v_out"),
        "Pack q/k/v and build flash-attn meta"
    );

    m.def(
        "concat",
        &concat_cuda,
        py::arg("a"),
        py::arg("b"),
        py::arg("c"),
        py::arg("out"),
        "Concatenate three [1, L, 12, 128] tensors along dim=1"
    );
}
