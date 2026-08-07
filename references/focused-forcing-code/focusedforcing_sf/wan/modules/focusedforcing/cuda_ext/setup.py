from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="focusedforcing_cuda",
    ext_modules=[
        CUDAExtension(
            name="focusedforcing_cuda",
            sources=[
                "bindings.cpp",
                "compute_key_diversity_kernel.cu",
                "rope_apply_temporal_shift_kernel.cu",
                "compute_attn_scores_kernel.cu",
                "causal_rope_apply_kernel.cu",
                "pack_qkv_kernel.cu",
                "select_kv_row_indices_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
