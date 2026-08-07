from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="causal_rope_apply_cuda",
    ext_modules=[
        CUDAExtension(
            name="causal_rope_apply_cuda",
            sources=[
                "causal_rope_apply.cpp",
                "causal_rope_apply_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
