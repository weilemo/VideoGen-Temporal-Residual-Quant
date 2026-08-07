from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rope_apply_temporal_shift_cuda",
    ext_modules=[
        CUDAExtension(
            name="rope_apply_temporal_shift_cuda",
            sources=[
                "rope_apply_temporal_shift.cpp",
                "rope_apply_temporal_shift_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
