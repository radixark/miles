import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Archs: the visible GPUs, else TORCH_CUDA_ARCH_LIST ("9.0;10.0" or "9.0 10.0+PTX"), else a default list.
arch_list = sorted(
    {f"{major}.{minor}" for major, minor in map(torch.cuda.get_device_capability, range(torch.cuda.device_count()))}
)
if not arch_list:
    env_arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    arch_list = [a.strip().replace("+PTX", "") for a in env_arch.replace(";", " ").split() if a.strip()]
if not arch_list:
    arch_list = ["8.0", "8.6", "8.9", "9.0"]

setup(
    name="causal_conv1d_bwd_cuda",
    ext_modules=[
        CUDAExtension(
            name="causal_conv1d_bwd_cuda",
            sources=["causal_conv1d_bwd.cu"],
            extra_compile_args={
                # No -std flag: torch adds the C++ standard its headers need.
                "cxx": ["-O3"],
                # causal-conv1d's own flags, so the numerics match the kernel this is adapted from.
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                ]
                + [f"-gencode=arch=compute_{a.replace('.', '')},code=sm_{a.replace('.', '')}" for a in arch_list],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
