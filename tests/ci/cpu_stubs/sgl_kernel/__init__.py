"""Stub sgl_kernel for CPU-only CI.

The real sgl_kernel ships CUDA kernels and cannot be installed on an
ubuntu-latest CPU runner. miles' import chain loads sglang modules that
unconditionally `from sgl_kernel... import ...` at module load time (gated
only on NPU/XPU/MPS, not CPU). Any attribute access on this stub returns a
MagicMock so imports succeed; if stub symbols are actually *called* at test
runtime, the MagicMock will return further MagicMocks — and if a test
depends on real kernel behavior it will fail loudly, which is correct.
"""

from unittest.mock import MagicMock


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return MagicMock(name=f"sgl_kernel.{name}")
