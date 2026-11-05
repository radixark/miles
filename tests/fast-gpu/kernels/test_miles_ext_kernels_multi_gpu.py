"""The miles-ext-kernels submodule's ``multi_gpu`` kernel tests (3rdparty/miles-ext-kernels/tests/cake) on four GPUs.

The tests of more than one GPU (context parallelism across ranks, a second device) of every op, in one pytest process
with ``-m multi_gpu``; ``test_miles_ext_kernels.py`` runs the others on one GPU. As there, every collected test must
run and pass, but for the hardware limits ``miles_ext_kernels_skips.py`` lists for its op (a test that needs more GPUs
than these four, or two GPUs of one part).
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1800, suite="stage-c-4-gpu-b200", labels=["precision"], hardware=["blackwell"], num_gpus=4)

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_miles_ext_kernels import _SUBMODULE, run_submodule_tests  # noqa: E402


def test_the_submodule_multi_gpu_kernel_tests_pass():
    run_submodule_tests(_SUBMODULE / "tests" / "cake", "multi_gpu")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
