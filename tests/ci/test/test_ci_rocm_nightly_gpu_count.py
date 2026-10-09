"""A file's ROCm nightly suite asks for as many GPUs as its CUDA stage.

The external ROCm nightly runs the ``nightly-stage-c-<N>-gpu-*`` suites, where
``<N>`` is the GPU count the test needs, as in a CUDA stage name. When a file
registered for both moves its CUDA registration to a stage with a different GPU
count, ``<N>`` moves with it in the same commit, so the nightly gives the test
the GPUs its CUDA stage does. A new test needs no ROCm registration, and the
unprefixed ROCm PR stage is out of scope.
"""

import re
from pathlib import Path

from tests.ci.ci_register import HWBackend, collect_tests, discover_ci_files, register_cpu_ci
from tests.ci.hardware import CUDA_STAGES

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

REPO_ROOT = Path(__file__).resolve().parents[3]

_ROCM_NIGHTLY = re.compile(r"nightly-stage-c-(\d+)-gpu-")


def test_rocm_nightly_gpu_count_follows_the_cuda_stage(monkeypatch):
    # discover_ci_files() globs repo-relative paths; pin cwd to the checkout.
    monkeypatch.chdir(REPO_ROOT)
    registrations = collect_tests(discover_ci_files())
    cuda_gpus = {r.filename: CUDA_STAGES[r.suite].num_gpus for r in registrations if r.backend is HWBackend.CUDA}
    mismatched = sorted(
        (r.filename, r.suite, f"CUDA stage has {cuda_gpus[r.filename]} GPUs")
        for r in registrations
        if r.backend is HWBackend.ROCM
        and (match := _ROCM_NIGHTLY.match(r.suite))
        and r.filename in cuda_gpus
        and int(match.group(1)) != cuda_gpus[r.filename]
    )
    assert mismatched == [], (
        "nightly-stage-c-<N>-gpu-* must use the GPU count of the file's CUDA stage; "
        f"update <N> together with the CUDA stage: {mismatched}"
    )
