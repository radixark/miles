"""The miles-ext-kernels submodule's kernel tests (3rdparty/miles-ext-kernels/tests/<package>) on Blackwell or Rubin.

The submodule's own CI has no GPU, so its CUDA tests run here, one separate pytest process per kernel package against
this checkout's submodule. Every collected test must run: on the runners this file is registered for, a skip means a
missing dependency (torch, CUDA, fla) or an unsupported GPU, and a missing submodule checkout fails too; either would
hide the kernel coverage this file exists for.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1700, suite="stage-c-4-gpu-b200", labels=["precision"], hardware=["blackwell"], num_gpus=1)

import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SUBMODULE = _REPO_ROOT / "3rdparty" / "miles-ext-kernels"
# The kernel packages whose tests/<package> directory runs here, in the submodule's package order.
_PACKAGES = ["gdn", "kda", "conv1d", "gated_rmsnorm", "rmsnorm"]
_OUTCOME = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)\b")


def _outcomes(stdout: str) -> Counter:
    """Outcome counts from pytest's final summary line, e.g. ``12 passed, 1 skipped in 3.40s``."""
    summary = next((line for line in reversed(stdout.splitlines()) if _OUTCOME.search(line)), "")
    return Counter({outcome.removesuffix("s"): int(count) for count, outcome in _OUTCOME.findall(summary)})


@pytest.mark.parametrize("package", _PACKAGES)
def test_the_submodule_kernel_tests_pass(package: str):
    tests = _SUBMODULE / "tests" / package
    assert (tests / "__init__.py").is_file(), (
        f"the miles-ext-kernels submodule has no tests/{package} package, or is not checked out: "
        "run 'git submodule update --init 3rdparty/miles-ext-kernels'"
    )
    # This checkout's submodule comes first on the path, and the submodule's own pyproject.toml (no pytest options)
    # replaces miles' pytest configuration, as in the submodule's CI.
    pythonpath = os.pathsep.join(filter(None, [str(_SUBMODULE), os.environ.get("PYTHONPATH")]))
    command = [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider"]
    command += ["-c", str(_SUBMODULE / "pyproject.toml"), str(tests)]
    result = subprocess.run(
        command, cwd=_SUBMODULE, env={**os.environ, "PYTHONPATH": pythonpath}, capture_output=True, text=True
    )
    outcomes = _outcomes(result.stdout)
    report = f"{dict(outcomes)}, exit code {result.returncode}\n{result.stdout[-20000:]}\n{result.stderr[-5000:]}"

    assert result.returncode == 0, report
    assert outcomes["passed"] >= 1, report
    assert outcomes["failed"] == outcomes["error"] == 0, report
    assert outcomes["skipped"] == 0, report


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
