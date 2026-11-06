"""The miles-ext-kernels submodule's kernel tests (3rdparty/miles-ext-kernels/tests/cake/<op>) on Blackwell or Rubin.

The submodule's own CI has no GPU, so its CUDA tests run here, one separate pytest process per op (and one for the
shared runtime, ``_runtime``) against this checkout's submodule. This job runs the tests of one GPU (``-m "not
multi_gpu"``); ``test_miles_ext_kernels_multi_gpu.py`` runs the ``multi_gpu`` ones. Every collected test must run, but
for the hardware limits ``miles_ext_kernels_skips.py`` lists for its op (a test that needs more GPUs than the runner
has, a case a GPU's kernels do not serve): on the runners this file is registered for, any other skip means a missing
dependency (torch, CUDA, fla) or an unsupported GPU, and a missing submodule checkout fails too; either would hide the
kernel coverage this file exists for.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from miles_ext_kernels_skips import is_allowed, skips  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SUBMODULE = _REPO_ROOT / "3rdparty" / "miles-ext-kernels"
_OUTCOME = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)\b")


def _env() -> dict[str, str]:
    """This checkout's submodule first on the path."""
    pythonpath = os.pathsep.join(filter(None, [str(_SUBMODULE), os.environ.get("PYTHONPATH")]))
    return {**os.environ, "PYTHONPATH": pythonpath}


def _ops() -> list[str]:
    """``miles_ext_kernels.cake.OPS`` of this checkout's submodule, in layer order: read in a child process (the module
    is torch-free), so that an installed copy of the package elsewhere cannot answer. Empty when the submodule is not
    checked out, which the ``_runtime`` case then reports."""
    result = subprocess.run(
        [sys.executable, "-c", "from miles_ext_kernels.cake import OPS; print(*OPS)"],
        cwd=_SUBMODULE if _SUBMODULE.is_dir() else _REPO_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
    )
    return result.stdout.split() if result.returncode == 0 else []


# The shared runtime, then every op, whose tests/cake/<package> directory runs here.
_PACKAGES = ["_runtime", *_ops()]


def _outcomes(stdout: str) -> Counter:
    """Outcome counts from pytest's final summary line, e.g. ``12 passed, 1 skipped in 3.40s``."""
    summary = next((line for line in reversed(stdout.splitlines()) if _OUTCOME.search(line)), "")
    return Counter({outcome.removesuffix("s"): int(count) for count, outcome in _OUTCOME.findall(summary)})


def run_submodule_tests(tests: Path, marker: str) -> None:
    """Run the submodule's tests under ``tests`` that ``-m marker`` selects; every one must pass, and only a listed
    hardware limit may skip."""
    assert (tests / "__init__.py").is_file(), (
        f"the miles-ext-kernels submodule has no {tests.relative_to(_SUBMODULE)} package, or is not checked out: "
        "run 'git submodule update --init 3rdparty/miles-ext-kernels'"
    )
    # The submodule's own pyproject.toml (its pytest options and markers) replaces miles' pytest configuration, as in
    # the submodule's CI.
    command = [sys.executable, "-m", "pytest", "-q", "-rs", "--color=no", "-p", "no:cacheprovider", "-m", marker]
    command += ["-c", str(_SUBMODULE / "pyproject.toml"), str(tests)]
    result = subprocess.run(command, cwd=_SUBMODULE, env=_env(), capture_output=True, text=True)
    outcomes = _outcomes(result.stdout)
    skipped = skips(result.stdout)
    report = f"{dict(outcomes)}, exit code {result.returncode}\n{result.stdout[-20000:]}\n{result.stderr[-5000:]}"

    assert result.returncode == 0, report
    assert outcomes["passed"] >= 1, report
    assert outcomes["failed"] == outcomes["error"] == 0, report
    # -rs lists every skip with its reason
    assert sum(count for count, _, _ in skipped) == outcomes["skipped"], report
    unlisted = [f"{path}: {reason}" for _, path, reason in skipped if not is_allowed(path, reason)]
    assert not unlisted, f"skips that miles_ext_kernels_skips.py does not list: {unlisted}\n{report}"


@pytest.mark.parametrize("package", _PACKAGES)
def test_the_submodule_kernel_tests_pass(package: str):
    run_submodule_tests(_SUBMODULE / "tests" / "cake" / package, "not multi_gpu")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
