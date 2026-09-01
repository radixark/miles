import os
import subprocess
import sys
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


_SGLANG_TESTS = (
    Path("test/registered/unit/entrypoints/anthropic/test_utils.py"),
    Path("test/registered/unit/entrypoints/anthropic/test_serving.py"),
)


def test_sglang_anthropic_conversion_contract():
    sglang_source_root = Path(os.environ["SGLANG_SOURCE_ROOT"])
    sglang_repo_root = sglang_source_root.parent
    missing_tests = [str(path) for path in _SGLANG_TESTS if not (sglang_repo_root / path).is_file()]
    assert not missing_tests, f"Missing SGLang Anthropic tests: {missing_tests}"

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            *(str(path) for path in _SGLANG_TESTS),
        ],
        cwd=sglang_repo_root,
        env=env,
        check=True,
    )
