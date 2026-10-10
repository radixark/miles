"""Exercise the client adapter in the cookbook's own dependency environment."""

import subprocess
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=300, suite="stage-b-cpu", labels=[])


def test_cookbook_client():
    cases = Path(__file__).with_name("tinker_client")
    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--isolated",
            "--index",
            "https://download.pytorch.org/whl/cpu",
            "--with-requirements",
            "examples/multi_lora/requirements.txt",
            "--with",
            "pytest",
            "--with",
            "pytest-asyncio",
            "python",
            "-m",
            "pytest",
            "--confcutdir",
            str(cases),
            *map(str, sorted(cases.glob("*_cases.py"))),
        ],
        check=True,
    )


if __name__ == "__main__":
    test_cookbook_client()
