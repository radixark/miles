"""Exercise the client adapter in the cookbook's own dependency environment."""

import subprocess
import sys
import tempfile
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=300, suite="stage-b-cpu", labels=["multi-lora"])


def test_cookbook_client():
    cases = Path(__file__).with_name("tinker_client")
    with tempfile.TemporaryDirectory(prefix="tinker-client-") as environment:
        subprocess.run(["uv", "venv", "--python", sys.executable, environment], check=True)
        python = str(Path(environment) / "bin/python")
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "--torch-backend",
                "cpu",
                "-r",
                "examples/multi_lora/requirements.txt",
                "./examples/multi_lora/tinker_agent[harbor]",
                "pytest",
                "pytest-asyncio",
            ],
            check=True,
        )
        subprocess.run(
            [
                python,
                "-I",
                "-m",
                "pytest",
                "--import-mode=importlib",
                "--confcutdir",
                str(cases),
                *map(str, sorted(cases.glob("*_cases.py"))),
            ],
            check=True,
        )


if __name__ == "__main__":
    test_cookbook_client()
