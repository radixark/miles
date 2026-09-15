from __future__ import annotations

import subprocess
from pathlib import Path

from miles.utils.external_utils.command_utils.common import run_process


class Helm:
    @staticmethod
    def run_raw(*arguments: str) -> subprocess.CompletedProcess[str]:
        return run_process(["helm", *arguments], capture_output=True, check=False)

    @staticmethod
    def upgrade_command(release: str, namespace: str, chart: str | Path, values_files: list[str | Path]) -> list[str]:
        command = ["helm", "upgrade", "--install", release, str(chart), "--namespace", namespace]
        for values_file in values_files:
            command += ["--values", str(values_file)]
        return command


class Kubectl:
    @staticmethod
    def run_raw(*arguments: str) -> subprocess.CompletedProcess[str]:
        return Kubectl._run(list(arguments))

    @staticmethod
    def _run(
        arguments: list[str], *, input: str | None = None, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return run_process(["kubectl", *arguments], capture_output=True, check=check, input=input)
