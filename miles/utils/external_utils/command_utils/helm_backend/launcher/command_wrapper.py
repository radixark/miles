from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.workers.worker_provider.kubernetes.helm.env import INSTANCE_LABEL

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class Helm:
    @staticmethod
    def run_raw(*arguments: str) -> subprocess.CompletedProcess[str]:
        return run_process(["helm", *arguments], capture_output=True, check=False)

    @staticmethod
    def upgrade(
        *,
        release: str,
        namespace: str,
        chart: str | Path,
        values_files: list[str | Path],
    ) -> None:
        _run(Helm.upgrade_command(release, namespace, chart, values_files), capture_output=False)

    @staticmethod
    def build_dependencies(chart: str | Path) -> None:
        if all((Path(chart) / "charts" / name).exists() for name in _locked_dependency_names(chart)):
            return
        _run(["helm", "dependency", "build", str(chart)], capture_output=False)

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
    def get_json(
        kind: str,
        *,
        return_type: type[_ModelT],
        name: str | None = None,
        namespace: str,
        selector: str | None = None,
        field_selector: str | None = None,
    ) -> _ModelT | None:
        command = ["get", kind]
        if name is not None:
            command.append(name)
        command += ["--namespace", namespace, "--output", "json", "--ignore-not-found"]
        if selector is not None:
            command += ["--selector", selector]
        if field_selector is not None:
            command += ["--field-selector", field_selector]
        result = Kubectl._run(command)
        if result.returncode != 0:
            raise RuntimeError(f"kubectl get {kind} failed with code {result.returncode}: {result.stderr.strip()}")
        if not result.stdout.strip():
            return None
        return return_type.model_validate_json(result.stdout)

    @staticmethod
    def logs_command(
        *,
        namespace: str,
        target: str,
        container: str | None = None,
        follow: bool = False,
        previous: bool = False,
        tail: int | None = None,
        since_time: str | None = None,
    ) -> list[str]:
        command = ["kubectl", "logs", target, "--namespace", namespace, "--timestamps"]
        command += ["-c", container] if container is not None else ["--all-containers"]
        if follow:
            command.append("--follow")
        if previous:
            command.append("--previous")
        if tail is not None:
            command += ["--tail", str(tail)]
        if since_time is not None:
            command += ["--since-time", since_time]
        return command

    @staticmethod
    def release_selector(release: str) -> str:
        return f"{INSTANCE_LABEL}={release}"

    @staticmethod
    def _run(
        arguments: list[str], *, input: str | None = None, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return run_process(["kubectl", *arguments], capture_output=True, check=check, input=input)


def _run(command: list[str], capture_output: bool) -> subprocess.CompletedProcess[str]:
    return run_process(command, capture_output=capture_output, check=True)


def _locked_dependency_names(chart: str | Path) -> list[str]:
    lock = Path(chart) / "Chart.lock"
    if not lock.exists():
        return []
    return [entry["name"] for entry in (yaml.safe_load(lock.read_text()) or {}).get("dependencies", [])]
