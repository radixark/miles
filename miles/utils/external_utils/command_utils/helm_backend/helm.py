from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

CHART_NAME = "miles-run"
CI_LABEL = "miles.radixark.io/ci-run"


def chart_dir(repo_base_dir: str | Path) -> Path:
    return Path(repo_base_dir) / "charts" / CHART_NAME


def upgrade_command(
    *,
    release: str,
    namespace: str,
    chart: str | Path,
    values_files: list[str | Path],
    dry_run: bool = False,
    ci_run: bool = False,
) -> list[str]:
    command = ["helm", "upgrade", "--install", release, str(chart), "--namespace", namespace]
    if ci_run:
        command += ["--labels", f"{CI_LABEL}=true"]
    for values_file in values_files:
        command += ["--values", str(values_file)]
    if dry_run:
        command += ["--dry-run"]
    return command


def dependency_build_command(chart: str | Path) -> list[str]:
    return ["helm", "dependency", "build", str(chart)]


def list_ci_releases_command(namespace: str) -> list[str]:
    return ["helm", "list", "--namespace", namespace, "--selector", f"{CI_LABEL}=true", "--output", "json"]


def uninstall_command(release: str, namespace: str) -> list[str]:
    return ["helm", "uninstall", release, "--namespace", namespace]


def parse_release_names(helm_list_output: str) -> list[str]:
    return [release["name"] for release in json.loads(helm_list_output or "[]")]


def run(command: list[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    print(f"EXEC: {shlex.join(command)}", flush=True)
    return subprocess.run(command, check=True, capture_output=capture_output, text=capture_output)


def release_exists(release: str, namespace: str) -> bool:
    listed = subprocess.run(["helm", "status", release, "--namespace", namespace], capture_output=True, text=True)
    if listed.returncode == 0:
        return True
    if "not found" in (listed.stderr + listed.stdout).lower():
        return False
    raise RuntimeError(
        f"cannot tell whether release {release} exists: {listed.stderr.strip() or listed.stdout.strip()}"
    )


def uninstall_ci_releases(namespace: str) -> list[str]:
    listed = run(list_ci_releases_command(namespace), capture_output=True)
    releases = parse_release_names(listed.stdout)
    for release in releases:
        run(uninstall_command(release, namespace))
    return releases
