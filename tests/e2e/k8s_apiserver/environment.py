# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from miles.utils.misc import exec_command

logger = logging.getLogger(__name__)

KEEP_ENV_VAR = "MILES_K8S_KEEP"
REQUIRE_ENV_VAR = "MILES_K8S_REQUIRE"
KUBECONFIG_ENV_VAR = "MILES_K8S_KUBECONFIG"


def new_run_id() -> str:
    return f"miles-k8s-{uuid.uuid4().hex[:8]}"


def keep_environment() -> bool:
    return _is_truthy(os.environ.get(KEEP_ENV_VAR))


def existing_kubeconfig() -> Path | None:
    configured = os.environ.get(KUBECONFIG_ENV_VAR)
    return Path(configured).expanduser() if configured else None


def require_docker() -> None:
    reason = _docker_failure_reason()
    if reason is None:
        return

    message = f"the Kubernetes suites need a working Docker daemon: {reason}"
    if _docker_is_mandatory():
        raise RuntimeError(message)
    pytest.skip(message)


def _docker_is_mandatory() -> bool:
    override = os.environ.get(REQUIRE_ENV_VAR)
    if override is not None:
        return _is_truthy(override)
    return _is_truthy(os.environ.get("CI"))


def _docker_failure_reason() -> str | None:
    if shutil.which("docker") is None:
        return "docker is not on PATH"
    try:
        exec_command("docker info", capture_output=True)
    except subprocess.CalledProcessError as error:
        return f"`docker info` exited with {error.returncode}"
    return None


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.lower() not in ("", "0", "false", "no")
