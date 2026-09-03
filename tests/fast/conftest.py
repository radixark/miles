import os
import sys

import pytest
from tests.fast.fixtures.timeouts import TIMEOUT_SCALE_ENV

from miles.utils.http_utils import PORT_RANGE_ENV

_PORT_RANGE_LOW = 20000
_PORTS_PER_WORKER = 195
_TIMEOUT_SCALE_UNDER_XDIST = "5"


@pytest.fixture(scope="session", autouse=True)
def isolate_xdist_worker() -> None:
    """Under pytest-xdist every worker draws ports from its own range below the kernel's ephemeral range and waits longer for subprocesses."""
    if (worker := os.environ.get("PYTEST_XDIST_WORKER")) is None:
        return
    low = _PORT_RANGE_LOW + int(worker.removeprefix("gw")) * _PORTS_PER_WORKER
    os.environ[PORT_RANGE_ENV] = f"{low}:{low + _PORTS_PER_WORKER}"
    os.environ[TIMEOUT_SCALE_ENV] = _TIMEOUT_SCALE_UNDER_XDIST


@pytest.fixture(autouse=True)
def plain_interpreter_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Launch commands copy the parent interpreter flags; the test runner's own flags must not leak in."""
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, "-m", "pytest"])
