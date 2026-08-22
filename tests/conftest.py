import asyncio
import faulthandler
import os
import signal
import sys
from types import FrameType
from typing import TextIO

import pytest

from tests.fast.fixtures.generation_fixtures import generation_env
from tests.fast.fixtures.rollout_fixtures import rollout_env

_ = rollout_env, generation_env


@pytest.fixture(autouse=True)
def no_env_reporting(monkeypatch):
    """Constructing a worker configures its logger, which in a real process starts a thread that
    shells out to pip and git; tests exercise that reporter directly instead."""
    monkeypatch.setattr("miles.utils.logging_utils.start_env_reporting", lambda args: None)


@pytest.fixture(autouse=True)
def enable_experimental_rollout_refactor():
    os.environ["MILES_EXPERIMENTAL_ROLLOUT_REFACTOR"] = "1"
    yield
    os.environ.pop("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", None)


@pytest.fixture(scope="session")
def ray_local_mode():
    """Session-scoped Ray init. On CI ``RAY_ADDRESS`` points at an existing
    cluster, so we connect without ``num_cpus`` (Ray rejects it when joining).
    Tests that only need pure-Python helpers should not depend on this."""
    import ray

    if not ray.is_initialized():
        kwargs: dict = dict(
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
        )
        if not os.environ.get("RAY_ADDRESS"):
            # address="local" forces a fresh cluster: with no address, ray.init
            # auto-connects to any leaked local cluster (via /tmp/ray), and
            # connecting with num_cpus/num_gpus set is a hard ValueError.
            kwargs["address"] = "local"
            kwargs["num_cpus"] = 32
            # Logical GPU resource so real_ray placement-group tests (engines
            # are mocked via MockSGLangEngine; no real GPU is used) can satisfy
            # their {"GPU": 0.2} bundles on GPU-less CPU CI runners.
            kwargs["num_gpus"] = 8
        ray.init(**kwargs)
    yield
    # Don't shut down — other session-scoped suites may share this cluster.


_STALL_DUMP_SECONDS = float(os.environ.get("MILES_TEST_STALL_DUMP_SECONDS", "60"))
_STALL_DUMP_ENABLED = hasattr(signal, "SIGALRM") and bool(
    os.environ.get("CI") or os.environ.get("MILES_TEST_STALL_DUMP_SECONDS")
)
_stall_config: pytest.Config | None = None
_stall_nodeid: str = ""


def pytest_configure(config: pytest.Config) -> None:
    global _stall_config
    if not _STALL_DUMP_ENABLED:
        return
    _stall_config = config
    signal.signal(signal.SIGALRM, _dump_stall)
    faulthandler.register(signal.SIGTERM, all_threads=True, chain=True)


def pytest_runtest_logstart(nodeid: str, location: tuple[str, int | None, str]) -> None:
    global _stall_nodeid
    if not _STALL_DUMP_ENABLED:
        return
    _stall_nodeid = nodeid
    signal.setitimer(signal.ITIMER_REAL, _STALL_DUMP_SECONDS)


def pytest_runtest_logfinish(nodeid: str, location: tuple[str, int | None, str]) -> None:
    if not _STALL_DUMP_ENABLED:
        return
    signal.setitimer(signal.ITIMER_REAL, 0)


def _dump_stall(signum: int, frame: FrameType | None) -> None:
    capture_manager = _stall_config.pluginmanager.getplugin("capturemanager") if _stall_config else None
    if capture_manager is not None:
        capture_manager.suspend_global_capture(in_=False)
    try:
        out = sys.stderr
        print(f"\n===== MILES STALL DUMP: {_stall_nodeid} exceeded {_STALL_DUMP_SECONDS}s =====", file=out, flush=True)
        faulthandler.dump_traceback(file=out, all_threads=True)
        _dump_asyncio_tasks(out)
        print("===== MILES STALL DUMP END =====", file=out, flush=True)
    finally:
        if capture_manager is not None:
            capture_manager.resume_global_capture()
    signal.setitimer(signal.ITIMER_REAL, _STALL_DUMP_SECONDS)


def _dump_asyncio_tasks(out: TextIO) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        print("no running event loop in the main thread", file=out, flush=True)
        return
    for task in asyncio.all_tasks(loop):
        print(f"----- {task!r}", file=out)
        for frame in task.get_stack(limit=30):
            print(f"        {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}", file=out)
    print("", file=out, flush=True)
