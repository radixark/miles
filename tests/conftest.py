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
def clear_legacy_rollout_gate(monkeypatch):
    # an ambient value changes which arguments the parser registers
    monkeypatch.delenv("MILES_USE_LEGACY_ROLLOUT_V1", raising=False)


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
_STALL_DUMP_ENABLED = hasattr(signal, "SIGALRM") and "MILES_TEST_STALL_DUMP_SECONDS" in os.environ


class _StallDumper:
    def __init__(self, *, config: pytest.Config, interval_seconds: float) -> None:
        self._config = config
        self._interval_seconds = interval_seconds
        self._nodeid: str = ""
        self._uncaptured_stderr_fd: int = _duplicate_uncaptured_stderr(config)

        signal.signal(signal.SIGALRM, self._dump)
        faulthandler.register(signal.SIGTERM, file=self._uncaptured_stderr_fd, all_threads=True, chain=True)

    def arm(self, *, nodeid: str) -> None:
        self._nodeid = nodeid
        signal.setitimer(signal.ITIMER_REAL, self._interval_seconds)

    def disarm(self) -> None:
        signal.setitimer(signal.ITIMER_REAL, 0)

    def _dump(self, signum: int, frame: FrameType | None) -> None:
        capture_manager = self._config.pluginmanager.getplugin("capturemanager")
        if capture_manager is not None:
            capture_manager.suspend_global_capture(in_=False)
        try:
            out = sys.stderr
            print(
                f"\n===== MILES STALL DUMP: {self._nodeid} exceeded {self._interval_seconds}s =====",
                file=out,
                flush=True,
            )
            faulthandler.dump_traceback(file=out, all_threads=True)
            self._dump_asyncio_tasks(out)
            print("===== MILES STALL DUMP END =====", file=out, flush=True)
        finally:
            if capture_manager is not None:
                capture_manager.resume_global_capture()
        signal.setitimer(signal.ITIMER_REAL, self._interval_seconds)

    def _dump_asyncio_tasks(self, out: TextIO) -> None:
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


def _duplicate_uncaptured_stderr(config: pytest.Config) -> int:
    if (capture_manager := config.pluginmanager.getplugin("capturemanager")) is None:
        return os.dup(2)

    capture_manager.suspend_global_capture(in_=False)
    try:
        return os.dup(2)
    finally:
        capture_manager.resume_global_capture()


_stall_dumper: _StallDumper | None = None


def pytest_configure(config: pytest.Config) -> None:
    global _stall_dumper
    if not _STALL_DUMP_ENABLED:
        return
    _stall_dumper = _StallDumper(config=config, interval_seconds=_STALL_DUMP_SECONDS)


def pytest_runtest_logstart(nodeid: str, location: tuple[str, int | None, str]) -> None:
    if _stall_dumper is not None:
        _stall_dumper.arm(nodeid=nodeid)


def pytest_runtest_logfinish(nodeid: str, location: tuple[str, int | None, str]) -> None:
    if _stall_dumper is not None:
        _stall_dumper.disarm()
