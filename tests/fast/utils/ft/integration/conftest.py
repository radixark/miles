from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator

import pytest
import ray

from miles.utils.ft.controller.types import ControllerStatus

logger = logging.getLogger(__name__)

_TIMEOUT_SCALE = float(os.environ.get("FT_TEST_TIMEOUT_SCALE", "1.0"))
FAST_TIMEOUT = 30.0 * _TIMEOUT_SCALE
RECOVERY_TIMEOUT = 60.0 * _TIMEOUT_SCALE
LONG_RECOVERY_TIMEOUT = 120.0 * _TIMEOUT_SCALE


def _init_local_ray(*, include_dashboard: bool) -> tuple[ray.runtime_env.RuntimeContext, str | None]:
    if ray.is_initialized():
        ray.shutdown()
    kwargs: dict[str, object] = dict(
        address="local",
        num_cpus=32,
        num_gpus=0,
        include_dashboard=include_dashboard,
    )
    if include_dashboard:
        kwargs.update(dashboard_host="127.0.0.1", dashboard_port=0)
    ctx = ray.init(**kwargs)
    dashboard_url = f"http://{ctx.dashboard_url}" if include_dashboard and ctx.dashboard_url else None
    return ctx, dashboard_url


@pytest.fixture(scope="module")
def local_ray() -> Generator[None, None, None]:
    _init_local_ray(include_dashboard=False)
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def local_ray_with_dashboard() -> Generator[str, None, None]:
    _, url = _init_local_ray(include_dashboard=True)
    assert url is not None
    yield url
    ray.shutdown()


def _kill_named_actor(name: str) -> None:
    try:
        handle = ray.get_actor(name)
        ray.kill(handle, no_restart=True)
    except ValueError:
        pass
    except Exception:
        logger.warning("Failed to kill actor %s", name, exc_info=True)


def get_status(handle: ray.actor.ActorHandle, timeout: float = 5) -> ControllerStatus:
    return ray.get(handle.get_status.remote(), timeout=timeout)


def poll_for_run_id(
    handle: ray.actor.ActorHandle,
    timeout: float = 10.0,
    interval: float = 0.2,
) -> str:
    """Poll get_status until active_run_id is set, return it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = get_status(handle)
        if status.active_run_id is not None:
            return status.active_run_id
        time.sleep(interval)
    raise TimeoutError("active_run_id not set within timeout")
