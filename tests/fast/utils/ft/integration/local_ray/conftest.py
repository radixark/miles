from __future__ import annotations

import logging
import time
from collections.abc import Callable, Generator
from typing import Any

import pytest
import ray

from miles.utils.ft.models import ControllerStatus
from miles.utils.ft.platform.controller_actor import FtControllerActor
from miles.utils.ft.platform.controller_factory import FtControllerConfig
from miles.utils.ft.protocols.platform import ft_controller_actor_name

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.local_ray,
]


@pytest.fixture(scope="module")
def local_ray() -> Generator[None, None, None]:
    if ray.is_initialized():
        ray.shutdown()
    ray.init(address="local", num_cpus=4, num_gpus=0, include_dashboard=False)
    yield
    ray.shutdown()


def _kill_named_actor(name: str) -> None:
    try:
        handle = ray.get_actor(name)
        ray.kill(handle, no_restart=True)
    except ValueError:
        pass
    except Exception:
        logger.warning("Failed to kill actor %s", name, exc_info=True)


@pytest.fixture(autouse=True)
def _cleanup_default_controller(local_ray: None) -> Generator[None, None, None]:
    _kill_named_actor(ft_controller_actor_name(""))
    yield
    _kill_named_actor(ft_controller_actor_name(""))


@pytest.fixture
def controller_actor(
    local_ray: None,
) -> Generator[ray.actor.ActorHandle, None, None]:
    actor_name = ft_controller_actor_name("")
    handle = FtControllerActor.options(name=actor_name).remote(
        config=FtControllerConfig(platform="stub", tick_interval=0.05),
    )
    yield handle
    try:
        ray.get(handle.shutdown.remote(), timeout=10)
    except Exception:
        pass
    _kill_named_actor(actor_name)


@pytest.fixture
def make_controller_actor(
    local_ray: None,
) -> Generator[Callable[..., ray.actor.ActorHandle], None, None]:
    created_actors: list[tuple[ray.actor.ActorHandle, str]] = []

    def _factory(
        ft_id: str = "",
        tick_interval: float = 0.05,
        **overrides: Any,
    ) -> ray.actor.ActorHandle:
        actor_name = ft_controller_actor_name(ft_id)
        _kill_named_actor(actor_name)
        handle = FtControllerActor.options(name=actor_name).remote(
            config=FtControllerConfig(
                platform="stub",
                tick_interval=tick_interval,
                ft_id=ft_id,
            ),
            **overrides,
        )
        created_actors.append((handle, actor_name))
        return handle

    yield _factory

    for handle, name in created_actors:
        try:
            ray.get(handle.shutdown.remote(), timeout=5)
        except Exception:
            pass
        _kill_named_actor(name)


def get_status(handle: ray.actor.ActorHandle, timeout: float = 5) -> ControllerStatus:
    return ray.get(handle.get_status.remote(), timeout=timeout)


@pytest.fixture
def running_controller(
    local_ray: None,
) -> Generator[tuple[ray.actor.ActorHandle, str], None, None]:
    """Controller actor that has already submitted a training run via StubTrainingJob.

    Yields (handle, run_id) where run_id is the auto-generated ID from StubTrainingJob.
    """
    actor_name = ft_controller_actor_name("")
    handle = FtControllerActor.options(name=actor_name).remote(
        config=FtControllerConfig(platform="stub", tick_interval=0.05),
    )
    handle.submit_and_run.remote()
    time.sleep(0.3)
    status = ray.get(handle.get_status.remote(), timeout=5)
    assert status.active_run_id is not None, "StubTrainingJob did not set run_id"
    yield handle, status.active_run_id
    try:
        ray.get(handle.shutdown.remote(), timeout=10)
    except Exception:
        pass
    _kill_named_actor(actor_name)
