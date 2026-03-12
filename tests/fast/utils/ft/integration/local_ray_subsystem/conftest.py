from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any

import pytest
import ray
from tests.fast.utils.ft.integration.conftest import _kill_named_actor, poll_for_run_id

from miles.utils.ft.adapters.config import FtControllerConfig
from miles.utils.ft.adapters.impl.ray.controller_actor import FtControllerActor
from miles.utils.ft.adapters.types import ft_controller_actor_name
from miles.utils.ft.factories.controller.from_config import build_ft_controller

pytestmark = [
    pytest.mark.local_ray,
]


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
        builder=build_ft_controller,
        config=FtControllerConfig(platform="stub", tick_interval=0.05, rollout_num_cells=0),
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
            builder=build_ft_controller,
            config=FtControllerConfig(
                platform="stub",
                tick_interval=tick_interval,
                ft_id=ft_id,
                rollout_num_cells=0,
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


@pytest.fixture
def running_controller(
    local_ray: None,
) -> Generator[tuple[ray.actor.ActorHandle, str], None, None]:
    """Controller actor that has already submitted a training run via StubMainJob.

    Yields (handle, run_id) where run_id is the auto-generated ID from StubMainJob.
    """
    actor_name = ft_controller_actor_name("")
    handle = FtControllerActor.options(name=actor_name).remote(
        builder=build_ft_controller,
        config=FtControllerConfig(platform="stub", tick_interval=0.05, rollout_num_cells=0),
    )
    handle.submit_and_run.remote()
    run_id = poll_for_run_id(handle)
    yield handle, run_id
    try:
        ray.get(handle.shutdown.remote(), timeout=10)
    except Exception:
        pass
    _kill_named_actor(actor_name)
