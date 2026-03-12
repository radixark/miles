"""Local Ray: Actor lifecycle — naming, discovery, shutdown, restart."""

from __future__ import annotations

import time

import pytest
import ray
from tests.fast.utils.ft.integration.conftest import _kill_named_actor, poll_for_run_id

from miles.utils.ft.adapters.config import FtControllerConfig
from miles.utils.ft.adapters.impl.ray.controller_actor import FtControllerActor
from miles.utils.ft.adapters.types import ft_controller_actor_name
from miles.utils.ft.controller.types import ControllerMode
from miles.utils.ft.factories.controller.from_config import build_ft_controller

pytestmark = [
    pytest.mark.local_ray,
]


class TestNamedActorCreation:
    def test_create_and_discover_via_get_actor(self, local_ray: None) -> None:
        name = ft_controller_actor_name("")
        _handle = FtControllerActor.options(name=name).remote(
            builder=build_ft_controller,
            config=FtControllerConfig(platform="stub", tick_interval=1.0, rollout_num_cells=0),
        )
        try:
            discovered = ray.get_actor(name)
            status = ray.get(discovered.get_status.remote(), timeout=5)
            assert status.mode == ControllerMode.MONITORING
        finally:
            _kill_named_actor(name)


class TestFtIdIsolation:
    def test_two_controllers_with_different_ft_id_are_independent(
        self,
        local_ray: None,
    ) -> None:
        name_a = ft_controller_actor_name("alpha")
        name_b = ft_controller_actor_name("beta")

        handle_a = FtControllerActor.options(name=name_a).remote(
            builder=build_ft_controller,
            config=FtControllerConfig(platform="stub", tick_interval=1.0, ft_id="alpha", rollout_num_cells=0),
        )
        handle_b = FtControllerActor.options(name=name_b).remote(
            builder=build_ft_controller,
            config=FtControllerConfig(platform="stub", tick_interval=1.0, ft_id="beta", rollout_num_cells=0),
        )
        try:
            handle_a.submit_and_run.remote()
            handle_b.submit_and_run.remote()

            run_id_a = poll_for_run_id(handle_a)
            run_id_b = poll_for_run_id(handle_b)
            assert run_id_a != run_id_b

            ray.get(
                handle_a.register_training_rank.remote(
                    run_id=run_id_a,
                    rank=0,
                    world_size=1,
                    node_id="n0",
                    exporter_address="http://n0:9090",
                    pid=1000,
                ),
                timeout=5,
            )
            ray.get(
                handle_b.register_training_rank.remote(
                    run_id=run_id_b,
                    rank=0,
                    world_size=1,
                    node_id="n1",
                    exporter_address="http://n1:9090",
                    pid=2000,
                ),
                timeout=5,
            )
        finally:
            try:
                ray.get(handle_a.shutdown.remote(), timeout=5)
            except Exception:
                pass
            try:
                ray.get(handle_b.shutdown.remote(), timeout=5)
            except Exception:
                pass
            _kill_named_actor(name_a)
            _kill_named_actor(name_b)


class TestDuplicateActorName:
    def test_duplicate_name_raises(
        self,
        controller_actor: ray.actor.ActorHandle,
    ) -> None:
        name = ft_controller_actor_name("")
        with pytest.raises(ValueError):
            FtControllerActor.options(name=name).remote(
                builder=build_ft_controller,
                config=FtControllerConfig(platform="stub", rollout_num_cells=0),
            )


class TestShutdownReleasesName:
    @pytest.mark.anyio
    async def test_name_released_after_shutdown(self, local_ray: None) -> None:
        name = ft_controller_actor_name("")
        handle = FtControllerActor.options(name=name).remote(
            builder=build_ft_controller,
            config=FtControllerConfig(platform="stub", tick_interval=0.05, rollout_num_cells=0),
        )

        run_ref = handle.run.remote()
        await __import__("asyncio").sleep(0.2)
        ray.get(handle.shutdown.remote(), timeout=5)
        ray.get(run_ref, timeout=5)

        ray.kill(handle, no_restart=True)
        del handle

        released = False
        for _ in range(20):
            time.sleep(0.5)
            try:
                ray.get_actor(name)
            except ValueError:
                released = True
                break
        assert released, f"Actor name {name!r} was not released after kill"

        handle2 = FtControllerActor.options(name=name).remote(
            builder=build_ft_controller,
            config=FtControllerConfig(platform="stub", tick_interval=1.0, rollout_num_cells=0),
        )
        try:
            status = ray.get(handle2.get_status.remote(), timeout=5)
            assert status.mode == ControllerMode.MONITORING
        finally:
            _kill_named_actor(name)


class TestKillAndAutoRestart:
    def test_kill_restarts_actor_with_fresh_state(
        self,
        controller_actor: ray.actor.ActorHandle,
    ) -> None:
        status_before = ray.get(controller_actor.get_status.remote(), timeout=5)
        assert status_before.mode == ControllerMode.MONITORING

        ray.kill(controller_actor, no_restart=False)

        name = ft_controller_actor_name("")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                restarted = ray.get_actor(name)
                status_after = ray.get(restarted.get_status.remote(), timeout=2)
                break
            except Exception:
                time.sleep(0.3)
        else:
            raise TimeoutError("Actor did not restart within 10s")

        assert status_after.mode == ControllerMode.MONITORING
        assert status_after.tick_count == 0


class TestNonExistentActorRaises:
    def test_get_actor_raises_value_error(self, local_ray: None) -> None:
        with pytest.raises(ValueError):
            ray.get_actor("ft_controller_does_not_exist_12345")
