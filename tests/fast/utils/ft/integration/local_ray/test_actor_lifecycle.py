"""Local Ray: Actor lifecycle — naming, discovery, shutdown, restart."""
from __future__ import annotations

import time

import pytest
import ray

from miles.utils.ft.models import ControllerMode
from miles.utils.ft.platform.controller_actor import FtControllerActor
from miles.utils.ft.platform.controller_factory import FtControllerConfig
from miles.utils.ft.protocols.platform import ft_controller_actor_name

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.timeout(60),
]


def _kill_actor(name: str) -> None:
    try:
        ray.kill(ray.get_actor(name), no_restart=True)
    except ValueError:
        pass


class TestNamedActorCreation:
    def test_create_and_discover_via_get_actor(self, local_ray: None) -> None:
        name = ft_controller_actor_name("")
        handle = FtControllerActor.options(name=name).remote(
            config=FtControllerConfig(platform="stub", tick_interval=1.0),
        )
        try:
            discovered = ray.get_actor(name)
            status = ray.get(discovered.get_status.remote(), timeout=5)
            assert status.mode == ControllerMode.MONITORING
        finally:
            _kill_actor(name)


class TestFtIdIsolation:
    def test_two_controllers_with_different_ft_id_are_independent(
        self, local_ray: None,
    ) -> None:
        name_a = ft_controller_actor_name("alpha")
        name_b = ft_controller_actor_name("beta")

        handle_a = FtControllerActor.options(name=name_a).remote(
            config=FtControllerConfig(platform="stub", tick_interval=1.0, ft_id="alpha"),
        )
        handle_b = FtControllerActor.options(name=name_b).remote(
            config=FtControllerConfig(platform="stub", tick_interval=1.0, ft_id="beta"),
        )
        try:
            handle_a.submit_and_run.remote()
            handle_b.submit_and_run.remote()
            time.sleep(0.3)

            status_a = ray.get(handle_a.get_status.remote(), timeout=5)
            status_b = ray.get(handle_b.get_status.remote(), timeout=5)

            assert status_a.active_run_id is not None
            assert status_b.active_run_id is not None
            assert status_a.active_run_id != status_b.active_run_id

            ray.get(handle_a.register_training_rank.remote(
                run_id=status_a.active_run_id, rank=0, world_size=1,
                node_id="n0", exporter_address="http://n0:9090",
            ), timeout=5)
            ray.get(handle_b.register_training_rank.remote(
                run_id=status_b.active_run_id, rank=0, world_size=1,
                node_id="n1", exporter_address="http://n1:9090",
            ), timeout=5)
        finally:
            try:
                ray.get(handle_a.shutdown.remote(), timeout=5)
            except Exception:
                pass
            try:
                ray.get(handle_b.shutdown.remote(), timeout=5)
            except Exception:
                pass
            _kill_actor(name_a)
            _kill_actor(name_b)


class TestDuplicateActorName:
    def test_duplicate_name_raises(
        self, controller_actor: ray.actor.ActorHandle,
    ) -> None:
        name = ft_controller_actor_name("")
        with pytest.raises(Exception):
            FtControllerActor.options(name=name).remote(
                config=FtControllerConfig(platform="stub"),
            )


class TestShutdownReleasesName:
    @pytest.mark.anyio
    async def test_name_released_after_shutdown(self, local_ray: None) -> None:
        name = ft_controller_actor_name("")
        handle = FtControllerActor.options(name=name).remote(
            config=FtControllerConfig(platform="stub", tick_interval=0.05),
        )

        run_ref = handle.run.remote()
        await __import__("asyncio").sleep(0.2)
        ray.get(handle.shutdown.remote(), timeout=5)
        ray.get(run_ref, timeout=5)

        time.sleep(1.0)

        with pytest.raises(ValueError):
            ray.get_actor(name)

        handle2 = FtControllerActor.options(name=name).remote(
            config=FtControllerConfig(platform="stub", tick_interval=1.0),
        )
        try:
            status = ray.get(handle2.get_status.remote(), timeout=5)
            assert status.mode == ControllerMode.MONITORING
        finally:
            _kill_actor(name)


class TestKillAndAutoRestart:
    def test_kill_restarts_actor_with_fresh_state(
        self, controller_actor: ray.actor.ActorHandle,
    ) -> None:
        status_before = ray.get(controller_actor.get_status.remote(), timeout=5)
        assert status_before.mode == ControllerMode.MONITORING

        ray.kill(controller_actor, no_restart=False)
        time.sleep(2.0)

        name = ft_controller_actor_name("")
        restarted = ray.get_actor(name)
        status_after = ray.get(restarted.get_status.remote(), timeout=5)

        assert status_after.mode == ControllerMode.MONITORING
        assert status_after.tick_count == 0


class TestNonExistentActorRaises:
    def test_get_actor_raises_value_error(self, local_ray: None) -> None:
        with pytest.raises(ValueError):
            ray.get_actor("ft_controller_does_not_exist_12345")
