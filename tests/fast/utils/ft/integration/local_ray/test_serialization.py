"""Local Ray: Serialization boundary — enum fidelity, NaN/Inf, exceptions, config."""
from __future__ import annotations

import pytest
import ray

from miles.utils.ft.models import ControllerMode, ControllerStatus
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


class TestControllerStatusSerialization:
    def test_enum_types_preserved_across_ray_boundary(
        self, controller_actor: ray.actor.ActorHandle,
    ) -> None:
        status: ControllerStatus = ray.get(
            controller_actor.get_status.remote(), timeout=5,
        )
        assert type(status.mode) is ControllerMode
        assert status.mode == ControllerMode.MONITORING


class TestNanInfSerialization:
    def test_nan_and_inf_metrics_do_not_raise(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
    ) -> None:
        handle, run_id = running_controller

        ray.get(handle.register_training_rank.remote(
            run_id=run_id, rank=0, world_size=1,
            node_id="n0", exporter_address="http://n0:9090",
        ), timeout=5)

        ray.get(handle.log_step.remote(
            run_id=run_id,
            step=1,
            metrics={
                "loss": float("nan"),
                "grad_norm": float("inf"),
                "neg_inf": float("-inf"),
            },
        ), timeout=5)


class TestConfigSerialization:
    def test_frozen_pydantic_config_survives_cloudpickle(
        self, local_ray: None,
    ) -> None:
        name = ft_controller_actor_name("config-test")
        config = FtControllerConfig(
            platform="stub",
            tick_interval=0.1,
            ft_id="config-test",
        )
        handle = FtControllerActor.options(name=name).remote(config=config)
        try:
            status = ray.get(handle.get_status.remote(), timeout=5)
            assert isinstance(status, ControllerStatus)
        finally:
            _kill_actor(name)


class TestExceptionSerialization:
    def test_actor_survives_after_invalid_registration(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
    ) -> None:
        """An RPC with invalid args should not kill the actor."""
        handle, run_id = running_controller

        try:
            ray.get(handle.register_training_rank.remote(
                run_id=run_id,
                rank=-1,
                world_size=0,
                node_id="",
                exporter_address="",
            ), timeout=5)
        except ray.exceptions.RayTaskError:
            pass

        status = ray.get(handle.get_status.remote(), timeout=5)
        assert isinstance(status.mode, ControllerMode)
