"""Local Ray: Serialization boundary — enum fidelity, NaN/Inf, exceptions, config."""

from __future__ import annotations

import pytest
import ray
from tests.fast.utils.ft.integration.conftest import _kill_named_actor

from miles.utils.ft.adapters.config import FtControllerConfig
from miles.utils.ft.adapters.impl.ray.controller_actor import FtControllerActor
from miles.utils.ft.adapters.types import ft_controller_actor_name
from miles.utils.ft.controller.types import ControllerMode, ControllerStatus
from miles.utils.ft.factories.controller.from_config import build_ft_controller

pytestmark = [
    pytest.mark.local_ray,
]


class TestControllerStatusSerialization:
    def test_enum_types_preserved_across_ray_boundary(
        self,
        controller_actor: ray.actor.ActorHandle,
    ) -> None:
        status: ControllerStatus = ray.get(
            controller_actor.get_status.remote(),
            timeout=5,
        )
        assert type(status.mode) is ControllerMode
        assert status.mode == ControllerMode.MONITORING


class TestNanInfSerialization:
    def test_nan_and_inf_metrics_do_not_raise(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
    ) -> None:
        handle, run_id = running_controller

        ray.get(
            handle.register_training_rank.remote(
                run_id=run_id,
                rank=0,
                world_size=1,
                node_id="n0",
                exporter_address="http://n0:9090",
                pid=1000,
            ),
            timeout=5,
        )

        ray.get(
            handle.log_step.remote(
                run_id=run_id,
                step=1,
                metrics={
                    "loss": float("nan"),
                    "grad_norm": float("inf"),
                    "neg_inf": float("-inf"),
                },
            ),
            timeout=5,
        )


class TestConfigSerialization:
    def test_frozen_pydantic_config_survives_cloudpickle(
        self,
        local_ray: None,
    ) -> None:
        name = ft_controller_actor_name("config-test")
        config = FtControllerConfig(
            platform="stub",
            tick_interval=0.1,
            ft_id="config-test",
            rollout_num_cells=0,
        )
        handle = FtControllerActor.options(name=name).remote(builder=build_ft_controller, config=config)
        try:
            status = ray.get(handle.get_status.remote(), timeout=5)
            assert isinstance(status, ControllerStatus)
        finally:
            _kill_named_actor(name)


class TestExceptionSerialization:
    def test_actor_survives_after_invalid_registration(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
    ) -> None:
        """An RPC with invalid args should not kill the actor."""
        handle, run_id = running_controller

        try:
            ray.get(
                handle.register_training_rank.remote(
                    run_id=run_id,
                    rank=-1,
                    world_size=0,
                    node_id="",
                    exporter_address="",
                    pid=1000,
                ),
                timeout=5,
            )
        except ray.exceptions.RayTaskError:
            pass

        status = ray.get(handle.get_status.remote(), timeout=5)
        assert isinstance(status.mode, ControllerMode)


class TestLargeDictSerialization:
    def test_thousand_metric_keys_serialize_successfully(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
    ) -> None:
        handle, run_id = running_controller

        large_metrics = {f"metric_{i}": float(i) for i in range(1000)}
        ray.get(
            handle.log_step.remote(
                run_id=run_id,
                step=1,
                metrics=large_metrics,
            ),
            timeout=10,
        )
