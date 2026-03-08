"""Local Ray: Async concurrency — run() interleaving, concurrent RPCs, shutdown."""

from __future__ import annotations

import asyncio
import time

import pytest
import ray
from tests.fast.utils.ft.integration.conftest import get_status

from miles.utils.ft.models.recovery import ControllerMode

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]


class TestRunLoopInterleaving:
    """run() should not block concurrent RPCs thanks to async await points."""

    async def test_get_status_while_run_is_active(
        self,
        controller_actor: ray.actor.ActorHandle,
    ) -> None:
        controller_actor.run.remote()
        await asyncio.sleep(0.2)

        status = ray.get(controller_actor.get_status.remote(), timeout=5)
        assert status.mode == ControllerMode.MONITORING

    async def test_register_training_rank_while_run_is_active(
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

        status = get_status(handle)
        assert status.active_run_id == run_id

    async def test_log_step_while_run_is_active(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
    ) -> None:
        handle, run_id = running_controller

        ray.get(
            handle.log_step.remote(
                run_id=run_id,
                step=1,
                metrics={"loss": 0.5},
            ),
            timeout=5,
        )


class TestSubmitAndRunInterleaving:
    async def test_get_status_during_submit_and_run(
        self,
        controller_actor: ray.actor.ActorHandle,
    ) -> None:
        controller_actor.submit_and_run.remote()

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            status = ray.get(controller_actor.get_status.remote(), timeout=5)
            assert isinstance(status.mode, ControllerMode)
            if status.active_run_id is not None:
                break
            await asyncio.sleep(0.3)
        else:
            raise TimeoutError("active_run_id was not set within 10s after submit_and_run")


class TestShutdownInterruptsRun:
    async def test_shutdown_causes_run_to_exit(
        self,
        controller_actor: ray.actor.ActorHandle,
    ) -> None:
        run_ref = controller_actor.run.remote()
        await asyncio.sleep(0.2)

        ray.get(controller_actor.shutdown.remote(), timeout=5)
        ray.get(run_ref, timeout=5)


class TestConcurrentRegisterTrainingRank:
    async def test_eight_ranks_register_simultaneously(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
    ) -> None:
        handle, run_id = running_controller

        refs = [
            handle.register_training_rank.remote(
                run_id=run_id,
                rank=i,
                world_size=8,
                node_id=f"n-{i}",
                exporter_address=f"http://n-{i}:9090",
                pid=1000 + i,
            )
            for i in range(8)
        ]
        ray.get(refs, timeout=10)

        status = get_status(handle)
        assert status.active_run_id == run_id


class TestConcurrentGetStatus:
    async def test_twenty_concurrent_get_status(
        self,
        controller_actor: ray.actor.ActorHandle,
    ) -> None:
        refs = [controller_actor.get_status.remote() for _ in range(20)]
        results = ray.get(refs, timeout=10)
        assert len(results) == 20
        for s in results:
            assert isinstance(s.mode, ControllerMode)


class TestFireAndForgetBackpressure:
    """1000 rapid fire-and-forget log_step calls should not crash."""

    async def test_thousand_log_steps_no_error(
        self,
        running_controller: tuple[ray.actor.ActorHandle, str],
    ) -> None:
        handle, run_id = running_controller

        for i in range(1000):
            handle.log_step.remote(
                run_id=run_id,
                step=i,
                metrics={"loss": float(i)},
            )

        await asyncio.sleep(2.0)

        status = get_status(handle)
        assert isinstance(status.mode, ControllerMode)
