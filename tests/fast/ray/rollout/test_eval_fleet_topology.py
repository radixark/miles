import argparse
import asyncio
from types import SimpleNamespace

from miles.ray.rollout.eval_fleet import InferenceControllerEvalFleet


class TestEvalFleetTopology:
    async def test_info_waits_for_topology_update_and_returns_a_detached_snapshot(self) -> None:
        """A topology read observes completed locked updates and cannot alias mutable server state."""
        lock = asyncio.Lock()
        counts = [2]
        server = SimpleNamespace(context_lock=lock, engine_gpu_counts=counts, router_ip="127.0.0.1", router_port=8000)
        fleet = InferenceControllerEvalFleet(argparse.Namespace(), srv=server)

        async with lock:
            pending = asyncio.create_task(fleet.info())
            await asyncio.sleep(0)
            assert not pending.done()
            counts.append(4)
        info = await asyncio.wait_for(pending, timeout=1)
        counts.append(8)

        assert info.engine_gpu_counts == [2, 4]
        info.engine_gpu_counts.append(16)
        assert counts == [2, 4, 8]

    async def test_empty_fleet_is_reported_without_inventing_a_configured_engine(self) -> None:
        """An observed empty fleet stays empty regardless of configured deployment size."""
        server = SimpleNamespace(
            context_lock=asyncio.Lock(), engine_gpu_counts=[], router_ip="127.0.0.1", router_port=8000
        )
        fleet = InferenceControllerEvalFleet(argparse.Namespace(eval_num_gpus=16), srv=server)

        assert (await fleet.info()).engine_gpu_counts == []
