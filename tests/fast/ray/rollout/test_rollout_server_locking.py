import asyncio
from types import SimpleNamespace

import pytest

from miles.ray.rollout.rollout_server import RolloutServer
from miles.utils.context_lock import ContextLock
from miles.utils.workers.worker_spec import NamedHostAndPorts


class _StubProvider:
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        raise AssertionError(f"no cell in this module is ever addressed ({worker_name=})")


def _make_server(context_lock: ContextLock | None = None, **overrides) -> RolloutServer:
    return RolloutServer(
        server_cells={},
        args=SimpleNamespace(colocate=True),
        context_lock=context_lock or ContextLock("InferenceController"),
        engine_provider=_StubProvider(),
        **overrides,
    )


class TestRolloutServerLockDiscipline:
    @pytest.mark.asyncio
    async def test_methods_reject_callers_that_do_not_hold_the_lock(self):
        """Driving RolloutServer directly, bypassing InferenceController, is rejected."""
        srv = _make_server()
        with pytest.raises(AssertionError, match="must be called with"):
            await srv.offload()

    def test_properties_reject_callers_that_do_not_hold_the_lock(self):
        """Engine list snapshots are only meaningful under the controller lock."""
        srv = _make_server()
        with pytest.raises(AssertionError, match="must be called with"):
            _ = srv.api_clients

    @pytest.mark.asyncio
    async def test_methods_accept_callers_inside_the_lock(self):
        """A caller holding the controller lock may drive RolloutServer freely."""
        srv = _make_server()
        async with srv.context_lock:
            await srv.offload()
            await srv.onload()
            assert srv.api_clients == []
            assert srv.engine_gpu_counts == []
            assert srv.engine_gpu_offsets == []

    @pytest.mark.asyncio
    async def test_holding_a_different_lock_object_does_not_authorize_the_server(self):
        """The server must be handed the controller's own lock, not merely some lock."""
        srv = _make_server()
        async with ContextLock("InferenceController"):
            with pytest.raises(AssertionError, match="must be called with"):
                await srv.offload()

    @pytest.mark.asyncio
    async def test_all_servers_of_one_controller_share_a_single_lock(self):
        """One controller lock guards every model's server, so one acquire covers them all."""
        shared_lock = ContextLock("InferenceController")
        first = _make_server(shared_lock)
        second = _make_server(shared_lock)

        async with shared_lock:
            await first.offload()
            await second.offload()


class TestWaitExpectedNumCellsIsLockFree:
    @pytest.mark.asyncio
    async def test_it_can_be_awaited_without_the_lock(self):
        """It runs during startup from create(), which deliberately holds no lock."""
        srv = _make_server(init_expected_num_cells=0)
        await srv.wait_init_expected_num_cells()

    @pytest.mark.asyncio
    async def test_cells_can_still_be_added_while_it_polls(self):
        """Polling must not hold the lock, otherwise reconcile could never add the cells it waits for."""
        srv = _make_server(init_expected_num_cells=1)
        waiter = asyncio.create_task(srv.wait_init_expected_num_cells())
        await asyncio.sleep(0)

        async with srv.context_lock:
            srv.server_cells["inference-engine-0-0-0"] = SimpleNamespace()

        await asyncio.wait_for(waiter, timeout=5)
