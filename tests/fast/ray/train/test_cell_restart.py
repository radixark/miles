import asyncio

import pytest
import ray
from tests.fast.ray.train import conftest as train_conftest
from tests.fast.ray.train.conftest import get_raw_actor_handles, make_cell

from miles.ray.train import cell as cell_module

pytestmark = pytest.mark.asyncio


class _HangingKillSelfHandle:
    """A worker handle whose kill_self never returns, so _kill_worker has to time it out."""

    def __init__(self) -> None:
        self.kill_self_call_count: int = 0
        self.wait_dead_call_count: int = 0

    async def kill_self(self) -> None:
        self.kill_self_call_count += 1
        await asyncio.Event().wait()

    async def wait_dead(self, *, timeout: float) -> None:
        self.wait_dead_call_count += 1


class TestCellKillAndRestart:
    async def test_killing_a_failed_cell_reaches_the_workers_directly(self):
        """Waiting for an external controller would leave the other cells hanging in NCCL."""
        cell = make_cell(2)
        handles = get_raw_actor_handles(cell)

        await cell._kill_workers_and_confirm_dead()

        assert train_conftest.fake_worker_manager.stopped_cell_ids == []
        for handle in handles:
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(handle.get_calls.remote())

    async def test_a_replacement_cell_picks_up_the_fresh_actor_handles(self):
        """Reusing the dead handles would make every later call fail."""
        cell = make_cell(0)
        old_handles = get_raw_actor_handles(cell)
        await cell._kill_workers_and_confirm_dead()

        train_conftest.fake_worker_manager._stop_cells([cell.cell_id])
        replacement = make_cell(0)

        assert get_raw_actor_handles(replacement) != old_handles

    async def test_killing_twice_is_harmless(self):
        """Healing may tear down an already dead cell, which must not raise."""
        cell = make_cell(0)
        await cell._kill_workers_and_confirm_dead()

        await cell._kill_workers_and_confirm_dead()


class TestKillRpcTimeout:
    async def test_a_kill_rpc_that_never_returns_gives_up_at_the_timeout(self, monkeypatch: pytest.MonkeyPatch):
        """A worker whose kill_self RPC hangs forever must not block _kill_worker forever."""
        monkeypatch.setattr(cell_module, "KILL_RPC_TIMEOUT_S", 0.05)
        handle = _HangingKillSelfHandle()

        await asyncio.wait_for(cell_module._kill_worker(handle), timeout=10.0)

        assert handle.kill_self_call_count == 1

    async def test_a_hanging_kill_rpc_still_reaches_the_death_confirmation(self, monkeypatch: pytest.MonkeyPatch):
        """When every kill_self RPC hangs, teardown must fall through to the death probe instead of stalling."""
        monkeypatch.setattr(cell_module, "KILL_RPC_TIMEOUT_S", 0.05)
        cell = make_cell(2)
        hanging_handles: list[_HangingKillSelfHandle] = [_HangingKillSelfHandle(), _HangingKillSelfHandle()]
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: hanging_handles)

        await asyncio.wait_for(cell._kill_workers_and_confirm_dead(), timeout=10.0)

        assert [handle.kill_self_call_count for handle in hanging_handles] == [1, 1]
        assert [handle.wait_dead_call_count for handle in hanging_handles] == [1, 1]
