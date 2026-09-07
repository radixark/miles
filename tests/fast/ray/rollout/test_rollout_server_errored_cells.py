from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from tests.fast.ray.rollout.conftest import make_args, track_server_cell

from miles.ray.rollout import server_cell as server_cell_module
from miles.ray.rollout.cell_state import CellAddrInfo
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.workers.worker_spec import HostAndPort

pytestmark = pytest.mark.usefixtures("dispose_tracked_server_cells")


def _server_url(gpu_offset: int) -> str:
    return f"http://10.0.0.1:{30000 + gpu_offset}"


class _StubProvider:
    async def get_addrs(self, worker_name: str) -> dict[str, HostAndPort]:
        raise AssertionError(f"the cells of this module resolve no addresses ({worker_name=})")


class _RecordingRouterApiClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def add_worker(self, **kwargs) -> None:
        self.calls.append(("add_worker", kwargs))

    async def remove_worker(self, **kwargs) -> None:
        self.calls.append(("remove_worker", kwargs))


class _RecordingApiClient:
    calls: list[tuple[str, str]] = []

    def __init__(self, server_url: str, api_key: str | None = None) -> None:
        self.server_url = server_url
        self.api_key = api_key

    async def release_memory_occupation(self, tags=None) -> None:
        _RecordingApiClient.calls.append(("release_memory_occupation", self.server_url))

    async def resume_memory_occupation(self, tags=None) -> None:
        _RecordingApiClient.calls.append(("resume_memory_occupation", self.server_url))

    async def check_weights(self, **kwargs) -> None:
        _RecordingApiClient.calls.append(("check_weights", self.server_url))

    async def abort_all_requests(self, timeout: float | None = None) -> None:
        _RecordingApiClient.calls.append(("abort_all_requests", self.server_url))


def _make_meta(*, gpu_offset: int, needs_offload: bool) -> ServerCellMetadata:
    return ServerCellMetadata(
        model_id="default",
        worker_type="regular",
        cell_id=f"inference-engine-0-0-{gpu_offset}",
        num_gpus_per_engine=gpu_offset + 1,
        gpu_offset=gpu_offset,
        sglang_api_key=None,
        worker_name=f"inference-engine-0-0-{gpu_offset}-0",
        needs_offload=needs_offload,
        update_weights=True,
        workers_hash=f"pseudo-hash-{gpu_offset}",
    )


@pytest.fixture
def cell_env(monkeypatch):
    """Give every cell of this module a per-offset address and a recording engine client."""
    _RecordingApiClient.calls = []

    async def _compute_addr_info(self) -> CellAddrInfo:
        return CellAddrInfo(server_url=_server_url(self.meta.gpu_offset), bootstrap_port=None, gate_url=None)

    async def _probe(server_url: str, api_key, timeout: float = 5.0) -> bool:
        return True

    monkeypatch.setattr(server_cell_module, "probe_server_healthy", _probe)
    monkeypatch.setattr(server_cell_module, "SGLangApiClient", _RecordingApiClient)
    monkeypatch.setattr(ServerCell, "_compute_addr_info", _compute_addr_info)

    return _RecordingApiClient


def _make_server(*, colocate: bool = False, **overrides) -> RolloutServer:
    return RolloutServer(
        server_cells={},
        args=make_args(colocate=colocate),
        context_lock=ContextLock("InferenceController"),
        engine_provider=_StubProvider(),
        health_checker_activeness=ActivenessTracker(active=True),
        **overrides,
    )


async def _add_serving_cell(
    srv: RolloutServer, router: _RecordingRouterApiClient, *, gpu_offset: int, needs_offload: bool = False
) -> ServerCell:
    cell = track_server_cell(
        ServerCell(
            args=srv.args,
            meta=_make_meta(gpu_offset=gpu_offset, needs_offload=needs_offload),
            router_api_client=router,
            provider=_StubProvider(),
        )
    )
    srv.server_cells[cell.meta.cell_id] = cell
    await cell.init()
    await cell.tick()
    await cell.mark_weights_ready()
    return cell


class TestErroredCellsLeaveTheEngineLists:
    async def test_every_per_engine_list_drops_the_errored_cell_together(self, cell_env):
        """These lists are indexed in parallel by the trainer, so dropping one only would misplace every shard."""
        srv = _make_server()
        router = _RecordingRouterApiClient()
        healthy = await _add_serving_cell(srv, router, gpu_offset=0)
        failed = await _add_serving_cell(srv, router, gpu_offset=1)
        await failed.mark_errored()

        async with srv.context_lock:
            assert srv.engine_cells == [healthy]
            assert [client.server_url for client in srv.api_clients] == [_server_url(0)]
            assert srv.engine_gpu_counts == [1]
            assert srv.engine_gpu_offsets == [0]

    async def test_a_server_whose_only_cell_errored_offers_no_engine(self, cell_env):
        """Handing out the failed engine would send the next update straight back into it."""
        srv = _make_server()
        router = _RecordingRouterApiClient()
        only = await _add_serving_cell(srv, router, gpu_offset=0)
        await only.mark_errored()

        async with srv.context_lock:
            assert srv.engine_cells == []
            assert srv.api_clients == []


class TestErroredCellsAreNotAskedForAnything:
    @pytest.mark.parametrize("op", ["offload", "onload"])
    async def test_memory_fan_out_skips_the_errored_cell(self, cell_env, op: str):
        """Its engine is being reclaimed, so the call would hang or fail the whole fan-out."""
        srv = _make_server(colocate=True)
        router = _RecordingRouterApiClient()
        await _add_serving_cell(srv, router, gpu_offset=0, needs_offload=True)
        failed = await _add_serving_cell(srv, router, gpu_offset=1, needs_offload=True)
        await failed.mark_errored()
        cell_env.calls.clear()

        async with srv.context_lock:
            await getattr(srv, op)()

        assert cell_env.calls == [(f"{'release' if op == 'offload' else 'resume'}_memory_occupation", _server_url(0))]

    async def test_check_weights_skips_the_errored_cell(self, cell_env):
        """A checksum of an engine nobody updated would fail a run that is recovering correctly."""
        srv = _make_server()
        router = _RecordingRouterApiClient()
        await _add_serving_cell(srv, router, gpu_offset=0)
        failed = await _add_serving_cell(srv, router, gpu_offset=1)
        await failed.mark_errored()
        cell_env.calls.clear()

        async with srv.context_lock:
            await srv.check_weights(action="checksum")

        assert cell_env.calls == [("check_weights", _server_url(0))]

    async def test_abort_all_skips_the_errored_cell(self, cell_env):
        """Aborting on a dead engine would time out and hide a live engine's real abort failure."""
        srv = _make_server()
        router = _RecordingRouterApiClient()
        await _add_serving_cell(srv, router, gpu_offset=0)
        failed = await _add_serving_cell(srv, router, gpu_offset=1)
        await failed.mark_errored()
        cell_env.calls.clear()

        async with srv.context_lock:
            await srv.abort_all()

        assert cell_env.calls == [("abort_all_requests", _server_url(0))]


class TestErroredCellsAreNotStartable:
    async def test_the_startup_barrier_does_not_count_an_errored_cell(self, cell_env, monkeypatch):
        """Counting it would declare the pool ready while one of its engines is being replaced."""
        srv = _make_server(colocate=True, init_expected_num_cells=2)
        router = _RecordingRouterApiClient()
        await _add_serving_cell(srv, router, gpu_offset=0, needs_offload=True)
        failed = await _add_serving_cell(srv, router, gpu_offset=1, needs_offload=True)
        await failed.mark_errored()

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(srv.wait_init_expected_num_cells(timeout=10.0), timeout=0.2)

    async def test_the_startup_barrier_passes_once_every_remaining_cell_is_ready(self, cell_env):
        """The count must still be met by the cells that did come up, or startup could never finish."""
        srv = _make_server(init_expected_num_cells=1)
        router = _RecordingRouterApiClient()
        await _add_serving_cell(srv, router, gpu_offset=0)
        failed = await _add_serving_cell(srv, router, gpu_offset=1)
        await failed.mark_errored()

        await asyncio.wait_for(srv.wait_init_expected_num_cells(timeout=10.0), timeout=1.0)


class TestStartUpdateWeightsSkipsErroredCells:
    def _controller(self, srv: RolloutServer) -> InferenceController:
        controller = InferenceController.__new__(InferenceController)
        controller.args = srv.args
        controller.servers = {"default": srv}
        controller.context_lock = srv.context_lock
        controller._cell_operations = AsyncMock()
        return controller

    async def test_the_next_update_targets_only_the_cells_that_are_still_alive(self, cell_env):
        """Handing the trainer a failed engine would repeat the transfer that just failed."""
        srv = _make_server()
        router = _RecordingRouterApiClient()
        healthy = await _add_serving_cell(srv, router, gpu_offset=0)
        failed = await _add_serving_cell(srv, router, gpu_offset=1)
        await failed.mark_errored()

        updatable = await self._controller(srv).start_update_weights()

        assert updatable.engine_cell_ids == [healthy.meta.cell_id]
        assert updatable.snapshot_cell_id_to_hashes == {healthy.meta.cell_id: healthy.meta.workers_hash}
        assert updatable.engine_gpu_counts == [healthy.meta.num_gpus_per_engine]

    async def test_the_window_does_not_wait_for_an_errored_cell_to_become_ready(self, cell_env):
        """An errored cell never becomes ready again, so waiting for it would stall every later update."""
        srv = _make_server()
        router = _RecordingRouterApiClient()
        await _add_serving_cell(srv, router, gpu_offset=0)
        failed = await _add_serving_cell(srv, router, gpu_offset=1)
        await failed.mark_errored()

        await asyncio.wait_for(self._controller(srv).start_update_weights(), timeout=1.0)
