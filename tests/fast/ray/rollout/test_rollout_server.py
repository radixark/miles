from __future__ import annotations

from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.sglang_utils.sglang_config import (
    _compute_megatron_num_gpus,
    _compute_rollout_offset,
    resolve_sglang_config,
)
from miles.ray.rollout.cell_state import CellAddrInfo, StateServing
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.context_lock import ContextLock


class TestRolloutServerPureFunctions:
    def test_resolve_sglang_config_yaml_gpu_mismatch_asserts(self, tmp_path):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 1\n"
        )
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=8)
        with pytest.raises(AssertionError, match="total GPUs"):
            resolve_sglang_config(args)

    def test_compute_rollout_offset_colocate_returns_zero(self):
        args = make_args(
            colocate=True,
            debug_train_only=False,
            debug_rollout_only=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            use_critic=False,
        )
        assert _compute_rollout_offset(args) == 0

    def test_compute_rollout_offset_critic_train_only(self):
        args = make_args(
            colocate=False,
            debug_train_only=False,
            debug_rollout_only=False,
            critic_train_only=True,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
        )
        assert _compute_rollout_offset(args) == 4

    def test_compute_rollout_offset_shared_actor_critic(self):
        args = make_args(
            colocate=False,
            debug_train_only=False,
            debug_rollout_only=False,
            critic_train_only=False,
            use_critic=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
        )
        assert _compute_rollout_offset(args) == 8

    def test_compute_megatron_num_gpus_for_actor_only(self):
        args = make_args(
            actor_num_nodes=2,
            actor_num_gpus_per_node=8,
            use_critic=False,
            debug_rollout_only=False,
            critic_train_only=False,
        )
        assert _compute_megatron_num_gpus(args) == 16

    def test_compute_megatron_num_gpus_with_shared_critic(self):
        args = make_args(
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            use_critic=True,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
            debug_rollout_only=False,
            critic_train_only=False,
        )
        assert _compute_megatron_num_gpus(args) == 8

    def test_compute_megatron_num_gpus_zero_when_debug_rollout_only(self):
        args = make_args(debug_rollout_only=True)
        assert _compute_megatron_num_gpus(args) == 0


class TestEngineListOrdering:
    def _server_with_cells(self, num_cells: int) -> RolloutServer:
        cells = {}
        for index in sorted(range(num_cells), key=lambda i: f"inference-engine-0-0-{i}"):
            meta = SimpleNamespace(num_gpus_per_engine=index + 1, gpu_offset=index)
            cells[f"inference-engine-0-0-{index}"] = SimpleNamespace(meta=meta, api_client=f"client-{index}")
        return RolloutServer(server_cells=cells, args=SimpleNamespace(), context_lock=_make_lock())

    @pytest.mark.asyncio
    async def test_engine_lists_are_ordered_by_gpu_offset_not_insertion(self):
        """With 12 cells inserted in string-sorted id order all three derived lists come out offset-ordered."""
        srv = self._server_with_cells(12)
        async with srv.context_lock:
            assert srv.engine_gpu_offsets == list(range(12))
            assert srv.api_clients == [f"client-{i}" for i in range(12)]
            assert srv.engine_gpu_counts == [i + 1 for i in range(12)]


class TestAddCellRollback:
    def _make_meta(self) -> ServerCellMetadata:
        return ServerCellMetadata(
            model_id="default",
            worker_type="regular",
            cell_id="inference-engine-0-0-0",
            num_gpus_per_engine=1,
            gpu_offset=0,
            sglang_api_key=None,
            worker_name="inference-engine-0-0-0-0",
            needs_offload=False,
            update_weights=True,
            workers_hash="pseudo-hash-0",
        )

    @pytest.mark.asyncio
    async def test_a_failed_add_still_tracks_the_cell_so_nothing_leaks(self, monkeypatch):
        """The cell is committed before init runs, so a failing init cannot orphan its health checker task."""
        srv = RolloutServer(
            server_cells={}, args=SimpleNamespace(colocate=False, ft_components=[]), context_lock=_make_lock()
        )
        monkeypatch.setattr(ServerCell, "init", _raise_async)

        async with srv.context_lock:
            with pytest.raises(RuntimeError, match="injected init failure"):
                await srv.add_cell(self._make_meta())

            assert list(srv.server_cells) == ["inference-engine-0-0-0"]
            await srv.dispose()

    @pytest.mark.asyncio
    async def test_disposing_the_server_removes_every_cell_it_tracks(self, monkeypatch):
        """Controller teardown must reach each cell so its health checker task stops with it."""
        srv = RolloutServer(
            server_cells={}, args=SimpleNamespace(colocate=True, ft_components=[]), context_lock=_make_lock()
        )
        monkeypatch.setattr(ServerCell, "init", _noop_async)

        async with srv.context_lock:
            await srv.add_cell(self._make_meta())
            await srv.dispose()

        assert srv.server_cells == {}

    @pytest.mark.asyncio
    async def test_a_successful_add_commits_the_cell(self, monkeypatch):
        """After the failure is gone the same cell id can be added normally."""
        srv = RolloutServer(
            server_cells={}, args=SimpleNamespace(colocate=False, ft_components=[]), context_lock=_make_lock()
        )
        monkeypatch.setattr(ServerCell, "init", _noop_async)

        async with srv.context_lock:
            await srv.add_cell(self._make_meta())

            assert list(srv.server_cells) == ["inference-engine-0-0-0"]
            await srv.dispose()


class TestAddCellInitTiming:
    @pytest.mark.asyncio
    async def test_a_disaggregated_cell_is_initialized_as_soon_as_it_appears(self, monkeypatch):
        """Nothing competes for its gpus, so waiting would only delay it becoming servable."""
        initialized: list[str] = []

        async def _record(self) -> None:
            initialized.append(self.meta.cell_id)

        srv = RolloutServer(
            server_cells={}, args=SimpleNamespace(colocate=False, ft_components=[]), context_lock=_make_lock()
        )
        monkeypatch.setattr(ServerCell, "init", _record)

        async with srv.context_lock:
            await srv.add_cell(TestAddCellRollback()._make_meta())
            await srv.dispose()

        assert initialized == ["inference-engine-0-0-0"]

    @pytest.mark.asyncio
    async def test_a_colocated_cell_is_only_tracked_until_the_weight_update_window(self, monkeypatch):
        """Its engine may not claim gpu memory while the trainer still holds it."""
        initialized: list[str] = []

        async def _record(self) -> None:
            initialized.append(self.meta.cell_id)

        srv = RolloutServer(
            server_cells={}, args=SimpleNamespace(colocate=True, ft_components=[]), context_lock=_make_lock()
        )
        monkeypatch.setattr(ServerCell, "init", _record)

        async with srv.context_lock:
            await srv.add_cell(TestAddCellRollback()._make_meta())

            assert initialized == []
            assert list(srv.server_cells) == ["inference-engine-0-0-0"]
            await srv.dispose()


def _make_lock() -> ContextLock:
    return ContextLock("InferenceController")


async def _raise_async(self):
    raise RuntimeError("injected init failure")


async def _noop_async(self):
    return None


class TestRemoveCell:
    @pytest.mark.asyncio
    async def test_removing_a_cell_detaches_it_from_every_view_derived_from_the_fleet(self, monkeypatch):
        """A removed cell left in one of the derived lists sends the trainer a weight shard for
        an engine that is gone, and the ranks disagree about how many engines there are."""
        srv = await _make_serving_server(monkeypatch, num_cells=2)

        async with srv.context_lock:
            await srv.remove_cell("default-0")

            assert list(srv.server_cells) == ["default-1"]
            assert srv.engine_gpu_offsets == [1]
            assert srv.engine_gpu_counts == [1]
            await srv.dispose()

    @pytest.mark.asyncio
    async def test_removing_a_cell_takes_its_url_out_of_the_router_first(self, monkeypatch):
        """Dropping the cell first loses the only record of what to unregister, and the router
        keeps sending generation to a worker that is being torn down."""
        srv = await _make_serving_server(monkeypatch, num_cells=2)
        router = srv.server_cells["default-0"].router_api_client

        async with srv.context_lock:
            await srv.remove_cell("default-0")
            await srv.dispose()

        assert ("remove_worker", "http://10.0.0.1:30000") in [
            (name, kwargs.get("worker_url")) for name, kwargs in router.calls
        ]


async def _make_serving_server(monkeypatch, *, num_cells: int) -> RolloutServer:
    """A server whose cells are all registered and serving, without dialling anything."""
    router = _RecordingRouterApiClient()
    monkeypatch.setattr(RolloutServer, "_router_api_client", property(lambda self: router))

    srv = RolloutServer(
        server_cells={},
        args=SimpleNamespace(colocate=True, ft_components=[], use_miles_router=False),
        context_lock=_make_lock(),
    )
    async with srv.context_lock:
        for cell_index in range(num_cells):
            await srv.add_cell(
                ServerCellMetadata(
                    model_id="default",
                    worker_type="regular",
                    cell_id=f"default-{cell_index}",
                    num_gpus_per_engine=1,
                    gpu_offset=cell_index,
                    sglang_api_key=None,
                    worker_name=f"default-{cell_index}-0",
                    needs_offload=False,
                    update_weights=True,
                    workers_hash=f"pseudo-hash-{cell_index}",
                )
            )
            cell = srv.server_cells[f"default-{cell_index}"]
            addr_info = CellAddrInfo(
                server_url=f"http://10.0.0.{cell_index + 1}:3000{cell_index}", bootstrap_port=None, gate_url=None
            )
            await cell._register_with_router(addr_info=addr_info)
            cell._state = StateServing(addr_info=addr_info)
    return srv


class _RecordingRouterApiClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def add_worker(self, **kwargs) -> None:
        self.calls.append(("add_worker", kwargs))

    async def remove_worker(self, **kwargs) -> None:
        self.calls.append(("remove_worker", kwargs))


def _make_lock() -> ContextLock:
    return ContextLock("InferenceController")


async def _raise_async(self):
    raise RuntimeError("injected init failure")


async def _noop_async(self):
    return None
