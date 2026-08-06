from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_dataclass_cells

from miles.backends.sglang_utils.sglang_config import (
    _compute_megatron_num_gpus,
    _compute_rollout_offset,
    resolve_sglang_config,
)
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata


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


@pytest.mark.skip(
    reason="TODO: rebuild against the meta/router_api_client ServerCell; make_dataclass_cells and "
    "_mark_allocated_uninitialized/_mark_addressing target the removed constructor and state API"
)
class TestRolloutServerCrossCellProperties:
    def test_api_clients_expose_one_client_per_cell(self):
        """Each cell is addressed through its primary (node-0) endpoint."""
        cells = make_dataclass_cells(num_cells=2, gpu_offset=0) + make_dataclass_cells(num_cells=2, gpu_offset=2)
        for index, cell in enumerate(cells):
            cell._mark_allocated_uninitialized()
            cell._mark_addressing(server_url=f"http://10.0.0.{index + 1}:30000")
        srv = RolloutServer(server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)})
        assert [client.server_url for client in srv.api_clients] == [
            f"http://10.0.0.{index + 1}:30000" for index in range(4)
        ]

    def test_engine_gpu_counts_parallel_to_engines(self):
        cells = make_dataclass_cells(num_cells=2, num_gpus_per_engine=1) + make_dataclass_cells(
            num_cells=2, num_gpus_per_engine=2
        )
        srv = RolloutServer(server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)})
        assert srv.engine_gpu_counts == [1, 1, 2, 2]

    def test_engine_gpu_offsets_consistent_across_cells(self):
        cells = make_dataclass_cells(num_cells=2, num_gpus_per_engine=1, gpu_offset=0) + make_dataclass_cells(
            num_cells=2, num_gpus_per_engine=2, gpu_offset=4
        )
        srv = RolloutServer(server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)})
        assert srv.engine_gpu_offsets == [0, 1, 4, 6]


class TestEngineListOrdering:
    def _server_with_cells(self, num_cells: int) -> RolloutServer:
        cells = {}
        for index in sorted(range(num_cells), key=lambda i: f"inference-engine-0-0-{i}"):
            meta = SimpleNamespace(num_gpus_per_engine=index + 1, gpu_offset=index)
            cells[f"inference-engine-0-0-{index}"] = SimpleNamespace(meta=meta, api_client=f"client-{index}")
        return RolloutServer(server_cells=cells, args=SimpleNamespace())

    def test_engine_lists_are_ordered_by_gpu_offset_not_insertion(self):
        """With 12 cells inserted in string-sorted id order all three derived lists come out offset-ordered."""
        srv = self._server_with_cells(12)
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
    async def test_a_failed_add_leaves_no_bookkeeping_so_the_next_reconcile_retries(self, monkeypatch):
        """A cell whose startup fails must not be committed, otherwise the hash no-op blocks any retry."""
        srv = RolloutServer(server_cells={}, args=SimpleNamespace())
        monkeypatch.setattr(ServerCell, "add", _raise_async)

        with pytest.raises(RuntimeError, match="injected add failure"):
            await srv.add_cell(self._make_meta())

        assert srv.server_cells == {}
        assert srv.has_new_engines is False

    @pytest.mark.asyncio
    async def test_a_successful_add_commits_the_cell_and_marks_new_engines(self, monkeypatch):
        """After the failure is gone the same cell id can be added normally."""
        srv = RolloutServer(server_cells={}, args=SimpleNamespace())
        monkeypatch.setattr(ServerCell, "add", _noop_async)

        await srv.add_cell(self._make_meta())

        assert list(srv.server_cells) == ["inference-engine-0-0-0"]
        assert srv.has_new_engines is True


async def _raise_async(self):
    raise RuntimeError("injected add failure")


async def _noop_async(self):
    return None


class TestRemoveCell:
    @pytest.mark.asyncio
    async def test_remove_cell_detaches_the_cell_from_every_server_view(self):
        """A removed cell is gone from server_cells and from every view derived from it."""
        events: list[dict[str, Any]] = []
        srv = _make_started_server(num_cells=2)

        with _with_recording_router(events):
            await srv.remove_cell("default-0")

        assert "default-0" not in srv.server_cells
        assert list(srv.server_cells) == ["default-1"]
        assert [client.server_url for client in srv.api_clients] == ["http://10.0.0.2:30001"]
        assert srv.engine_gpu_counts == [1]
        assert srv.engine_gpu_offsets == [1]

    @pytest.mark.asyncio
    async def test_remove_cell_unregisters_from_the_router_before_dropping_the_cell(self):
        """Dropping the cell without unregistering would leave the router routing to a dead worker."""
        events: list[dict[str, Any]] = []
        srv = _make_started_server(num_cells=2)

        with _with_recording_router(events):
            await srv.remove_cell("default-0")

        assert events == [{"call": "remove_worker", "worker_url": "http://10.0.0.1:30000", "use_legacy_api": False}]


def _make_started_server(*, num_cells: int) -> RolloutServer:
    args = make_args(num_gpus_per_node=8)
    srv = RolloutServer(server_cells={}, args=args)
    for cell_index in range(num_cells):
        meta = ServerCellMetadata(
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
        cell = ServerCell(args=args, meta=meta)
        cell._mark_addressing(AddrInfo(server_url=f"http://10.0.0.{cell_index + 1}:3000{cell_index}"))
        cell._mark_alive()
        srv.server_cells[meta.cell_id] = cell
    return srv


def _with_recording_router(events: list[dict[str, Any]]) -> Any:
    return patch.object(
        RolloutServer, "_router_api_client", property(lambda self: _RecordingRouterApiClient(events=events))
    )


class _RecordingRouterApiClient:
    def __init__(self, *, events: list[dict[str, Any]]) -> None:
        self._events = events

    async def add_worker(self, **kwargs: Any) -> None:
        self._events.append({"call": "add_worker", **kwargs})

    async def remove_worker(self, **kwargs: Any) -> None:
        self._events.append({"call": "remove_worker", **kwargs})
