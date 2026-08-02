from __future__ import annotations

import asyncio
from argparse import Namespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.fast.ray.rollout.conftest import fake_actor_handle, fake_engine, make_args

import miles.ray.rollout.server_cell as server_cell_module
from miles.ray.rollout.cell_state import AddrInfo
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.test_utils.mock_sglang_engine import parse_cmd_flags
from miles.utils.workers.addr_allocator import PortAllocator


def _allocated_cell(num_nodes: int = 1, *, alive: bool = True, addressed: bool = True) -> ServerCell:
    cell = ServerCell(
        num_nodes=num_nodes, args=make_args(num_gpus_per_node=8), worker_type="regular", cell_id="cell-0"
    )
    cell._mark_allocated_uninitialized([fake_actor_handle() for _ in range(num_nodes)])
    if not addressed:
        return cell
    cell._mark_addressing(AddrInfo(server_url="http://10.0.0.1:30000"))
    if alive:
        cell._mark_alive()
    return cell


class TestEngineGpuIds:
    def test_offsets_and_stride_follow_the_cell_layout(self):
        """The driver-side gpu layout must match what the launch handed each actor."""
        cell = ServerCell(
            args=make_args(num_gpus_per_node=8),
            worker_type="regular",
            cell_id="cell-0",
            pg=(None, [], [0, 1, 2, 3, 4, 5, 6, 7]),
            num_gpus_per_engine=2,
            gpu_offset=4,
        )
        assert cell.engine_gpu_ids == [[4, 5]]

    def test_each_node_rank_of_a_multi_node_engine_covers_its_node(self):
        """A 2-node engine reports one whole-node gpu range per node-rank."""
        cell = ServerCell(
            args=make_args(num_gpus_per_node=8),
            worker_type="regular",
            cell_id="cell-0",
            pg=(None, [], list(range(8)) + list(range(8))),
            num_nodes=2,
            num_gpus_per_engine=16,
        )
        assert cell.engine_gpu_ids == [list(range(8)), list(range(8))]


class TestServerCellState:
    def test_a_fresh_cell_is_stopped(self):
        """A cell owns one state machine for all of its node-ranks."""
        cell = ServerCell(num_nodes=2, args=make_args(num_gpus_per_node=8), worker_type="regular", cell_id="cell-0")
        assert not cell.is_allocated
        assert not cell.is_alive

    def test_allocating_covers_every_node_rank(self):
        """The cell's actors are the node-ranks of one engine, so they are held together."""
        cell = _allocated_cell(num_nodes=2, alive=False)
        assert cell.is_allocated and not cell.is_alive
        assert len(cell.actor_handles) == 2
        assert cell.primary_actor_handle is cell.actor_handles[0]

    def test_the_primary_addr_is_the_router_visible_one(self):
        """Only node 0 serves the endpoint the router routes to."""
        cell = _allocated_cell(num_nodes=2)
        assert cell.is_alive
        assert cell.addr_info.server_url == "http://10.0.0.1:30000"
        assert cell.api_client.server_url == "http://10.0.0.1:30000"

    def test_stopping_releases_the_whole_cell(self):
        """Teardown is whole-cell: no node-rank may outlive the engine."""
        cell = _allocated_cell(num_nodes=2)
        cell._mark_stopped()
        assert not cell.is_allocated
        assert not cell.is_alive

    def test_the_api_client_is_unavailable_before_the_url_is_known(self):
        """An allocated but unaddressed cell has no endpoint to talk to yet."""
        cell = _allocated_cell(num_nodes=2, addressed=False)
        with pytest.raises(AssertionError):
            _ = cell.api_client

    def test_going_alive_requires_an_addr(self):
        """A cell must not be reported alive before it knows its own url."""
        cell = _allocated_cell(num_nodes=2, addressed=False)
        with pytest.raises(AssertionError):
            cell._mark_alive()

    def test_restarting_replaces_the_addr(self):
        """A restarted cell must serve on its new endpoint, not the dead one."""
        cell = _allocated_cell(num_nodes=1)
        assert cell.api_client.server_url == "http://10.0.0.1:30000"

        cell._mark_stopped()
        cell._mark_allocated_uninitialized([fake_actor_handle()])
        cell._mark_addressing(AddrInfo(server_url="http://10.0.0.9:39999"))
        cell._mark_alive()

        assert cell.api_client.server_url == "http://10.0.0.9:39999"


class TestServerCellApiCalls:
    async def test_offload_releases_memory_on_the_primary_engine_only(self):
        """Non-primary node-ranks are workers without their own HTTP endpoint."""
        cell = _allocated_cell(num_nodes=2)
        client = MagicMock()
        client.release_memory_occupation = AsyncMock(return_value="released")
        with patch.object(ServerCell, "api_client", property(lambda self: client)):
            assert await cell.offload(tags=["weights"]) == "released"
        client.release_memory_occupation.assert_awaited_once_with(tags=["weights"])

    async def test_onload_resumes_memory_on_the_primary_engine_only(self):
        """Non-primary node-ranks are workers without their own HTTP endpoint."""
        cell = _allocated_cell(num_nodes=2)
        client = MagicMock()
        client.resume_memory_occupation = AsyncMock(return_value="resumed")
        with patch.object(ServerCell, "api_client", property(lambda self: client)):
            assert await cell.onload(tags=None) == "resumed"
        client.resume_memory_occupation.assert_awaited_once_with(tags=None)

    async def test_check_weights_forwards_all_arguments_to_the_primary_engine(self):
        """The whole keyword set must reach the engine api unchanged."""
        cell = _allocated_cell()
        client = MagicMock()
        client.check_weights = AsyncMock(return_value={"ok": True})
        with patch.object(ServerCell, "api_client", property(lambda self: client)):
            result = await cell.check_weights(
                action="report", allow_quant_error=True, selector="first", skip_list=["a"]
            )
        assert result == {"ok": True}
        client.check_weights.assert_awaited_once_with(
            action="report", allow_quant_error=True, selector="first", skip_list=["a"]
        )


def _launch_command_flags(*, rank_offset: int, num_nodes: int) -> list[dict[str, Any]]:
    num_gpus_per_engine: int = 8 * num_nodes
    actors: list[MagicMock] = [fake_engine(host=f"10.0.0.{index + 1}", port_seed=30000) for index in range(num_nodes)]
    for actor in actors:
        actor.run.remote.side_effect = lambda **kwargs: asyncio.sleep(0)

    cell = ServerCell(
        args=make_args(
            num_gpus_per_node=8,
            sglang_pp_size=1,
            sglang_ep_size=1,
            multi_lora=False,
            rollout_num_gpus_per_engine=num_gpus_per_engine,
        ),
        worker_type="regular",
        cell_id="cell-1",
        num_nodes=num_nodes,
        num_gpus_per_engine=num_gpus_per_engine,
        rank_offset=rank_offset,
        pg=(None, [], list(range(8)) * num_nodes),
    )

    pending: list[MagicMock] = list(actors)
    with (
        patch.object(server_cell_module, "launch_sglang_ray_actor", side_effect=lambda **kwargs: pending.pop(0)),
        patch.object(server_cell_module, "wait_server_healthy", new=AsyncMock()),
    ):
        asyncio.run(cell.start_engines(PortAllocator()))

    return [parse_cmd_flags(actor.run.remote.call_args.kwargs["cmd"]) for actor in actors]


class TestMultiNodeEngineNodeRank:
    def test_the_second_two_node_cell_numbers_its_own_nodes_from_zero(self, patch_ray_get):
        """--node-rank is cell-local, so the cell at rank_offset=2 launches node-ranks 0 and 1, not 2 and 3."""
        flags = _launch_command_flags(rank_offset=2, num_nodes=2)
        assert [entry["nnodes"] for entry in flags] == [2, 2]
        assert [entry.get("node_rank", 0) for entry in flags] == [0, 1]


def _startup_meta(*, needs_offload: bool, update_weights: bool) -> ServerCellMetadata:
    return ServerCellMetadata(
        model_id="default",
        worker_type="regular",
        cell_id="cell-0",
        num_gpus_per_engine=1,
        gpu_offset=0,
        sglang_api_key=None,
        worker_name="worker-0",
        needs_offload=needs_offload,
        update_weights=update_weights,
        workers_hash="hash-0",
    )


async def _run_add(
    *,
    needs_offload: bool = True,
    update_weights: bool = True,
    check_weight_update_equal: bool = True,
) -> list[str]:
    args = Namespace(
        rollout_external=False,
        use_miles_router=False,
        check_weight_update_equal=check_weight_update_equal,
        check_weight_update_skip_list=[],
    )
    cell = ServerCell(args=args, meta=_startup_meta(needs_offload=needs_offload, update_weights=update_weights))

    calls: list[str] = []
    client = MagicMock()
    client.check_weights = AsyncMock(side_effect=lambda **kwargs: calls.append(f"check_weights:{kwargs['action']}"))
    client.release_memory_occupation = AsyncMock(side_effect=lambda **kwargs: calls.append("release"))
    client.resume_memory_occupation = AsyncMock(side_effect=lambda **kwargs: calls.append("resume"))

    async def _fake_add_raw(self: ServerCell) -> None:
        self._mark_addressing(AddrInfo(server_url="http://10.0.0.1:30000"))

    router_api_client = MagicMock()
    router_api_client.add_worker = AsyncMock()

    with (
        patch.object(ServerCell, "api_client", property(lambda self: client)),
        patch.object(ServerCell, "_add_raw", _fake_add_raw),
    ):
        await cell.add(router_api_client)

    assert cell.is_alive
    router_api_client.add_worker.assert_awaited_once()
    return calls


class TestServerCellStartupSequence:
    async def test_the_weight_baseline_is_snapshotted_before_memory_is_released(self):
        """The checker's baseline must be the freshly loaded checkpoint, not the remapped weight storage."""
        assert await _run_add() == ["check_weights:snapshot", "release", "resume"]

    async def test_a_colocated_cell_releases_all_memory_and_resumes_only_the_weights(self):
        """Startup memory handover for an offloading cell lives on the cell, not on the controller."""
        assert await _run_add(check_weight_update_equal=False) == ["release", "resume"]

    async def test_a_resident_cell_touches_neither_memory_nor_the_checker(self):
        """A cell that never offloads keeps its memory, so startup issues no engine calls."""
        assert await _run_add(needs_offload=False, check_weight_update_equal=False) == []

    async def test_a_resident_cell_still_snapshots_its_weight_baseline(self):
        """The baseline is about the checkpoint, so it is taken whether or not the cell offloads."""
        assert await _run_add(needs_offload=False) == ["check_weights:snapshot"]

    async def test_a_frozen_model_gets_no_baseline(self):
        """A model that never receives weight updates would always mismatch the checker."""
        assert await _run_add(update_weights=False) == ["release", "resume"]

    async def test_the_snapshot_is_taken_over_the_whole_model_without_a_skip_list(self):
        """The baseline must match what the controller-side reset and comparison later cover."""
        calls: list[dict] = []
        args = Namespace(
            rollout_external=False,
            use_miles_router=False,
            check_weight_update_equal=True,
            check_weight_update_skip_list=["lm_head"],
        )
        cell = ServerCell(args=args, meta=_startup_meta(needs_offload=True, update_weights=True))

        client = MagicMock()
        client.check_weights = AsyncMock(side_effect=lambda **kwargs: calls.append(kwargs))
        client.release_memory_occupation = AsyncMock()
        client.resume_memory_occupation = AsyncMock()

        async def _fake_add_raw(self: ServerCell) -> None:
            self._mark_addressing(AddrInfo(server_url="http://10.0.0.1:30000"))

        router_api_client = MagicMock()
        router_api_client.add_worker = AsyncMock()

        with (
            patch.object(ServerCell, "api_client", property(lambda self: client)),
            patch.object(ServerCell, "_add_raw", _fake_add_raw),
        ):
            await cell.add(router_api_client)

        assert calls == [{"action": "snapshot", "allow_quant_error": False, "selector": "all", "skip_list": None}]


def _addressed_cell(
    *, worker_type: str = "regular", bootstrap_port: int | None = None, **args_overrides
) -> ServerCell:
    cell = ServerCell(
        args=make_args(num_gpus_per_node=8, **args_overrides),
        worker_type=worker_type,
        num_nodes=2,
        cell_id="cell-0",
    )
    cell._mark_allocated_uninitialized([fake_actor_handle() for _ in range(2)])
    cell._mark_addressing(AddrInfo(server_url="http://10.0.0.1:30000", bootstrap_port=bootstrap_port))
    cell._mark_alive()
    return cell


class TestServerCellRouterMembership:
    async def test_register_publishes_the_primary_engine_url_and_worker_type(self):
        """The router routes to the cell through its node-0 engine only."""
        client = MagicMock()
        client.add_worker = AsyncMock()
        await _addressed_cell().register(client)
        client.add_worker.assert_awaited_once_with(
            worker_url="http://10.0.0.1:30000",
            worker_type="regular",
            use_legacy_api=False,
            bootstrap_port=None,
        )

    async def test_register_passes_the_bootstrap_port_of_a_prefill_worker(self):
        """PD disaggregation needs the decode side to dial this port."""
        client = MagicMock()
        client.add_worker = AsyncMock()
        await _addressed_cell(worker_type="prefill", bootstrap_port=8998).register(client)
        assert client.add_worker.await_args.kwargs["worker_type"] == "prefill"
        assert client.add_worker.await_args.kwargs["bootstrap_port"] == 8998

    async def test_unregister_removes_the_same_url_register_published(self):
        """A mismatch would leave the router routing to a dead worker."""
        client = MagicMock()
        client.remove_worker = AsyncMock()
        await _addressed_cell().unregister(client)
        client.remove_worker.assert_awaited_once_with(worker_url="http://10.0.0.1:30000", use_legacy_api=False)

    async def test_use_miles_router_pins_the_legacy_api_on_both_calls(self):
        """--use-miles-router selects the query-string API for register and unregister alike."""
        client = MagicMock()
        client.add_worker = AsyncMock()
        client.remove_worker = AsyncMock()
        cell = _addressed_cell(use_miles_router=True)
        await cell.register(client)
        await cell.unregister(client)
        assert client.add_worker.await_args.kwargs["use_legacy_api"] is True
        assert client.remove_worker.await_args.kwargs["use_legacy_api"] is True
