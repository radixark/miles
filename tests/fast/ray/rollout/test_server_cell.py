from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.fast.ray.rollout.conftest import fake_actor_handle, make_args

from miles.ray.rollout.cell_state import AddrInfo
from miles.ray.rollout.rollout_server import RolloutServer, get_cell_indexer_of_id_map
from miles.ray.rollout.server_cell import ServerCell


def _allocated_cell(num_nodes: int = 1, *, alive: bool = True, addressed: bool = True) -> ServerCell:
    cell = ServerCell(num_nodes=num_nodes, args=make_args(num_gpus_per_node=8), worker_type="regular")
    cell._mark_allocated_uninitialized([fake_actor_handle() for _ in range(num_nodes)])
    if not addressed:
        return cell
    cell._mark_addressing([AddrInfo(server_url=f"http://10.0.0.{i + 1}:3000{i}") for i in range(num_nodes)])
    if alive:
        cell._mark_alive()
    return cell


class TestServerCellState:
    def test_a_fresh_cell_is_stopped(self):
        """A cell owns one state machine for all of its node-ranks."""
        cell = ServerCell(num_nodes=2, args=make_args(num_gpus_per_node=8), worker_type="regular")
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
        assert cell.addr_info is cell.addr_infos[0]
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
        cell._mark_addressing([AddrInfo(server_url="http://10.0.0.9:39999")])
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


def _addressed_cell(
    *, worker_type: str = "regular", bootstrap_port: int | None = None, **args_overrides
) -> ServerCell:
    cell = ServerCell(args=make_args(num_gpus_per_node=8, **args_overrides), worker_type=worker_type, num_nodes=2)
    cell._mark_allocated_uninitialized([fake_actor_handle() for _ in range(2)])
    cell._mark_addressing(
        [
            AddrInfo(server_url=f"http://10.0.0.{index + 1}:3000{index}", bootstrap_port=bootstrap_port)
            for index in range(2)
        ]
    )
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


def _build_servers(
    *, num_servers: int = 1, engines_per_server: int = 2, num_gpus_per_engine: int = 1
) -> dict[str, RolloutServer]:
    args = make_args(num_gpus_per_node=8)
    nodes_per_engine = max(1, num_gpus_per_engine // 8)
    servers: dict[str, RolloutServer] = {}
    for s_idx in range(num_servers):
        cells = [_allocated_cell(num_nodes=nodes_per_engine) for _ in range(engines_per_server // nodes_per_engine)]
        for cell in cells:
            cell.num_gpus_per_engine = num_gpus_per_engine
        servers[f"model_{s_idx}"] = RolloutServer(
            server_cells=cells,
            args=args,
            model_name=f"model_{s_idx}",
            update_weights=True,
        )
    return servers


class TestGetCellIndexerOfIdMap:
    def test_single_server_one_cell_per_engine(self):
        """Happy path: one server with N engines → N cells, each cell_index=i, all under model_0."""
        servers = _build_servers(num_servers=1, engines_per_server=3)
        cells = get_cell_indexer_of_id_map(servers)
        assert len(cells) == 3
        for i, cell in enumerate(cells):
            assert cell.srv_key == "model_0"
            assert cell.cell_index == i

    def test_multi_server_ordered_by_key_alphabetically(self):
        """When multiple servers exist, cells are emitted in srv_key order."""
        servers = _build_servers(num_servers=2, engines_per_server=1)
        cells = get_cell_indexer_of_id_map(servers)
        srv_keys_in_order = [c.srv_key for c in cells]
        assert srv_keys_in_order == sorted(srv_keys_in_order)
        assert srv_keys_in_order == ["model_0", "model_1"]

    def test_multinode_engine_slots_form_one_cell(self):
        """num_gpus_per_engine=16 and num_gpus_per_node=8 → nodes_per_engine=2;
        the 2 engine slots form one cell."""
        servers = _build_servers(num_servers=1, engines_per_server=2, num_gpus_per_engine=16)
        cells = get_cell_indexer_of_id_map(servers)
        assert len(cells) == 1
        assert cells[0].cell_index == 0

    def test_server_without_cells_emits_zero_cells(self):
        """A server with no cells (e.g. only placeholder groups) emits no cell ids."""
        srv = MagicMock()
        srv.server_cells = []
        out = get_cell_indexer_of_id_map({"only": srv})
        assert out == []

    def test_empty_server_dict_returns_empty_list(self):
        assert get_cell_indexer_of_id_map({}) == []
