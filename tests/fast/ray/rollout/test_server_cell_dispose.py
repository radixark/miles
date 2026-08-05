from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.cell_state import CellAddrInfo, StateDisposed, StatePendingWeights, StateServing
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata


def _make_meta() -> ServerCellMetadata:
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


def _make_cell(*, router_api_client: MagicMock, **args_overrides: object) -> ServerCell:
    return ServerCell(
        args=make_args(num_gpus_per_node=8, **args_overrides),
        meta=_make_meta(),
        router_api_client=router_api_client,
    )


def _make_router_api_client(*, remove_worker_side_effect: BaseException | None = None) -> MagicMock:
    client = MagicMock()
    client.remove_worker = AsyncMock(side_effect=remove_worker_side_effect)
    return client


def _mark_serving(cell: ServerCell, *, server_url: str, bootstrap_port: int | None) -> None:
    """Serving is the only state in which the router holds the cell's url.

    Both registration sites in ServerCell mark the cell serving in the same breath, so a cell
    observed in StatePendingWeights was never added to the router and has nothing to remove.
    """
    cell._state = StateServing(
        addr_info=CellAddrInfo(server_url=server_url, bootstrap_port=bootstrap_port, gate_url=f"{server_url}/gate")
    )


def _mark_pending_weights(cell: ServerCell, *, server_url: str, bootstrap_port: int | None) -> None:
    """A cell whose add_worker was awaited, or whose add_worker failed, sits here holding a url."""
    cell._state = StatePendingWeights(
        addr_info=CellAddrInfo(server_url=server_url, bootstrap_port=bootstrap_port, gate_url=f"{server_url}/gate")
    )


class TestServerCellDispose:
    @pytest.mark.asyncio
    async def test_disposing_a_registered_cell_unregisters_it_from_the_router(self) -> None:
        """Leaving the url behind makes the router keep dialing a worker that no longer exists."""
        client = _make_router_api_client()
        cell = _make_cell(router_api_client=client)
        _mark_serving(cell, server_url="http://10.0.0.2:30000", bootstrap_port=None)

        await cell.dispose()

        client.remove_worker.assert_awaited_once_with(worker_url="http://10.0.0.2:30000", use_legacy_api=False)
        assert isinstance(cell._state, StateDisposed)

    @pytest.mark.asyncio
    async def test_disposing_a_cell_that_never_registered_never_touches_the_router(self) -> None:
        """A cell that never reached a url has nothing to unregister."""
        client = _make_router_api_client()
        cell = _make_cell(router_api_client=client)

        await cell.dispose()

        client.remove_worker.assert_not_awaited()
        assert isinstance(cell._state, StateDisposed)

    @pytest.mark.asyncio
    async def test_a_failing_unregister_still_tears_the_cell_down(self) -> None:
        """Unregistering is idempotent cleanup, so its failure must not block disposal."""
        client = _make_router_api_client(remove_worker_side_effect=RuntimeError("injected remove failure"))
        cell = _make_cell(router_api_client=client)
        _mark_serving(cell, server_url="http://10.0.0.3:30000", bootstrap_port=None)

        await cell.dispose()

        client.remove_worker.assert_awaited_once()
        assert isinstance(cell._state, StateDisposed)

    @pytest.mark.asyncio
    async def test_use_miles_router_pins_the_legacy_api_when_disposing(self) -> None:
        """--use-miles-router selects the query-string API on the dispose-time unregister too."""
        client = _make_router_api_client()
        cell = _make_cell(router_api_client=client, use_miles_router=True)
        _mark_serving(cell, server_url="http://10.0.0.4:30000", bootstrap_port=9000)

        await cell.dispose()

        assert client.remove_worker.await_args.kwargs["use_legacy_api"] is True

    @pytest.mark.asyncio
    async def test_disposing_a_pending_weights_cell_unregisters_it_from_the_router(self) -> None:
        """mark_weights_ready awaits add_worker while the cell is still pending, and a cell whose
        add_worker failed stays pending on purpose, so the router can already hold its url."""
        client = _make_router_api_client()
        cell = _make_cell(router_api_client=client)
        _mark_pending_weights(cell, server_url="http://10.0.0.1:30000", bootstrap_port=None)

        await cell.dispose()

        client.remove_worker.assert_awaited_once_with(worker_url="http://10.0.0.1:30000", use_legacy_api=False)
        assert isinstance(cell._state, StateDisposed)
