from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import server_cell as server_cell_module
from miles.ray.rollout.cell_state import CellAddrInfo, StateDisposed, StateServing
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata


def _make_meta(*, needs_offload: bool = False) -> ServerCellMetadata:
    return ServerCellMetadata(
        model_id="default",
        worker_type="regular",
        cell_id="inference-engine-0-0-0",
        num_gpus_per_engine=1,
        gpu_offset=0,
        sglang_api_key=None,
        worker_name="inference-engine-0-0-0-0",
        needs_offload=needs_offload,
        update_weights=True,
        workers_hash="pseudo-hash-0",
    )


class _StubProvider:
    async def get_addrs(self, worker_name: str):
        raise AssertionError(f"disposing a cell must not address it ({worker_name=})")


def _make_cell(*, router_api_client: MagicMock, needs_offload: bool = False, **args_overrides: object) -> ServerCell:
    return ServerCell(
        args=make_args(num_gpus_per_node=8, **args_overrides),
        meta=_make_meta(needs_offload=needs_offload),
        router_api_client=router_api_client,
        provider=_StubProvider(),
    )


def _make_router_api_client(*, remove_worker_side_effect: BaseException | None = None) -> MagicMock:
    client = MagicMock()
    client.add_worker = AsyncMock()
    client.remove_worker = AsyncMock(side_effect=remove_worker_side_effect)
    return client


async def _register(cell: ServerCell, *, state: type, server_url: str, bootstrap_port: int | None) -> None:
    addr_info = CellAddrInfo(server_url=server_url, bootstrap_port=bootstrap_port, gate_url=f"{server_url}/gate")
    await cell._register_with_router(addr_info=addr_info)
    cell._state = state(addr_info=addr_info)


class TestServerCellDispose:
    @pytest.mark.asyncio
    async def test_disposing_a_registered_cell_unregisters_it_from_the_router(self) -> None:
        """Leaving the url behind makes the router keep dialing a worker that no longer exists."""
        client = _make_router_api_client()
        cell = _make_cell(router_api_client=client)
        await _register(cell, state=StateServing, server_url="http://10.0.0.2:30000", bootstrap_port=None)

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
        await _register(cell, state=StateServing, server_url="http://10.0.0.3:30000", bootstrap_port=None)

        await cell.dispose()

        client.remove_worker.assert_awaited_once()
        assert isinstance(cell._state, StateDisposed)

    @pytest.mark.asyncio
    async def test_use_miles_router_pins_the_legacy_api_when_disposing(self) -> None:
        """--use-miles-router selects the query-string API on the dispose-time unregister too."""
        client = _make_router_api_client()
        cell = _make_cell(router_api_client=client, use_miles_router=True)
        await _register(cell, state=StateServing, server_url="http://10.0.0.4:30000", bootstrap_port=9000)

        await cell.dispose()

        assert client.remove_worker.await_args.kwargs["use_legacy_api"] is True

    @pytest.mark.asyncio
    async def test_disposing_a_registered_cell_twice_unregisters_it_once(self) -> None:
        """Teardown paths overlap, so a second dispose must not aim a removal at an entry the
        first one already took out."""
        client = _make_router_api_client()
        cell = _make_cell(router_api_client=client)
        await _register(cell, state=StateServing, server_url="http://10.0.0.6:30000", bootstrap_port=None)

        await cell.dispose()
        await cell.dispose()

        client.remove_worker.assert_awaited_once()


class TestServerCellRegisterRobustness:
    @pytest.mark.asyncio
    async def test_use_miles_router_pins_the_legacy_api_when_registering_too(self) -> None:
        """Adding through the wrong API shape lands the engine in a table the miles router never
        reads, so it is never routed to while removals still appear to work."""
        client = _make_router_api_client()
        cell = _make_cell(router_api_client=client, use_miles_router=True)

        await _register(cell, state=StateServing, server_url="http://10.0.0.2:30000", bootstrap_port=None)
        await cell.dispose()

        assert client.add_worker.await_args.kwargs["use_legacy_api"] is True

    @pytest.mark.asyncio
    async def test_a_router_that_never_answers_the_unregister_does_not_wedge_teardown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shared http client has no read timeout, so without the bound here an unresponsive
        router stalls dispose forever, and with it the whole reconcile and heal path."""
        never_answers = asyncio.Event()

        async def _hang(**_kwargs) -> None:
            await never_answers.wait()

        client = _make_router_api_client()
        client.remove_worker = _hang
        monkeypatch.setattr(server_cell_module, "SHUTDOWN_TIMEOUT", 0.05)
        cell = _make_cell(router_api_client=client)
        await _register(cell, state=StateServing, server_url="http://10.0.0.2:30000", bootstrap_port=None)

        await asyncio.wait_for(cell.dispose(), timeout=5.0)

        assert isinstance(cell._state, StateDisposed)
