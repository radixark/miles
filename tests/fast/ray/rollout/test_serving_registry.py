from unittest.mock import AsyncMock

import pytest
from tests.fast.ray.rollout.conftest import make_args, track_server_cell
from tests.fast.ray.rollout.test_server_cell_state_machine import _make_meta, _StubProvider

from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.ray.rollout.cell_state import CellAddrInfo, StateDisposed, StatePendingWeights, StateServing
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell
from miles.ray.rollout.serving_registry import SGLangServingRegistry
from miles.utils.context_lock import ContextLock

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("dispose_tracked_server_cells")]


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("worker_type,bootstrap", [("regular", None), ("prefill", 1234), ("decode", None)])
async def test_default_adapter_preserves_router_contract(legacy, worker_type, bootstrap):
    client = AsyncMock()
    registry = SGLangServingRegistry(client=client, use_legacy_api=legacy)
    await registry.register(worker_url="http://engine:30000", worker_type=worker_type, bootstrap_port=bootstrap)
    await registry.unregister(worker_url="http://engine:30000")
    client.add_worker.assert_awaited_once_with(
        worker_url="http://engine:30000",
        worker_type=worker_type,
        bootstrap_port=bootstrap,
        use_legacy_api=legacy,
    )
    client.remove_worker.assert_awaited_once_with(worker_url="http://engine:30000", use_legacy_api=legacy)


def _cell(registry):
    cell = track_server_cell(
        ServerCell(
            args=make_args(),
            meta=_make_meta(),
            router_api_client=AsyncMock(),
            provider=_StubProvider(),
            serving_registry=registry,
        )
    )
    cell._state = StatePendingWeights(
        addr_info=CellAddrInfo(
            server_url="http://engine:30000",
            bootstrap_port=None,
            gate_url=None,
        )
    )
    return cell


async def test_injected_registry_owns_registration_and_cleanup():
    registry = AsyncMock()
    cell = _cell(registry)
    await cell.mark_weights_ready()
    assert isinstance(cell._state, StateServing)
    registry.register.assert_awaited_once_with(
        worker_url="http://engine:30000",
        worker_type=WorkerType.REGULAR,
        bootstrap_port=None,
    )
    await cell.dispose()
    registry.unregister.assert_awaited_once_with(worker_url="http://engine:30000")
    assert isinstance(cell._state, StateDisposed)
    cell.router_api_client.add_worker.assert_not_called()
    cell.router_api_client.remove_worker.assert_not_called()


async def test_failed_registration_keeps_pending_state():
    registry = AsyncMock()
    registry.register.side_effect = RuntimeError("not ready")
    cell = _cell(registry)
    with pytest.raises(RuntimeError, match="not ready"):
        await cell.mark_weights_ready()
    assert isinstance(cell._state, StatePendingWeights)
    await cell.dispose()
    registry.unregister.assert_not_called()


async def test_unregister_failure_preserves_existing_dispose_policy():
    registry = AsyncMock()
    registry.unregister.side_effect = RuntimeError("unreachable")
    cell = _cell(registry)
    await cell.mark_weights_ready()
    await cell.dispose()
    assert isinstance(cell._state, StateDisposed)


async def test_rollout_server_passes_registry_to_cells():
    registry = AsyncMock()
    server = RolloutServer(
        server_cells={},
        args=make_args(colocate=True),
        context_lock=ContextLock("registry-test"),
        engine_provider=_StubProvider(),
        serving_registry=registry,
    )
    async with server.context_lock:
        meta = _make_meta(needs_offload=True)
        await server.add_cell(meta)
        try:
            assert server.server_cells[meta.cell_id].serving_registry is registry
        finally:
            await server.dispose()
