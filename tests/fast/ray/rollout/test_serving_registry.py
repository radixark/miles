from unittest.mock import AsyncMock

import pytest
from tests.fast.ray.rollout.conftest import make_args, track_server_cell

from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.ray.rollout.cell_state import CellAddrInfo, StateDisposed, StatePendingWeights, StateServing
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("dispose_tracked_server_cells")]


class _RecordingRegistry:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def register(self, *, worker_url: str, worker_type: WorkerType, bootstrap_port: int | None) -> None:
        self.calls.append(
            ("register", dict(worker_url=worker_url, worker_type=worker_type, bootstrap_port=bootstrap_port))
        )

    async def unregister(self, *, worker_url: str) -> None:
        self.calls.append(("unregister", dict(worker_url=worker_url)))


async def test_injected_registry_owns_registration_and_cleanup():
    registry = _RecordingRegistry()
    router = AsyncMock()
    cell = track_server_cell(
        ServerCell(
            args=make_args(),
            meta=ServerCellMetadata(
                model_id="default",
                worker_type=WorkerType.PREFILL,
                cell_id="cell-0",
                num_gpus_per_engine=1,
                gpu_offset=0,
                sglang_api_key=None,
                worker_name="cell-0-0",
                needs_offload=False,
                update_weights=True,
                workers_hash="test-hash",
            ),
            router_api_client=router,
            provider=AsyncMock(),
            serving_registry=registry,
        )
    )
    cell._state = StatePendingWeights(
        addr_info=CellAddrInfo(server_url="http://engine:30000", bootstrap_port=1234, gate_url=None)
    )

    await cell.mark_weights_ready()

    assert isinstance(cell._state, StateServing)
    assert registry.calls == [
        ("register", dict(worker_url="http://engine:30000", worker_type=WorkerType.PREFILL, bootstrap_port=1234))
    ]

    await cell.dispose()

    assert isinstance(cell._state, StateDisposed)
    assert registry.calls == [
        ("register", dict(worker_url="http://engine:30000", worker_type=WorkerType.PREFILL, bootstrap_port=1234)),
        ("unregister", dict(worker_url="http://engine:30000")),
    ]
    assert router.mock_calls == []
