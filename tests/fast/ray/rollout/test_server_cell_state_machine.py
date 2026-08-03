from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.cell_state import StatePendingWeights, StateServing
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata

SERVER_URL = "http://10.0.0.1:30000"


class _RouterApiClient:
    def __init__(self, add_worker_error: Exception | None = None):
        self.add_worker_calls: list[dict] = []
        self._add_worker_error = add_worker_error

    async def add_worker(self, **kwargs) -> None:
        self.add_worker_calls.append(kwargs)
        if self._add_worker_error is not None:
            raise self._add_worker_error


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


def _make_pending_weights_cell(router_api_client: _RouterApiClient) -> ServerCell:
    cell = ServerCell(args=make_args(), meta=_make_meta(), router_api_client=router_api_client)
    cell._mark_pending_weights(server_url=SERVER_URL, bootstrap_port=None)
    return cell


async def test_a_cell_becomes_serving_only_after_the_router_accepted_it():
    """The router must know the worker before anything treats the cell as servable."""
    router_api_client = _RouterApiClient()
    cell = _make_pending_weights_cell(router_api_client)

    await cell.mark_weights_ready()

    assert [call["worker_url"] for call in router_api_client.add_worker_calls] == [SERVER_URL]
    assert isinstance(cell._state, StateServing)


async def test_a_cell_stays_pending_weights_when_the_router_rejects_the_registration():
    """Marking it serving on a failed add_worker would strand the cell: never registered, never retried."""
    router_api_client = _RouterApiClient(add_worker_error=RuntimeError("router returned 503"))
    cell = _make_pending_weights_cell(router_api_client)

    with pytest.raises(RuntimeError, match="router returned 503"):
        await cell.mark_weights_ready()

    assert isinstance(cell._state, StatePendingWeights)
    assert cell.is_pending_weights


async def test_a_failed_registration_can_be_retried_by_a_later_end_update_weights():
    """Because the cell is still pending, the next weight window registers it instead of skipping it."""
    router_api_client = _RouterApiClient(add_worker_error=RuntimeError("router returned 503"))
    cell = _make_pending_weights_cell(router_api_client)

    with pytest.raises(RuntimeError):
        await cell.mark_weights_ready()

    router_api_client._add_worker_error = None
    await cell.mark_weights_ready()

    assert len(router_api_client.add_worker_calls) == 2
    assert isinstance(cell._state, StateServing)
