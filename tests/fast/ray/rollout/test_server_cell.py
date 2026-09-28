from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_args, track_server_cell

from miles.ray.rollout import server_cell as server_cell_module
from miles.ray.rollout.cell_state import CellAddrInfo, StatePendingWeights
from miles.ray.rollout.server_cell import ABORT_REQUEST_TIMEOUT_SECONDS, ServerCell, ServerCellMetadata

pytestmark = pytest.mark.usefixtures("dispose_tracked_server_cells")


class TestServerCellAbortAll:
    async def test_abort_all_forwards_the_bounded_request_to_the_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An addressable cell forwards the abort budget and returns its engine's result."""
        result = object()
        timeouts: list[float | None] = []

        class _RecordingApiClient:
            def __init__(self, server_url: str, api_key: str | None = None) -> None:
                self.server_url = server_url
                self.api_key = api_key

            async def abort_all_requests(self, timeout: float | None = None) -> object:
                timeouts.append(timeout)
                return result

        monkeypatch.setattr(server_cell_module, "SGLangApiClient", _RecordingApiClient)
        cell = track_server_cell(
            ServerCell(
                args=make_args(),
                meta=ServerCellMetadata(
                    model_id="default",
                    worker_type="regular",
                    cell_id="inference-engine-0-0-0",
                    num_gpus_per_engine=1,
                    gpu_offset=0,
                    sglang_api_key="secret",
                    worker_name="inference-engine-0-0-0-0",
                    needs_offload=False,
                    update_weights=True,
                    workers_hash="pseudo-hash-0",
                ),
                router_api_client=None,
                provider=None,
            )
        )
        cell._state = StatePendingWeights(
            addr_info=CellAddrInfo(
                server_url="http://10.0.0.1:30000",
                bootstrap_port=None,
                gate_url=None,
            )
        )

        actual = await cell.abort_all()

        assert actual is result
        assert timeouts == [ABORT_REQUEST_TIMEOUT_SECONDS]
