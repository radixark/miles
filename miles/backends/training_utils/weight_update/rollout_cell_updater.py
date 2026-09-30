from collections.abc import Sequence
from concurrent.futures import Future
from typing import Any

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils import async_utils


class _RolloutCellUpdater:
    def __init__(self, cell_id: str, api_client: SGLangApiClient) -> None:
        self.cell_id = cell_id
        self._api_client = api_client

    def submit_client_call(self, name: str, **kwargs: Any) -> Future[Any]:
        return async_utils.submit(getattr(self._api_client, name)(**kwargs))


def create_rollout_cell_updaters(
    rollout_engines: Sequence[SGLangApiClient], engine_cell_ids: Sequence[str]
) -> dict[str, _RolloutCellUpdater]:
    return {
        cell_id: _RolloutCellUpdater(cell_id=cell_id, api_client=api_client)
        for api_client, cell_id in zip(rollout_engines, engine_cell_ids, strict=True)
    }
