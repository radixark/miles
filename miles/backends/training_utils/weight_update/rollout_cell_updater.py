import logging
from collections.abc import Coroutine, Sequence
from concurrent.futures import Future
from typing import Any

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils import async_utils

logger = logging.getLogger(__name__)


class _RolloutCellUpdater:
    def __init__(self, cell_id: str, api_client: SGLangApiClient) -> None:
        self.cell_id = cell_id
        self._api_client = api_client
        self._error: BaseException | None = None

    @property
    def is_errored(self) -> bool:
        return self._error is not None

    def mark_errored(self, error: BaseException) -> None:
        if self._error is not None:
            logger.warning(f"rollout cell {self.cell_id} failed again, keeping the first error", exc_info=error)
            return
        self._error = error
        logger.error(f"rollout cell {self.cell_id} can no longer be updated", exc_info=error)

    def submit_client_call(self, name: str, **kwargs: Any) -> Future[Any]:
        if self.is_errored:
            done: Future[Any] = Future()
            done.set_result(None)
            return done
        return async_utils.submit(self._run_guarded(getattr(self._api_client, name)(**kwargs)))

    async def _run_guarded(self, request: Coroutine[Any, Any, Any]) -> object | None:
        try:
            return await request
        except Exception as error:
            self.mark_errored(error)
            return None


def create_rollout_cell_updaters(
    rollout_engines: Sequence[SGLangApiClient], engine_cell_ids: Sequence[str]
) -> dict[str, _RolloutCellUpdater]:
    return {
        cell_id: _RolloutCellUpdater(cell_id=cell_id, api_client=api_client)
        for api_client, cell_id in zip(rollout_engines, engine_cell_ids, strict=True)
    }
