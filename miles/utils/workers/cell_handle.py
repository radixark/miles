from __future__ import annotations

import abc
from typing import Any

from miles.utils.async_utils import CancellableTasks
from miles.utils.workers.worker_handle import BaseWorkerHandle


class BaseCellHandle(abc.ABC):
    """Calls the same method on every worker of one cell and answers their results in worker order."""

    @abc.abstractmethod
    async def call_each(self, method: str, /, kwargs_per_worker: list[dict[str, Any]]) -> list[Any]: ...


def cell_handle_of(worker_handles: list[BaseWorkerHandle]) -> BaseCellHandle:
    """One call that reaches every worker of a cell, in one mesh call where the transport has one."""
    mesh_handle = worker_handles[0].cell_handle_with(worker_handles) if worker_handles else None
    return mesh_handle or GatherCellHandle(worker_handles)


class GatherCellHandle(BaseCellHandle):
    """One call per worker handle, gathered: what every transport without a mesh call does."""

    def __init__(self, worker_handles: list[BaseWorkerHandle]) -> None:
        self._worker_handles = worker_handles

    async def call_each(self, method: str, /, kwargs_per_worker: list[dict[str, Any]]) -> list[Any]:
        assert len(kwargs_per_worker) == len(
            self._worker_handles
        ), f"{len(kwargs_per_worker)} sets of arguments for the {len(self._worker_handles)} workers of a cell"
        # a worker whose call fails cancels its siblings' calls rather than leaving them running into the next one
        return await CancellableTasks().gather(
            (
                getattr(handle, method)(**kwargs)  # config-access-exempt: attribute selected at runtime
                for handle, kwargs in zip(self._worker_handles, kwargs_per_worker, strict=True)
            ),
            timeout=None,
        )
