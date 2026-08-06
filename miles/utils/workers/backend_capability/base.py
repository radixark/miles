from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miles.utils.workers.cell_operations.base import BaseCellOperations
    from miles.utils.workers.worker_provider.base import BaseWorkerProvider


class BackendCapability(abc.ABC):
    @abc.abstractmethod
    def dynamic_worker_provider(self, *, pool_ids: Sequence[str]) -> BaseWorkerProvider: ...

    @abc.abstractmethod
    def static_worker_provider(self, *, pool_id: str) -> BaseWorkerProvider: ...

    @abc.abstractmethod
    def cell_operations(self) -> BaseCellOperations: ...


class DeferredBackendCapability(BackendCapability):
    def __init__(self, *, create: Callable[[], BackendCapability]) -> None:
        self._create = create
        self._inner: BackendCapability | None = None

    def dynamic_worker_provider(self, *, pool_ids: Sequence[str]) -> BaseWorkerProvider:
        return self._resolve().dynamic_worker_provider(pool_ids=pool_ids)

    def static_worker_provider(self, *, pool_id: str) -> BaseWorkerProvider:
        return self._resolve().static_worker_provider(pool_id=pool_id)

    def cell_operations(self) -> BaseCellOperations:
        return self._resolve().cell_operations()

    def _resolve(self) -> BackendCapability:
        if self._inner is None:
            self._inner = self._create()
        return self._inner
