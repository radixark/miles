from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.worker_provider.base import BaseWorkerProvider


class FakeBackendCapability(BackendCapability):
    def __init__(
        self,
        *,
        cells_provider: Any = None,
        static_provider: Any = None,
        cell_operations: Any = None,
    ) -> None:
        self.cells_provider = cells_provider
        self.static_provider = static_provider
        self.operations = cell_operations
        self.requested_pool_ids: list[list[str]] = []
        self.requested_static_pool_ids: list[str] = []

    def dynamic_worker_provider(self, *, pool_ids: Sequence[str]) -> BaseWorkerProvider:
        self.requested_pool_ids.append(list(pool_ids))
        assert self.cells_provider is not None, "this capability was built without a cells provider"
        return self.cells_provider

    def static_worker_provider(self, *, pool_id: str) -> BaseWorkerProvider:
        self.requested_static_pool_ids.append(pool_id)
        assert self.static_provider is not None, "this capability was built without a static provider"
        return self.static_provider

    def cell_operations(self) -> Any:
        assert self.operations is not None, "this capability was built without cell operations"
        return self.operations
