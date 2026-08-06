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
        self.requested_spec_names: list[list[str]] = []
        self.requested_static_spec_names: list[str] = []

    def dynamic_worker_provider(self, *, spec_names: Sequence[str]) -> BaseWorkerProvider:
        self.requested_spec_names.append(list(spec_names))
        assert self.cells_provider is not None, "this capability was built without a cells provider"
        return self.cells_provider

    def static_worker_provider(self, *, spec_name: str) -> BaseWorkerProvider:
        self.requested_static_spec_names.append(spec_name)
        assert self.static_provider is not None, "this capability was built without a static provider"
        return self.static_provider

    def cell_operations(self) -> Any:
        assert self.operations is not None, "this capability was built without cell operations"
        return self.operations
