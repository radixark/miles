from __future__ import annotations

import pytest

from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider


class _Handle(BaseWorkerHandle):
    async def wait_ready(self, *, timeout: float) -> None: ...

    async def _probe_is_dead(self) -> bool:
        return False


class _RecordingProvider(BaseWorkerProvider):
    def __init__(self, *worker_names: str) -> None:
        self.requested_cell_ids: list[list[str]] = []
        self._infos = [
            WorkerInfo(name=name, generation=0, self_addrs={}, gpu_ids=[], handle=_Handle()) for name in worker_names
        ]

    async def get_addrs(self, worker_name: str):
        raise NotImplementedError

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        self.requested_cell_ids.append(cell_ids)
        return [self._infos]

    async def watch_cells(self, reconcile):
        raise NotImplementedError


class TestGetHandle:
    def test_the_cell_is_derived_from_the_worker_name(self):
        """A worker name carries its pool and cell, so the provider must not be asked for anything else."""
        provider = _RecordingProvider("inference-controller-0-0")

        provider.get_handle("inference-controller-0-0")

        assert provider.requested_cell_ids == [["inference-controller-0"]]

    def test_the_handle_of_the_named_worker_is_returned(self):
        """A cell holds several workers, so the one whose name matches must come back, not merely the first."""
        provider = _RecordingProvider("trainer-engine-actor-0-0", "trainer-engine-actor-0-1")

        handle = provider.get_handle("trainer-engine-actor-0-1")

        assert handle is provider._infos[1].handle

    def test_a_worker_the_cell_does_not_hold_is_rejected(self):
        """Answering with a handle to some other worker would silently drive the wrong process."""
        provider = _RecordingProvider("trainer-engine-actor-0-0")

        with pytest.raises(AssertionError, match="worker_name="):
            provider.get_handle("trainer-engine-actor-0-1")
