from __future__ import annotations

import pytest

from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider


class _Handle(BaseWorkerHandle):
    def __init__(self, worker_name: str) -> None:
        self.worker_name = worker_name

    async def wait_ready(self, *, timeout: float) -> None: ...

    async def _probe_is_dead(self) -> bool:
        return False


class _RecordingProvider(BaseWorkerProvider):
    def __init__(self, *worker_names: str, worker_class: str | None = "pkg.Worker") -> None:
        self.requested_cell_ids: list[list[str]] = []
        self._infos = [
            WorkerInfo(name=name, generation=0, self_addrs={}, gpu_ids=[], worker_class=worker_class)
            for name in worker_names
        ]

    async def get_addrs(self, worker_name: str):
        raise NotImplementedError

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        self.requested_cell_ids.append(cell_ids)
        return [self._infos]

    async def watch_cells(self, reconcile):
        raise NotImplementedError

    def _build_handle_of_worker_info(self, info: WorkerInfo) -> BaseWorkerHandle | None:
        return _Handle(info.name) if info.worker_class is not None else None


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

        assert handle.worker_name == "trainer-engine-actor-0-1"

    def test_a_worker_the_cell_does_not_hold_is_rejected(self):
        """Answering with a handle to some other worker would silently drive the wrong process."""
        provider = _RecordingProvider("trainer-engine-actor-0-0")

        with pytest.raises(AssertionError, match="worker_name="):
            provider.get_handle("trainer-engine-actor-0-1")


class TestTheHandlesOfACell:
    def test_a_worker_that_is_only_launched_has_no_handle(self):
        """An sglang engine is started by the run and called over its own http api, never as a worker."""
        provider = _RecordingProvider("inference-engine-0-0", worker_class=None)

        assert provider.get_handles_of_worker_infos(provider._infos) == {}
