from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo


class _Handle(BaseWorkerHandle):
    def __init__(self, worker_name: str) -> None:
        self.worker_name = worker_name

    async def wait_ready(self, *, timeout: float) -> None: ...

    async def probe_is_dead(self) -> bool:
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


class TestExpectedNumCells:
    def test_the_default_provider_has_no_expected_cell_count(self) -> None:
        """A provider without fleet sizing logic leaves the expected cell count unspecified."""
        provider = _RecordingProvider()

        assert provider.expected_num_cells(group_id="inference") is None


class TestCellInfoWireModel:
    def test_cell_info_round_trips_through_its_strict_wire_model(self) -> None:
        """Cell descriptions preserve every typed wire field and reject undeclared fields."""
        cell_info = CellInfo(
            cell_id="trainer-engine-actor-00003",
            pool_id="trainer-engine-actor",
            alive=True,
            worker_names=[
                "trainer-engine-actor-00003-00000",
                "trainer-engine-actor-00003-00001",
            ],
            workers_hash="workers-sha256",
            meta={"attempt": 2, "draining": False, "labels": ["trainer", "primary"]},
        )

        dumped = cell_info.model_dump()
        restored = CellInfo.model_validate(dumped)

        assert dumped == {
            "cell_id": "trainer-engine-actor-00003",
            "pool_id": "trainer-engine-actor",
            "alive": True,
            "worker_names": [
                "trainer-engine-actor-00003-00000",
                "trainer-engine-actor-00003-00001",
            ],
            "workers_hash": "workers-sha256",
            "meta": {"attempt": 2, "draining": False, "labels": ["trainer", "primary"]},
        }
        assert restored == cell_info
        assert restored.meta["attempt"] == 2
        assert isinstance(restored.meta["attempt"], int)
        assert restored.meta["draining"] is False

        with pytest.raises(ValidationError, match="extra_forbidden"):
            CellInfo.model_validate({**dumped, "undeclared": "field"})


class TestGetHandle:
    def test_the_cell_is_derived_from_the_worker_name(self):
        """A worker name carries its pool and cell, so the provider must not be asked for anything else."""
        provider = _RecordingProvider("inference-controller-00000-00000")

        provider.get_handle("inference-controller-00000-00000")

        assert provider.requested_cell_ids == [["inference-controller-00000"]]

    def test_the_handle_of_the_named_worker_is_returned(self):
        """A cell holds several workers, so the one whose name matches must come back, not merely the first."""
        provider = _RecordingProvider("trainer-engine-actor-00000-00000", "trainer-engine-actor-00000-00001")

        handle = provider.get_handle("trainer-engine-actor-00000-00001")

        assert handle.worker_name == "trainer-engine-actor-00000-00001"

    def test_a_worker_the_cell_does_not_hold_is_rejected(self):
        """Answering with a handle to some other worker would silently drive the wrong process."""
        provider = _RecordingProvider("trainer-engine-actor-00000-00000")

        with pytest.raises(AssertionError, match="worker_name="):
            provider.get_handle("trainer-engine-actor-00000-00001")


class TestTheHandlesOfACell:
    def test_a_worker_that_is_only_launched_has_no_handle(self):
        """An sglang engine is started by the run and called over its own http api, never as a worker."""
        provider = _RecordingProvider("inference-engine-00000-00000", worker_class=None)

        assert provider.get_handles_of_worker_infos(provider._infos) == {}
