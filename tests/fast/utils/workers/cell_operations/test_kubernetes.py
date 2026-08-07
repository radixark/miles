import asyncio

import pytest

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations import kubernetes as cell_operations_kubernetes
from miles.utils.workers.cell_operations.kubernetes import KubernetesCellOperations
from miles.utils.workers.worker_provider.base import CellInfo


class FakeProvider:
    def __init__(self, infos: dict[str, CellInfo]) -> None:
        self._infos = infos
        self.watches = 0

    async def watch_cells(self, reconcile):
        self.watches += 1
        return _stop_watching

    def cell_ids(self) -> list[str]:
        return sorted(self._infos)

    def cell_info(self, cell_id: str) -> CellInfo | None:
        return self._infos.get(cell_id)

    def pod_names(self, cell_id: str) -> list[str]:
        info = self._infos.get(cell_id)
        return list(info.worker_names) if info is not None else []


def _info(cell_id="trainer-actor-0", pool_id="trainer-actor", workers=("trainer-actor-0-0",)):
    return CellInfo(
        cell_id=cell_id,
        pool_id=pool_id,
        alive=True,
        worker_names=list(workers),
        workers_hash="h",
        meta={},
    )


@pytest.fixture
def deleted(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[str]]]:
    recorded: list[tuple[str, list[str]]] = []

    async def fake_delete_pods(*, namespace: str, pod_names: list[str]) -> None:
        recorded.append((namespace, list(pod_names)))

    monkeypatch.setattr(cell_operations_kubernetes, "_delete_pods", fake_delete_pods)
    return recorded


def _operations(infos):
    return KubernetesCellOperations(provider=FakeProvider(infos), namespace="rl")


class TestCellInfos:
    def test_reports_the_cells_of_the_specs_it_was_asked_about(self):
        """A trainer handler must not list the inference cells that share the namespace."""
        infos = {"trainer-actor-0": _info(), "engine-0": _info(cell_id="engine-0", pool_id="engine")}
        operations = _operations(infos)

        listed = asyncio.run(operations.cell_infos(pool_ids=["trainer-actor"]))

        assert list(listed) == ["trainer-actor-0"]

    def test_reports_nothing_when_no_cell_exists_yet(self):
        """A run whose pods are still being scheduled has no cells, which is not an error."""
        assert asyncio.run(_operations({}).cell_infos(pool_ids=["trainer-actor"])) == {}


class TestSuspend:
    def test_deletes_the_pods_of_the_cell_in_the_runs_namespace(self, deleted):
        """Deleting them is the whole operation: the workload brings the group back by itself."""
        operations = _operations({"trainer-actor-0": _info(workers=("p0", "p1"))})

        asyncio.run(operations.suspend(cell_id="trainer-actor-0"))

        assert deleted == [("rl", ["p0", "p1"])]

    def test_touches_no_other_cell(self, deleted):
        """Healing one dp group must leave the others training."""
        infos = {
            "trainer-actor-0": _info(workers=("a",)),
            "trainer-actor-1": _info(cell_id="trainer-actor-1", workers=("b",)),
        }

        asyncio.run(_operations(infos).suspend(cell_id="trainer-actor-0"))

        assert deleted == [("rl", ["a"])]

    def test_refuses_a_cell_with_no_pods(self, deleted):
        """There is nothing to delete, and silently succeeding would report a heal that never happened."""
        with pytest.raises(AssertionError, match="no pods"):
            asyncio.run(_operations({}).suspend(cell_id="trainer-actor-0"))


class TestResume:
    def test_says_it_cannot_promise_the_moment_a_cell_comes_back(self):
        """The workload recreates a deleted cell on its own schedule, so a caller that waited here would be lied to."""
        with pytest.raises(NotImplementedError, match="no moment to return at"):
            asyncio.run(_operations({"trainer-actor-0": _info()}).resume(cell_id="trainer-actor-0"))


class TestInjectFault:
    def test_says_it_cannot_reach_into_a_worker_process(self):
        """Silently doing nothing would make a fault-injection test pass while injecting no fault."""
        with pytest.raises(NotImplementedError, match="rpc layer"):
            asyncio.run(
                _operations({"trainer-actor-0": _info()}).inject_fault(
                    cell_id="trainer-actor-0", mode=list(FailureMode)[0], sub_index=0
                )
            )


async def _stop_watching() -> None:
    return None
