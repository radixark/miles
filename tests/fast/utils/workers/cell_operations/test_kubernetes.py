import asyncio

import pytest

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations import kubernetes as cell_operations_kubernetes
from miles.utils.workers.cell_operations.kubernetes import KubernetesCellOperations
from miles.utils.workers.worker_handle import WorkerUnreachableError
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo


class FakeHandle:
    def __init__(self, name: str, *, calls: list[tuple[str, str]], effect: str = "return") -> None:
        self._name = name
        self._calls = calls
        self._effect = effect

    async def inject_fault(self, *, mode: str) -> None:
        self._calls.append((self._name, mode))
        if self._effect == "unreachable":
            raise WorkerUnreachableError(f"{self._name} is gone")
        if self._effect == "never_answers":
            await asyncio.sleep(3600)


class FakeProvider:
    def __init__(
        self,
        infos: dict[str, CellInfo],
        *,
        start_delay: float = 0.0,
        handle_effect: str = "return",
        unserved_workers: tuple[str, ...] = (),
    ) -> None:
        self._infos = infos
        self._start_delay = start_delay
        self._handle_effect = handle_effect
        self._unserved_workers = unserved_workers
        self.watches = 0
        self.injections: list[tuple[str, str]] = []

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [self._worker_infos_of_cell(cell_id) for cell_id in cell_ids]

    def get_worker_infos_if_generation(
        self,
        *,
        cell_id: str,
        expected_workers_hash: str,
    ) -> list[WorkerInfo] | None:
        info = self._infos.get(cell_id)
        if info is None or info.workers_hash != expected_workers_hash:
            return None
        return self._worker_infos_of_cell(cell_id)

    def get_handles_of_worker_infos(self, infos: list[WorkerInfo]) -> dict[str, FakeHandle]:
        return {
            info.name: FakeHandle(info.name, calls=self.injections, effect=self._handle_effect)
            for info in infos
            if info.name not in self._unserved_workers
        }

    async def watch_cells(self, reconcile):
        self.watches += 1
        await asyncio.sleep(self._start_delay)
        return _stop_watching

    def cell_ids(self) -> list[str]:
        return sorted(self._infos)

    def cell_info(self, cell_id: str) -> CellInfo | None:
        return self._infos.get(cell_id)

    def pod_names_of_cell(self, cell_id: str) -> list[str]:
        info = self._infos.get(cell_id)
        return list(info.worker_names) if info is not None else []

    def _worker_infos_of_cell(self, cell_id: str) -> list[WorkerInfo]:
        info = self._infos.get(cell_id)
        return [
            WorkerInfo(name=name, generation=0, self_addrs={}, gpu_ids=[], worker_class="fake.Worker")
            for name in (info.worker_names if info is not None else [])
        ]


def _info(cell_id="trainer-engine-actor-0", pool_id="trainer-engine-actor", workers=("trainer-engine-actor-0-0",)):
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


def _operations(infos, *, start_delay: float = 0.0, handle_effect: str = "return", unserved_workers=()):
    provider = FakeProvider(
        infos, start_delay=start_delay, handle_effect=handle_effect, unserved_workers=unserved_workers
    )
    return KubernetesCellOperations(provider=provider, namespace="rl")


class TestCellInfos:
    def test_reports_the_cells_of_the_specs_it_was_asked_about(self):
        """A trainer handler must not list the inference cells that share the namespace."""
        infos = {"trainer-engine-actor-0": _info(), "engine-0": _info(cell_id="engine-0", pool_id="engine")}
        operations = _operations(infos)

        listed = asyncio.run(operations.cell_infos(pool_ids=["trainer-engine-actor"]))

        assert list(listed) == ["trainer-engine-actor-0"]

    def test_reports_nothing_when_no_cell_exists_yet(self):
        """A run whose pods are still being scheduled has no cells, which is not an error."""
        assert asyncio.run(_operations({}).cell_infos(pool_ids=["trainer-engine-actor"])) == {}


class TestWatching:
    def test_the_first_read_starts_the_watch_it_needs(self):
        """Nothing else starts it, and reading the store before the reflector filled it reports an empty run."""
        operations = _operations({"trainer-engine-actor-0": _info()})

        asyncio.run(operations.cell_infos(pool_ids=["trainer-engine-actor"]))

        assert operations._provider.watches == 1

    def test_later_reads_reuse_the_watch_already_running(self, deleted):
        """A second reflector would double the apiserver load and leak the first one's session."""
        operations = _operations({"trainer-engine-actor-0": _info()})

        async def scenario():
            await operations.cell_infos(pool_ids=["trainer-engine-actor"])
            await operations.cell_infos(pool_ids=["trainer-engine-actor"])
            await operations.suspend(cell_id="trainer-engine-actor-0")

        asyncio.run(scenario())

        assert operations._provider.watches == 1

    def test_concurrent_first_reads_start_one_watch_between_them(self, deleted):
        """The api server gathers its handlers, which share one instance, so the very first request races itself."""
        operations = _operations({"trainer-engine-actor-0": _info()}, start_delay=0.05)

        async def scenario():
            await asyncio.gather(*[operations.cell_infos(pool_ids=["trainer-engine-actor"]) for _ in range(3)])

        asyncio.run(scenario())

        assert operations._provider.watches == 1


class TestSuspend:
    def test_deletes_the_pods_of_the_cell_in_the_runs_namespace(self, deleted):
        """Deleting them is the whole operation: the workload brings the group back by itself."""
        operations = _operations({"trainer-engine-actor-0": _info(workers=("p0", "p1"))})

        asyncio.run(operations.suspend(cell_id="trainer-engine-actor-0"))

        assert deleted == [("rl", ["p0", "p1"])]

    def test_touches_no_other_cell(self, deleted):
        """Healing one dp group must leave the others training."""
        infos = {
            "trainer-engine-actor-0": _info(workers=("a",)),
            "trainer-engine-actor-1": _info(cell_id="trainer-engine-actor-1", workers=("b",)),
        }

        asyncio.run(_operations(infos).suspend(cell_id="trainer-engine-actor-0"))

        assert deleted == [("rl", ["a"])]

    def test_refuses_a_cell_with_no_pods(self, deleted):
        """There is nothing to delete, and silently succeeding would report a heal that never happened."""
        with pytest.raises(AssertionError, match="no pods"):
            asyncio.run(_operations({}).suspend(cell_id="trainer-engine-actor-0"))


class TestResume:
    def test_says_it_cannot_promise_the_moment_a_cell_comes_back(self):
        """The workload recreates a deleted cell on its own schedule, so a caller that waited here would be lied to."""
        with pytest.raises(NotImplementedError, match="no moment to return at"):
            asyncio.run(_operations({"trainer-engine-actor-0": _info()}).resume(cell_id="trainer-engine-actor-0"))


class TestInjectFault:
    def test_calls_the_worker_the_sub_index_picks(self):
        """A multi-pod cell is crashed by crashing one named rank, not whichever rank came first."""
        operations = _operations({"engine-0": _info(cell_id="engine-0", workers=("engine-0-0", "engine-0-1"))})

        asyncio.run(
            operations.inject_fault(
                cell_id="engine-0", expected_workers_hash="h", mode=FailureMode.SIGKILL, sub_index=1
            )
        )

        assert operations._provider.injections == [("engine-0-1", "sigkill")]

    def test_passes_the_requested_mode_to_the_worker(self):
        """The caller chose the failure mode, so the worker must not be crashed some other way."""
        operations = _operations({"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))})

        asyncio.run(
            operations.inject_fault(
                cell_id="engine-0", expected_workers_hash="h", mode=FailureMode.SEGFAULT, sub_index=0
            )
        )

        assert operations._provider.injections == [("engine-0-0", "segfault")]

    def test_a_worker_that_dies_before_answering_is_a_success(self):
        """The call kills its own callee, so an unreachable worker is the outcome that was asked for."""
        operations = _operations(
            {"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))}, handle_effect="unreachable"
        )

        asyncio.run(
            operations.inject_fault(
                cell_id="engine-0", expected_workers_hash="h", mode=FailureMode.SIGKILL, sub_index=0
            )
        )

        assert operations._provider.injections == [("engine-0-0", "sigkill")]

    def test_a_worker_that_never_answers_does_not_hang_the_caller(self, monkeypatch: pytest.MonkeyPatch):
        """A killed process leaves the rpc poll retrying for an hour, which would block the api server request."""
        monkeypatch.setattr(cell_operations_kubernetes, "INJECT_FAULT_TIMEOUT_SECONDS", 0.05)
        operations = _operations(
            {"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))}, handle_effect="never_answers"
        )

        asyncio.run(
            operations.inject_fault(
                cell_id="engine-0", expected_workers_hash="h", mode=FailureMode.SIGKILL, sub_index=0
            )
        )

        assert operations._provider.injections == [("engine-0-0", "sigkill")]

    def test_a_sub_index_beyond_the_cell_is_rejected(self):
        """Injecting into a neighbouring cell by accident would corrupt the test's premise."""
        operations = _operations({"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))})

        with pytest.raises(AssertionError, match="out of range"):
            asyncio.run(
                operations.inject_fault(
                    cell_id="engine-0", expected_workers_hash="h", mode=FailureMode.SIGKILL, sub_index=1
                )
            )

    def test_a_negative_sub_index_is_rejected(self):
        """Negative indexing would silently select the last worker instead of failing."""
        operations = _operations({"engine-0": _info(cell_id="engine-0", workers=("engine-0-0", "engine-0-1"))})

        with pytest.raises(AssertionError, match="out of range"):
            asyncio.run(
                operations.inject_fault(
                    cell_id="engine-0", expected_workers_hash="h", mode=FailureMode.SIGKILL, sub_index=-1
                )
            )

    def test_a_worker_that_is_not_served_over_rpc_is_rejected(self):
        """There is no call to make, and succeeding here would report a crash that never happened."""
        operations = _operations(
            {"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))}, unserved_workers=("engine-0-0",)
        )

        with pytest.raises(AssertionError, match="not served over rpc"):
            asyncio.run(
                operations.inject_fault(
                    cell_id="engine-0", expected_workers_hash="h", mode=FailureMode.SIGKILL, sub_index=0
                )
            )

    def test_a_replaced_generation_is_rejected_before_any_worker_is_called(self):
        """A stale API observation must not crash a same-name replacement pod."""
        operations = _operations({"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))})

        with pytest.raises(RuntimeError, match="no longer has workers generation"):
            asyncio.run(
                operations.inject_fault(
                    cell_id="engine-0",
                    expected_workers_hash="stale",
                    mode=FailureMode.SIGKILL,
                    sub_index=0,
                )
            )

        assert operations._provider.injections == []

    def test_worker_selection_does_not_reread_after_the_generation_precondition(self):
        """A replacement between legacy split reads must not redirect the fault to the new worker."""
        old_info = _info(cell_id="engine-0", workers=("engine-0-old",))
        new_info = old_info.model_copy(update={"worker_names": ["engine-0-new"], "workers_hash": "new-generation"})
        operations = _operations({"engine-0": old_info})
        original_cell_info = operations._provider.cell_info

        def replace_after_returning_old_info(cell_id: str) -> CellInfo | None:
            info = original_cell_info(cell_id)
            operations._provider._infos[cell_id] = new_info
            return info

        operations._provider.cell_info = replace_after_returning_old_info

        asyncio.run(
            operations.inject_fault(
                cell_id="engine-0",
                expected_workers_hash="h",
                mode=FailureMode.SIGKILL,
                sub_index=0,
            )
        )

        assert operations._provider.injections == [("engine-0-old", "sigkill")]


async def _stop_watching() -> None:
    return None
