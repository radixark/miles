import asyncio
import sys
from types import ModuleType
from typing import Any

import pytest

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations import kubernetes as cell_operations_kubernetes
from miles.utils.workers.cell_operations.kubernetes import KubernetesCellOperations
from miles.utils.workers.worker_provider.base import CellInfo


class FakeProvider:
    def __init__(self, infos: dict[str, CellInfo], *, start_delay: float = 0.0) -> None:
        self._infos = infos
        self._start_delay = start_delay
        self.watches = 0

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


def _operations(infos, *, start_delay: float = 0.0):
    return KubernetesCellOperations(provider=FakeProvider(infos, start_delay=start_delay), namespace="rl")


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

    async def test_a_failed_watch_is_retried_by_the_next_operation(self) -> None:
        """A failed watch propagates once and the next operation starts a fresh watch."""

        class FailingOnceProvider(FakeProvider):
            async def watch_cells(self, reconcile: Any) -> Any:
                self.watches += 1
                if self.watches == 1:
                    raise RuntimeError("watch failed")
                return _stop_watching

        provider = FailingOnceProvider({"trainer-engine-actor-0": _info()})
        operations = KubernetesCellOperations(provider=provider, namespace="rl")

        with pytest.raises(RuntimeError, match="watch failed"):
            await operations.cell_infos(pool_ids=["trainer-engine-actor"])

        listed = await operations.cell_infos(pool_ids=["trainer-engine-actor"])

        assert list(listed) == ["trainer-engine-actor-0"]
        assert provider.watches == 2

    async def test_cancelling_the_waiter_cancels_and_discards_its_watch(self) -> None:
        """A cancelled request must not leave its unfinished cell watch running in the background."""
        watch_started = asyncio.Event()
        watch_cancelled = asyncio.Event()

        class CancellationAwareProvider(FakeProvider):
            async def watch_cells(self, reconcile: Any) -> Any:
                self.watches += 1
                watch_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    watch_cancelled.set()

        provider = CancellationAwareProvider({"trainer-engine-actor-0": _info()})
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        waiter = asyncio.create_task(operations.cell_infos(pool_ids=["trainer-engine-actor"]))
        await watch_started.wait()

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        assert watch_cancelled.is_set()
        assert operations._watching is None

    async def test_cancelling_one_waiter_does_not_cancel_the_shared_watch(self) -> None:
        """Cancelling one startup waiter leaves the shared watch available to another waiter."""
        watch_started = asyncio.Event()
        release_watch = asyncio.Event()

        class EventControlledProvider(FakeProvider):
            async def watch_cells(self, reconcile: Any) -> Any:
                self.watches += 1
                watch_started.set()
                await release_watch.wait()
                return _stop_watching

        provider = EventControlledProvider({"trainer-engine-actor-0": _info()})
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        cancelled_waiter = asyncio.create_task(operations.cell_infos(pool_ids=["trainer-engine-actor"]))
        completing_waiter = asyncio.create_task(operations.cell_infos(pool_ids=["trainer-engine-actor"]))
        await watch_started.wait()
        await asyncio.sleep(0)

        cancelled_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_waiter
        release_watch.set()

        listed = await completing_waiter

        assert list(listed) == ["trainer-engine-actor-0"]
        assert provider.watches == 1


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


class TestDeletePods:
    async def test_delete_pods_uses_the_in_cluster_client_and_requested_namespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pod deletion loads in-cluster config and targets every requested pod in its namespace."""
        config_loads: list[None] = []
        deletions: list[tuple[str, str]] = []

        class FakeApiClient:
            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *args: Any) -> None:
                return None

        class FakeCoreV1Api:
            def __init__(self, api_client: FakeApiClient) -> None:
                self.api_client = api_client

            async def delete_namespaced_pod(self, *, name: str, namespace: str) -> None:
                deletions.append((name, namespace))

        client_module = ModuleType("kubernetes_asyncio.client")
        client_module.ApiClient = FakeApiClient
        client_module.CoreV1Api = FakeCoreV1Api
        config_module = ModuleType("kubernetes_asyncio.config")
        config_module.load_incluster_config = lambda: config_loads.append(None)
        package = ModuleType("kubernetes_asyncio")
        package.client = client_module
        package.config = config_module
        monkeypatch.setitem(sys.modules, "kubernetes_asyncio", package)
        monkeypatch.setitem(sys.modules, "kubernetes_asyncio.client", client_module)
        monkeypatch.setitem(sys.modules, "kubernetes_asyncio.config", config_module)

        await cell_operations_kubernetes._delete_pods(namespace="training", pod_names=["pod-0", "pod-1"])

        assert config_loads == [None]
        assert deletions == [("pod-0", "training"), ("pod-1", "training")]


class TestResume:
    def test_says_it_cannot_promise_the_moment_a_cell_comes_back(self):
        """The workload recreates a deleted cell on its own schedule, so a caller that waited here would be lied to."""
        with pytest.raises(NotImplementedError, match="no moment to return at"):
            asyncio.run(_operations({"trainer-engine-actor-0": _info()}).resume(cell_id="trainer-engine-actor-0"))


class TestInjectFault:
    def test_says_it_cannot_reach_into_a_worker_process(self):
        """Silently doing nothing would make a fault-injection test pass while injecting no fault."""
        with pytest.raises(NotImplementedError, match="rpc layer"):
            asyncio.run(
                _operations({"trainer-engine-actor-0": _info()}).inject_fault(
                    cell_id="trainer-engine-actor-0", mode=list(FailureMode)[0], sub_index=0
                )
            )


async def _stop_watching() -> None:
    return None
