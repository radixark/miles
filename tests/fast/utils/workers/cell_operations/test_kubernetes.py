import asyncio
import sys
from contextlib import asynccontextmanager
from types import ModuleType
from typing import Any

import pytest

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations import kubernetes as cell_operations_kubernetes
from miles.utils.workers.cell_operations.base import (
    CellTerminationNotConfirmedError,
    CellTerminationOutcome,
    FaultTarget,
    StaleFaultTargetError,
)
from miles.utils.workers.cell_operations.kubernetes import KubernetesCellOperations
from miles.utils.workers.worker_handle import WorkerUnreachableError
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_provider.kubernetes.core.cell_view import CellIncarnation, PodIdentity
from miles.utils.workers.worker_provider.kubernetes.core.pod_view import (
    ContainerIdentity,
    ParsedPod,
    cell_members_hash,
)


class FakeHandle:
    def __init__(
        self,
        name: str,
        *,
        calls: list[tuple[str, str]],
        submissions: list[tuple[str, str, str]],
        effect: str | Exception = "return",
    ) -> None:
        self._name = name
        self._calls = calls
        self._submissions = submissions
        self._effect = effect

    async def inject_fault(self, *, mode: str) -> None:
        self._calls.append((self._name, mode))
        if self._effect == "unreachable":
            raise WorkerUnreachableError(f"{self._name} is gone")
        if self._effect == "never_answers":
            await asyncio.sleep(3600)
        if isinstance(self._effect, Exception):
            raise self._effect

    async def submit_without_result(self, method_name: str, /, **kwargs: Any) -> None:
        self._submissions.append((self._name, method_name, kwargs["mode"]))
        await self.inject_fault(mode=kwargs["mode"])


class FakeProvider:
    def __init__(
        self,
        infos: dict[str, CellInfo],
        *,
        start_delay: float = 0.0,
        handle_effect: str | Exception = "return",
        unserved_workers: tuple[str, ...] = (),
    ) -> None:
        self._infos = infos
        self._start_delay = start_delay
        self._handle_effect = handle_effect
        self._unserved_workers = unserved_workers
        self.watches = 0
        self.injections: list[tuple[str, str]] = []
        self.submissions: list[tuple[str, str, str]] = []

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [self._worker_infos_of_cell(cell_id) for cell_id in cell_ids]

    def get_handles_of_worker_infos(self, infos: list[WorkerInfo]) -> dict[str, FakeHandle]:
        return {
            info.name: FakeHandle(
                info.name,
                calls=self.injections,
                submissions=self.submissions,
                effect=self._handle_effect,
            )
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


def _operations(infos, *, start_delay: float = 0.0, handle_effect: str | Exception = "return", unserved_workers=()):
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
    @pytest.mark.parametrize("changed", ["hash", "pod", "gone", "boot", "cell", "index"])
    async def test_a_stale_observed_target_is_rejected_before_rpc_dispatch(self, changed: str) -> None:
        """An old cell or pod identity cannot be dispatched to a replacement worker."""
        provider = IncarnationProvider(pods=[] if changed == "gone" else [make_pod("engine-0-0", uid="current-pod")])
        operations = KubernetesCellOperations(provider=provider, namespace="ns")
        target = FaultTarget(
            cell_id="other-cell" if changed == "cell" else "engine-0",
            sub_index=1 if changed == "index" else 0,
            workers_hash="old-hash" if changed == "hash" else "hash-1",
            pod_uid="old-pod" if changed == "pod" else "current-pod",
            boot_uuid=None if changed == "boot" else "boot",
        )

        with pytest.raises(StaleFaultTargetError):
            await operations.inject_fault(
                cell_id="engine-0", mode=FailureMode.SIGKILL, sub_index=0, expected_target=target
            )
        assert provider.submissions == []

    def test_submits_the_crash_without_waiting_for_a_result(self) -> None:
        """A self-crashing RPC is sent through the acknowledgement-only worker-handle operation."""
        operations = _operations({"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))})

        asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SIGKILL, sub_index=0))

        assert operations._provider.submissions == [("engine-0-0", "inject_fault", "sigkill")]

    def test_calls_the_worker_the_sub_index_picks(self):
        """A multi-pod cell is crashed by crashing one named rank, not whichever rank came first."""
        operations = _operations({"engine-0": _info(cell_id="engine-0", workers=("engine-0-0", "engine-0-1"))})

        asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SIGKILL, sub_index=1))

        assert operations._provider.injections == [("engine-0-1", "sigkill")]

    def test_passes_the_requested_mode_to_the_worker(self):
        """The caller chose the failure mode, so the worker must not be crashed some other way."""
        operations = _operations({"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))})

        asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SEGFAULT, sub_index=0))

        assert operations._provider.injections == [("engine-0-0", "segfault")]

    def test_a_worker_that_dies_before_answering_is_a_success(self):
        """The call kills its own callee, so an unreachable worker is the outcome that was asked for."""
        operations = _operations(
            {"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))}, handle_effect="unreachable"
        )

        asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SIGKILL, sub_index=0))

        assert operations._provider.injections == [("engine-0-0", "sigkill")]

    def test_a_worker_that_never_answers_does_not_hang_the_caller(self, monkeypatch: pytest.MonkeyPatch):
        """A killed process leaves the rpc poll retrying for an hour, which would block the api server request."""
        monkeypatch.setattr(cell_operations_kubernetes, "INJECT_FAULT_TIMEOUT_SECONDS", 0.05)
        operations = _operations(
            {"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))}, handle_effect="never_answers"
        )

        asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SIGKILL, sub_index=0))

        assert operations._provider.injections == [("engine-0-0", "sigkill")]

    def test_an_unexpected_rpc_failure_is_propagated(self):
        """An unrelated RPC failure must not be mistaken for confirmation that the worker crashed."""
        operations = _operations(
            {"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))},
            handle_effect=RuntimeError("rpc protocol failed"),
        )

        with pytest.raises(RuntimeError, match="rpc protocol failed"):
            asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SIGKILL, sub_index=0))

    def test_a_sub_index_beyond_the_cell_is_rejected(self):
        """Injecting into a neighbouring cell by accident would corrupt the test's premise."""
        operations = _operations({"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))})

        with pytest.raises(AssertionError, match="out of range"):
            asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SIGKILL, sub_index=1))

    def test_a_negative_sub_index_is_rejected(self):
        """Negative indexing would silently select the last worker instead of failing."""
        operations = _operations({"engine-0": _info(cell_id="engine-0", workers=("engine-0-0", "engine-0-1"))})

        with pytest.raises(AssertionError, match="out of range"):
            asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SIGKILL, sub_index=-1))

    def test_a_worker_that_is_not_served_over_rpc_is_rejected(self):
        """There is no call to make, and succeeding here would report a crash that never happened."""
        operations = _operations(
            {"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))}, unserved_workers=("engine-0-0",)
        )

        with pytest.raises(AssertionError, match="not served over rpc"):
            asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SIGKILL, sub_index=0))


async def _stop_watching() -> None:
    return None


TERMINATED_CELL_ID = "trainer-engine-actor-0"


class _FakeApiException(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"the api server answered {status}")
        self.status = status


class _FakeDeleteOptions:
    def __init__(self, *, preconditions: Any) -> None:
        self.preconditions = preconditions


class _FakePreconditions:
    def __init__(self, *, uid: str, resource_version: str | None) -> None:
        self.uid = uid
        self.resource_version = resource_version


class _Deletion:
    def __init__(self, *, name: str, namespace: str, uid: str | None, resource_version: str | None) -> None:
        self.name = name
        self.namespace = namespace
        self.uid = uid
        self.resource_version = resource_version


class IncarnationProvider(FakeProvider):
    """A reflector store whose pods the fake api server mutates, the way a real deletion would."""

    def __init__(self, *, pods: list[PodIdentity], workers_hash: str = "hash-1", watch_delay: float = 0.0) -> None:
        super().__init__({}, start_delay=watch_delay)
        self.pods = list(pods)
        self.workers_hash = workers_hash

    def cell_incarnation(self, cell_id: str) -> CellIncarnation | None:
        if not self.pods:
            return None
        return CellIncarnation(cell_id=cell_id, workers_hash=self.workers_hash, pods=list(self.pods))

    def replace_pod(self, name: str, *, pod: PodIdentity | None) -> None:
        self.pods = [existing for existing in self.pods if existing.name != name] + ([] if pod is None else [pod])


def make_pod(
    name: str,
    *,
    uid: str | None = None,
    resource_version: str = "rv-1",
    restart_count: int = 0,
    declared_container_names: list[str] | None = None,
    containers: list[ContainerIdentity] | None = None,
) -> PodIdentity:
    reported = [make_container(restart_count=restart_count)] if containers is None else containers
    return PodIdentity(
        name=name,
        uid=uid if uid is not None else f"uid-of-{name}",
        resource_version=resource_version,
        restart_count=restart_count,
        declared_container_names=(["worker"] if declared_container_names is None else list(declared_container_names)),
        containers=reported,
    )


_DEFAULT_CONTAINER_ID = object()


def make_container(
    *, name: str = "worker", restart_count: int = 0, container_id: str | None | object = _DEFAULT_CONTAINER_ID
) -> ContainerIdentity:
    if container_id is _DEFAULT_CONTAINER_ID:
        container_id = f"containerd://{name}-{restart_count}"
    assert isinstance(container_id, str) or container_id is None

    return ContainerIdentity(
        name=name,
        container_id=container_id,
        restart_count=restart_count,
    )


def install_fake_api(
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider: IncarnationProvider,
    statuses: dict[str, list[int]] | None = None,
    repeat_statuses: bool = False,
    forget_deleted_pods: bool = True,
    pods_after_delete: dict[str, PodIdentity | None] | None = None,
) -> list[_Deletion]:
    deletions: list[_Deletion] = []
    pending = {name: list(codes) for name, codes in (statuses or {}).items()}
    after_delete = dict(pods_after_delete or {})

    class FakeCoreV1Api:
        async def delete_namespaced_pod(self, *, name: str, namespace: str, body: Any = None) -> None:
            preconditions = None if body is None else body.preconditions
            deletions.append(
                _Deletion(
                    name=name,
                    namespace=namespace,
                    uid=None if preconditions is None else preconditions.uid,
                    resource_version=None if preconditions is None else preconditions.resource_version,
                )
            )
            if codes := pending.get(name):
                status = codes[0] if repeat_statuses else codes.pop(0)
                if status == 404 and forget_deleted_pods:
                    provider.replace_pod(name, pod=None)
                raise _FakeApiException(status)
            if name in after_delete:
                provider.replace_pod(name, pod=after_delete[name])
            elif forget_deleted_pods:
                provider.replace_pod(name, pod=None)

    @asynccontextmanager
    async def fake_core_v1_api():
        yield FakeCoreV1Api()

    client_module = ModuleType("kubernetes_asyncio.client")
    client_module.V1DeleteOptions = _FakeDeleteOptions
    client_module.V1Preconditions = _FakePreconditions
    client_module.ApiException = _FakeApiException
    package = ModuleType("kubernetes_asyncio")
    package.client = client_module
    monkeypatch.setitem(sys.modules, "kubernetes_asyncio", package)
    monkeypatch.setitem(sys.modules, "kubernetes_asyncio.client", client_module)
    monkeypatch.setattr(cell_operations_kubernetes, "_core_v1_api", fake_core_v1_api)
    monkeypatch.setattr(cell_operations_kubernetes, "TERMINATE_POLL_INTERVAL_SECONDS", 0.001)
    return deletions


class TestTerminateIncarnation:
    async def test_every_pod_is_deleted_under_its_own_uid_precondition(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bare name deletes whatever pod holds it now, which after a restart is the innocent replacement."""
        provider = IncarnationProvider(pods=[make_pod("p0"), make_pod("p1", resource_version="rv-7")])
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        deletions = install_fake_api(monkeypatch, provider=provider)

        outcome = await operations.terminate_incarnation(
            cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=5.0
        )

        assert outcome is CellTerminationOutcome.TERMINATED
        assert sorted((d.name, d.uid, d.resource_version, d.namespace) for d in deletions) == [
            ("p0", "uid-of-p0", "rv-1", "rl"),
            ("p1", "uid-of-p1", "rv-7", "rl"),
        ]

    async def test_a_replaced_incarnation_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The failure report may be older than the cell it names, and the fresh pods are serving the run."""
        provider = IncarnationProvider(pods=[make_pod("p0")], workers_hash="hash-2")
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        deletions = install_fake_api(monkeypatch, provider=provider)

        outcome = await operations.terminate_incarnation(
            cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=5.0
        )

        assert outcome is CellTerminationOutcome.STALE
        assert deletions == []

    async def test_a_cell_with_no_observed_pod_is_already_gone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nothing is running under that name, so demanding a deletion would block healing forever."""
        provider = IncarnationProvider(pods=[])
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        deletions = install_fake_api(monkeypatch, provider=provider)

        outcome = await operations.terminate_incarnation(
            cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=5.0
        )

        assert outcome is CellTerminationOutcome.ALREADY_GONE
        assert deletions == []

    async def test_a_pod_the_api_server_never_heard_of_has_left(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 404 means the old pod is gone, which is the outcome the caller asked for."""
        provider = IncarnationProvider(pods=[make_pod("p0")])
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        install_fake_api(monkeypatch, provider=provider, statuses={"p0": [404]})

        outcome = await operations.terminate_incarnation(
            cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=5.0
        )

        assert outcome is CellTerminationOutcome.TERMINATED

    async def test_a_pod_replaced_under_the_same_name_is_not_deleted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: retrying a conflicted delete without preconditions would kill the fresh incarnation."""
        provider = IncarnationProvider(pods=[make_pod("p0")])
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        deletions = install_fake_api(monkeypatch, provider=provider, statuses={"p0": [409]})
        provider.replace_pod("p0", pod=make_pod("p0", uid="uid-of-the-replacement", resource_version="rv-9"))

        outcome = await operations.terminate_incarnation(
            cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=5.0
        )

        assert outcome is CellTerminationOutcome.TERMINATED
        assert [(d.name, d.uid) for d in deletions] == [("p0", "uid-of-p0")]
        assert [pod.uid for pod in provider.pods] == ["uid-of-the-replacement"]

    async def test_a_restarted_container_is_not_deleted_under_the_old_observation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pod kept its uid but the process the caller wanted gone already died with the restart."""
        provider = IncarnationProvider(pods=[make_pod("p0")])
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        deletions = install_fake_api(monkeypatch, provider=provider, statuses={"p0": [409]})
        provider.replace_pod("p0", pod=make_pod("p0", resource_version="rv-9", restart_count=1))

        await operations.terminate_incarnation(cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=5.0)

        assert [(d.uid, d.resource_version) for d in deletions] == [("uid-of-p0", "rv-1")]

    async def test_a_conflict_on_the_same_pod_is_retried_with_what_was_observed_next(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A conflict usually only means the observation lagged, so the same pod is deleted at its current version."""
        provider = IncarnationProvider(pods=[make_pod("p0")])
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        deletions = install_fake_api(monkeypatch, provider=provider, statuses={"p0": [409]})
        provider.replace_pod("p0", pod=make_pod("p0", resource_version="rv-2"))

        outcome = await operations.terminate_incarnation(
            cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=5.0
        )

        assert outcome is CellTerminationOutcome.TERMINATED
        assert [(d.uid, d.resource_version) for d in deletions] == [("uid-of-p0", "rv-1"), ("uid-of-p0", "rv-2")]

    async def test_a_long_run_of_conflicts_on_one_generation_still_ends_in_a_delete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A busy apiserver can bump the resourceVersion more often than any fixed retry budget allows."""
        provider = IncarnationProvider(pods=[make_pod("p0")])
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        deletions = install_fake_api(monkeypatch, provider=provider, statuses={"p0": [409] * 6})

        outcome = await operations.terminate_incarnation(
            cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=5.0
        )

        assert outcome is CellTerminationOutcome.TERMINATED
        assert len(deletions) == 7

    async def test_a_pod_that_never_accepts_the_delete_is_reported_as_a_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retrying forever, or claiming success, would leave a live rank holding the gpus of a healed cell."""
        provider = IncarnationProvider(pods=[make_pod("p0")])
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        deletions = install_fake_api(monkeypatch, provider=provider, statuses={"p0": [409]}, repeat_statuses=True)

        with pytest.raises(CellTerminationNotConfirmedError):
            await operations.terminate_incarnation(
                cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=0.05
            )

        assert deletions

    async def test_an_api_error_is_not_mistaken_for_a_termination(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A refused or broken api call says nothing about whether the workers are still running."""
        provider = IncarnationProvider(pods=[make_pod("p0")])
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        install_fake_api(monkeypatch, provider=provider, statuses={"p0": [500]})

        with pytest.raises(_FakeApiException):
            await operations.terminate_incarnation(
                cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=5.0
            )

    async def test_a_pod_still_observed_after_its_deletion_is_not_confirmed_dead(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deletion request is not a dead process: the kubelet may still be tearing the container down."""
        provider = IncarnationProvider(pods=[make_pod("p0")])
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        install_fake_api(monkeypatch, provider=provider, forget_deleted_pods=False)

        with pytest.raises(CellTerminationNotConfirmedError):
            await operations.terminate_incarnation(
                cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=0.05
            )

    async def test_a_restarted_sole_worker_container_confirms_the_old_process_left(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pod that hosts exactly one worker container reports the restart that ended the process asked about."""
        provider = IncarnationProvider(pods=[make_pod("p0")])
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        install_fake_api(
            monkeypatch,
            provider=provider,
            forget_deleted_pods=False,
            pods_after_delete={"p0": make_pod("p0", resource_version="rv-9", restart_count=1)},
        )

        outcome = await operations.terminate_incarnation(
            cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=5.0
        )

        assert outcome is CellTerminationOutcome.TERMINATED

    async def test_a_pod_declaring_a_sidecar_is_not_confirmed_by_the_only_status_it_reports(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: a status list can be short, and one restart there would read as the whole pod having died."""
        declared = ["worker", "log-shipper"]
        provider = IncarnationProvider(
            pods=[make_pod("p0", declared_container_names=declared, containers=[make_container(name="log-shipper")])]
        )
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        install_fake_api(
            monkeypatch,
            provider=provider,
            forget_deleted_pods=False,
            pods_after_delete={
                "p0": make_pod(
                    "p0",
                    resource_version="rv-9",
                    restart_count=1,
                    declared_container_names=declared,
                    containers=[make_container(name="log-shipper", restart_count=1)],
                )
            },
        )

        with pytest.raises(CellTerminationNotConfirmedError):
            await operations.terminate_incarnation(
                cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=0.05
            )

    async def test_a_sole_declared_worker_whose_status_is_missing_confirms_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pod that declares one worker but reports no status for it says nothing about that process."""
        provider = IncarnationProvider(pods=[make_pod("p0", containers=[])])
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        install_fake_api(
            monkeypatch,
            provider=provider,
            forget_deleted_pods=False,
            pods_after_delete={"p0": make_pod("p0", resource_version="rv-9", restart_count=1, containers=[])},
        )

        with pytest.raises(CellTerminationNotConfirmedError):
            await operations.terminate_incarnation(
                cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=0.05
            )

    async def test_a_sole_declared_worker_whose_status_names_another_container_confirms_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A status entry under a name the spec never declared cannot be the worker this termination targeted."""
        provider = IncarnationProvider(pods=[make_pod("p0", containers=[make_container(name="init-weights")])])
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        install_fake_api(
            monkeypatch,
            provider=provider,
            forget_deleted_pods=False,
            pods_after_delete={
                "p0": make_pod(
                    "p0",
                    resource_version="rv-9",
                    restart_count=1,
                    containers=[make_container(name="init-weights", restart_count=1)],
                )
            },
        )

        with pytest.raises(CellTerminationNotConfirmedError):
            await operations.terminate_incarnation(
                cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=0.05
            )

    @pytest.mark.parametrize(
        ("before_container_id", "after_container_id"),
        [
            (None, "containerd://worker-1"),
            ("", "containerd://worker-1"),
            ("containerd://worker-0", None),
            ("containerd://worker-0", ""),
        ],
    )
    async def test_a_sole_declared_worker_with_an_incomplete_container_id_confirms_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        before_container_id: str | None,
        after_container_id: str | None,
    ) -> None:
        """Without a container id on both readings the restart count alone could belong to a different runtime."""
        provider = IncarnationProvider(
            pods=[make_pod("p0", containers=[make_container(container_id=before_container_id)])]
        )
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        install_fake_api(
            monkeypatch,
            provider=provider,
            forget_deleted_pods=False,
            pods_after_delete={
                "p0": make_pod(
                    "p0",
                    resource_version="rv-9",
                    restart_count=1,
                    containers=[make_container(restart_count=1, container_id=after_container_id)],
                )
            },
        )

        with pytest.raises(CellTerminationNotConfirmedError):
            await operations.terminate_incarnation(
                cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=0.05
            )

    async def test_a_pod_that_declares_no_container_at_all_confirms_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An observation with no spec projection cannot say the single status it carries is the worker's."""
        provider = IncarnationProvider(pods=[make_pod("p0", declared_container_names=[])])
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        install_fake_api(
            monkeypatch,
            provider=provider,
            forget_deleted_pods=False,
            pods_after_delete={
                "p0": make_pod("p0", resource_version="rv-9", restart_count=1, declared_container_names=[])
            },
        )

        with pytest.raises(CellTerminationNotConfirmedError):
            await operations.terminate_incarnation(
                cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=0.05
            )

    async def test_a_sidecar_restart_in_a_multi_container_pod_confirms_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: a summed restart count rises for any container, and a live worker would read as dead."""
        declared = ["worker", "log-shipper"]
        before = [make_container(name="worker"), make_container(name="log-shipper")]
        after = [make_container(name="worker"), make_container(name="log-shipper", restart_count=1)]
        provider = IncarnationProvider(pods=[make_pod("p0", declared_container_names=declared, containers=before)])
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        install_fake_api(
            monkeypatch,
            provider=provider,
            forget_deleted_pods=False,
            pods_after_delete={
                "p0": make_pod(
                    "p0",
                    resource_version="rv-9",
                    restart_count=1,
                    declared_container_names=declared,
                    containers=after,
                )
            },
        )

        with pytest.raises(CellTerminationNotConfirmedError):
            await operations.terminate_incarnation(
                cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=0.05
            )

    async def test_a_multi_container_pod_is_confirmed_once_its_uid_is_gone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a container that can be named the worker's, only the pod itself leaving proves the ranks died."""
        containers = [make_container(name="worker"), make_container(name="log-shipper")]
        provider = IncarnationProvider(
            pods=[make_pod("p0", declared_container_names=["worker", "log-shipper"], containers=containers)]
        )
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        install_fake_api(monkeypatch, provider=provider)

        outcome = await operations.terminate_incarnation(
            cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=5.0
        )

        assert outcome is CellTerminationOutcome.TERMINATED

    async def test_the_watch_is_running_before_any_pod_is_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reading the store before the reflector filled it would report every cell as already gone."""
        provider = IncarnationProvider(pods=[make_pod("p0")])
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        install_fake_api(monkeypatch, provider=provider)

        await operations.terminate_incarnation(cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=5.0)

        assert provider.watches == 1

    async def test_a_watch_that_never_starts_does_not_block_the_termination_forever(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reflector is started by this very call, so a stuck apiserver would hang the caller before any deadline."""
        provider = IncarnationProvider(pods=[make_pod("p0")], watch_delay=3600)
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        deletions = install_fake_api(monkeypatch, provider=provider)

        with pytest.raises(CellTerminationNotConfirmedError):
            await asyncio.wait_for(
                operations.terminate_incarnation(
                    cell_id=TERMINATED_CELL_ID, expected_workers_hash="hash-1", timeout=0.05
                ),
                timeout=5.0,
            )

        assert deletions == []

    async def test_a_pod_only_marked_for_deletion_is_reported_as_stale_rather_than_terminated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """deletionTimestamp moves the cell hash while the same uid keeps running, so stale is no proof of death."""
        pods = [_parsed_pod(deleting=False)]
        deleting = [_parsed_pod(deleting=True)]
        provider = IncarnationProvider(pods=[make_pod("p0")], workers_hash=cell_members_hash(deleting))
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        deletions = install_fake_api(monkeypatch, provider=provider)

        outcome = await operations.terminate_incarnation(
            cell_id=TERMINATED_CELL_ID, expected_workers_hash=cell_members_hash(pods), timeout=5.0
        )

        assert outcome is CellTerminationOutcome.STALE
        assert deletions == []


def _parsed_pod(*, deleting: bool) -> ParsedPod:
    return ParsedPod(
        name="p0",
        cell_id=TERMINATED_CELL_ID,
        cell_index=0,
        pool_id="trainer-engine-actor",
        pod_in_cell_index=0,
        ready=True,
        deleting=deleting,
        pod_ip="10.0.0.1",
        uid="uid-of-p0",
        resource_version="rv-1",
        restart_count=0,
        declared_container_names=("worker",),
        containers=(make_container(),),
        meta={},
        cell_size=1,
        subdomain=None,
        gpu_ids=(),
    )
