from __future__ import annotations

import asyncio
from argparse import Namespace
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from tests.fast.utils.workers.worker_provider.kubernetes import fake_pod_api
from tests.fast.utils.workers.worker_provider.kubernetes.core.test_pod_view import make_pod

from miles.ray import placement_group
from miles.ray.placement_group import create_rollout_components
from miles.ray.specs import inference as specs_inference
from miles.ray.specs import rollout as specs_rollout
from miles.ray.specs import train as specs_train
from miles.ray.specs.train import POOL_CATEGORY_TRAINER_ENGINE
from miles.ray.train.cell import TrainerCell
from miles.utils import http_utils
from miles.utils.ft_utils.api_server.models import CellStatus
from miles.utils.workers.cell_operations import kubernetes as cell_operations_kubernetes
from miles.utils.workers.k8s_types import Pod
from miles.utils.workers.reconcile.k8s_api import PodListPage
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.common.wire_types import Pickled
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.worker_provider.kubernetes.core import provider as core_provider
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.helm import env, naming
from miles.utils.workers.worker_provider.kubernetes.helm.builder import compute_helm_backend_capability
from miles.utils.workers.worker_provider.kubernetes.helm.env import NAMESPACE_ENV_VAR, RELEASE_ENV_VAR
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec

NAMESPACE = "rl"
_RELEASE = "miles-run-260805"

STATIC_HOSTS = {
    pool_id: naming.static_worker_host(_RELEASE, pool_id, 0)
    for pool_id in ("rollout-executor", "inference-controller", "trainer-controller-actor")
}
POOL = "trainer-engine-actor"
CELL_ID = "trainer-engine-actor-0"


class FakeTrainWorker:
    def __init__(self) -> None:
        self.configured: list[tuple[str, int]] = []

    def configure_master_addr_and_port(self, master_addr: str, master_port: int) -> int:
        self.configured.append((master_addr, master_port))
        return master_port

    def kill_self(self) -> None:
        return None


class FakeRolloutExecutor:
    def __init__(self) -> None:
        self.initialized = False
        self.loaded: list[int | None] = []
        self.train_parallel_config: dict[str, Any] | None = None

    async def init(self) -> None:
        self.initialized = True

    def dispose(self) -> None:
        return None

    async def get(self, rollout_id: int) -> dict[str, Any]:
        return {"sample_indices": [rollout_id], "data_ref": f"ref-{rollout_id}"}

    async def eval(self, rollout_id: int) -> None:
        return None

    def save(self, rollout_id: int) -> None:
        return None

    def load(self, rollout_id: int | None = None) -> None:
        self.loaded.append(rollout_id)

    def get_num_rollout_per_epoch(self) -> int:
        return 7

    def set_train_parallel_config(self, config: dict[str, Any]) -> None:
        self.train_parallel_config = config


class FakeInferenceController:
    def __init__(self) -> None:
        self.initialized = False
        self.prepared: list[int] = []

    async def init(self) -> None:
        self.initialized = True

    async def is_initialized(self) -> bool:
        return self.initialized

    async def prepare_rollout(self, rollout_id: int) -> None:
        self.prepared.append(rollout_id)

    async def prepare_eval(self) -> None:
        return None

    async def dispose(self) -> None:
        return None

    async def offload(self, tags: list[str] | None = None) -> None:
        return None

    async def onload(self, tags: list[str] | None = None) -> None:
        return None

    async def get_cell_statuses(self) -> dict[str, CellStatus]:
        return {}


class FakeTrainerController:
    def __init__(self) -> None:
        self.initialized = False
        self.trained: list[tuple[int, dict[str, Any]]] = []

    async def init(self, args: Pickled) -> list[Any]:
        self.initialized = args
        return [5]

    async def train(
        self, rollout_id: int, rollout_data_pack: dict[str, Any], external_data: list[Any] | None = None
    ) -> list[Any]:
        self.trained.append((rollout_id, rollout_data_pack))
        return []

    async def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        return None

    async def update_weights(self, rollout_id: int | None = None) -> None:
        return None

    async def onload(self) -> None:
        return None

    async def offload(self) -> None:
        return None

    async def clear_memory(self) -> None:
        return None

    async def reconcile_adapters(self) -> None:
        return None

    async def get_train_parallel_config(self) -> dict[str, Any]:
        return {"dp_size": 2}

    async def get_cell_statuses(self) -> dict[str, CellStatus]:
        return {}

    async def dispose(self) -> None:
        return None


class FakePodApi:
    def __init__(self, pods: list[Pod]) -> None:
        self.pods = pods
        self.selectors: list[tuple[str, str]] = []

    async def list_pods(self, *, namespace: str, label_selector: str) -> PodListPage:
        self.selectors.append((namespace, label_selector))
        return PodListPage(pods=list(self.pods), resource_version="1")

    async def stream_pods(self, *, namespace, label_selector, resource_version, timeout_seconds):
        await asyncio.sleep(3600)
        yield None


class _PerHostTransport(httpx.AsyncBaseTransport):
    def __init__(self, apps_by_host: dict[str, Any]) -> None:
        self._transports = {host: httpx.ASGITransport(app=app) for host, app in apps_by_host.items()}
        self.hosts_called: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        self.hosts_called.append(host)
        transport = self._transports.get(host)
        if transport is None:
            raise httpx.ConnectError(f"nothing listens on {host}", request=request)
        return await transport.handle_async_request(request)


def trainer_spec(*, num_workers_per_cell: int, num_gpus_per_node: int) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name=POOL,
        category=POOL_CATEGORY_TRAINER_ENGINE,
        port_infos=[PortInfo(name="master", static_port=9000, mode="master")],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=num_workers_per_cell,
            num_gpus_per_worker=1,
            num_gpu_slots_per_worker=1,
            num_gpus_per_node=num_gpus_per_node,
        ),
        worker_class=f"{__name__}.FakeTrainWorker",
        ctor_kwargs=lambda context: {},
    )


def cell_pods(count: int) -> list[Pod]:
    return [
        make_pod(
            name=f"{POOL}-0-{index}",
            pool_id=POOL,
            cell_id_suffix="0",
            pod_in_cell_index=str(index),
            pod_ip=f"10.0.0.{index + 1}",
        )
        for index in range(count)
    ]


def rollout_executor_args() -> SimpleNamespace:
    return SimpleNamespace(
        cluster_backend="kubernetes",
        pin_rollout_manager_to_head=False,
        debug_train_only=True,
        use_critic=False,
        kl_coef=0,
        use_kl_loss=False,
        use_opd=False,
        opd_type="megatron",
    )


@pytest.fixture
def deleted(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    recorded: list[list[str]] = []

    async def fake_delete_pods(*, namespace: str, pod_names: list[str]) -> None:
        recorded.append(list(pod_names))

    monkeypatch.setattr(cell_operations_kubernetes, "_delete_pods", fake_delete_pods)
    return recorded


def install(monkeypatch: pytest.MonkeyPatch, *, pods: list[Pod], workers_per_pod: int = 1):
    monkeypatch.setenv(NAMESPACE_ENV_VAR, NAMESPACE)
    monkeypatch.setenv(RELEASE_ENV_VAR, _RELEASE)
    fake_pod_api.reset()
    fake_pod_api.install(FakePodApi(pods))
    monkeypatch.setattr(core_provider, "_kubernetes_pod_api", fake_pod_api.installed)
    monkeypatch.setattr(specs_rollout, "ROLLOUT_EXECUTOR_WORKER_CLASS", f"{__name__}.FakeRolloutExecutor")
    monkeypatch.setattr(specs_inference, "INFERENCE_CONTROLLER_WORKER_CLASS", f"{__name__}.FakeInferenceController")
    monkeypatch.setattr(specs_train, "TRAINER_CONTROLLER_WORKER_CLASS", f"{__name__}.FakeTrainerController")

    specs = [
        trainer_spec(num_workers_per_cell=len(pods) * workers_per_pod, num_gpus_per_node=workers_per_pod),
        specs_rollout.spec_rollout_executor(rollout_executor_args()),
        specs_inference.spec_inference_controller(rollout_executor_args()),
        specs_train.spec_trainer_controller_actor(rollout_executor_args()),
    ]
    return compute_helm_backend_capability(specs=specs)


async def _ignore_cell(cell_id, info) -> None:
    return None


def installed_cells_provider(capability):
    return capability.dynamic_worker_provider(pool_ids=[POOL])


class TestKubernetesDriverAssembly:
    def test_the_installed_capability_answers_every_component_of_the_process(self, monkeypatch):
        """A run announces its backend once, and every later component has to see that answer."""
        capability = install(monkeypatch, pods=cell_pods(2))

        provider = installed_cells_provider(capability)

        assert isinstance(provider, KubernetesWorkerProvider)
        assert provider._run is installed_cells_provider(capability)._run

    def test_the_watch_is_scoped_to_this_release_and_its_pools(self, monkeypatch):
        """The selector is the only thing keeping one run from healing the cells of another run's release."""
        provider = installed_cells_provider(install(monkeypatch, pods=cell_pods(2)))
        api = fake_pod_api.current()

        async def scenario():
            stop = await provider.watch_cells(_ignore_cell)
            await stop()

        asyncio.run(scenario())

        namespace, selector = api.selectors[0]
        assert namespace == NAMESPACE
        assert selector == f"{env.INSTANCE_LABEL}={_RELEASE},{env.DEFAULT_LABEL_KEYS.pool_id} in ({POOL})"

    def test_refuses_to_hand_out_a_provider_for_a_pool_it_does_not_watch(self, monkeypatch):
        """A provider that silently watches nothing would leave those cells unhealed forever."""
        capability = install(monkeypatch, pods=cell_pods(2))

        with pytest.raises(AssertionError, match="are not pool_ids"):
            capability.dynamic_worker_provider(pool_ids=["some-other-pool_id"])

    def test_observing_a_cell_yields_rank_ordered_workers_with_handles(self, monkeypatch):
        """This is what a trainer cell is built from, so the order and the handles are the whole product."""
        provider = installed_cells_provider(install(monkeypatch, pods=cell_pods(3)))

        async def scenario():
            stop = await provider.watch_cells(_ignore_cell)
            try:
                return provider.get_worker_infos(cell_ids=[CELL_ID])[0]
            finally:
                await stop()

        infos = asyncio.run(scenario())

        assert [info.name for info in infos] == [f"{POOL}-0-{index}" for index in range(3)]
        assert [info.self_addrs["rpc"].host for info in infos] == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
        assert all(isinstance(info.handle, RpcWorkerHandle) for info in infos)

    def test_one_pod_serving_several_ranks_yields_one_worker_per_rank(self, monkeypatch):
        """A trainer pod supervises one process per gpu, and a cell that saw one of them would hang the collective."""
        provider = installed_cells_provider(install(monkeypatch, pods=cell_pods(1), workers_per_pod=2))

        async def scenario():
            stop = await provider.watch_cells(_ignore_cell)
            try:
                return provider.get_worker_infos(cell_ids=[CELL_ID])[0]
            finally:
                await stop()

        infos = asyncio.run(scenario())

        assert [info.name for info in infos] == [f"{POOL}-0-0", f"{POOL}-0-1"]
        assert [(info.self_addrs["rpc"].host, info.self_addrs["rpc"].port) for info in infos] == [
            ("10.0.0.1", 8000),
            ("10.0.0.1", 8001),
        ]
        assert [info.handle._transport._server_url for info in infos] == [
            "http://10.0.0.1:8000",
            "http://10.0.0.1:8001",
        ]
        assert [info.self_addrs["master"].port for info in infos] == [9000, 9000]

    def test_a_trainer_cell_builds_and_drives_its_ranks_over_rpc(self, monkeypatch: pytest.MonkeyPatch):
        """The point of the whole assembly: training over workers the platform created, not ones miles launched."""
        provider = installed_cells_provider(install(monkeypatch, pods=cell_pods(2)))

        workers = [FakeTrainWorker(), FakeTrainWorker()]
        apps = {f"10.0.0.{index + 1}": create_rpc_app(worker) for index, worker in enumerate(workers)}
        transport = _PerHostTransport(apps)

        async def scenario():
            async with httpx.AsyncClient(transport=transport) as client:
                monkeypatch.setattr(http_utils.GeneralHttpClientProvider, "client", classmethod(lambda cls: client))
                for app in apps.values():
                    await app.router.lifespan_context(app).__aenter__()
                stop = await provider.watch_cells(_ignore_cell)
                try:
                    cell = TrainerCell(
                        args=SimpleNamespace(),
                        role="actor",
                        with_ref=False,
                        cell_id=CELL_ID,
                        cell_index=0,
                        workers_hash="hash-1",
                        health_checker=SimpleNamespace(start=lambda: None, status=None),
                        provider=provider,
                    )
                    return cell, await cell.execute(
                        "configure_master_addr_and_port", master_addr="10.0.0.1", master_port=9000
                    )
                finally:
                    await stop()

        cell, results = asyncio.run(scenario())

        assert results == [9000, 9000]
        assert [worker.configured for worker in workers] == [[("10.0.0.1", 9000)], [("10.0.0.1", 9000)]]
        assert set(transport.hosts_called) == {"10.0.0.1", "10.0.0.2"}

    def test_healing_a_cell_deletes_its_pods(self, monkeypatch, deleted):
        """Under Kubernetes a cell heals because its workload recreates the pods somebody deleted."""
        capability = install(monkeypatch, pods=cell_pods(2))

        operations = capability.cell_operations()
        asyncio.run(operations.suspend(cell_id=CELL_ID))

        assert deleted == [[f"{POOL}-0-0", f"{POOL}-0-1"]]

    def test_the_rollout_executor_answers_over_rpc_with_no_actor_behind_it(self, monkeypatch: pytest.MonkeyPatch):
        """Under Kubernetes the executor is a pod in the address book, not an actor the driver creates."""
        capability = install(monkeypatch, pods=cell_pods(2))

        executor = FakeRolloutExecutor()
        host = STATIC_HOSTS[specs_rollout.ROLLOUT_EXECUTOR_POOL_ID]
        app = create_rpc_app(executor)
        transport = _PerHostTransport({host: app})

        async def scenario():
            async with httpx.AsyncClient(transport=transport) as client:
                monkeypatch.setattr(http_utils.GeneralHttpClientProvider, "client", classmethod(lambda cls: client))
                await app.router.lifespan_context(app).__aenter__()
                handle = specs_rollout.create_rollout_executor_handle(capability=capability)
                await handle.set_train_parallel_config(config={"dp_size": 4})
                await handle.load(rollout_id=11)
                return handle, await handle.get(rollout_id=3)

        handle, rollout_data = asyncio.run(scenario())

        assert isinstance(handle, RpcWorkerHandle)
        assert rollout_data == {"sample_indices": [3], "data_ref": "ref-3"}
        assert executor.train_parallel_config == {"dp_size": 4}
        assert executor.loaded == [11]
        assert set(transport.hosts_called) == {host}

    def test_the_rollout_components_assemble_around_those_handles(self, monkeypatch: pytest.MonkeyPatch):
        """create_rollout_components is the driver's door into rollout, so it too must open over rpc."""
        capability = install(monkeypatch, pods=cell_pods(2))

        controller = FakeInferenceController()
        executor = FakeRolloutExecutor()
        executor_host = STATIC_HOSTS[specs_rollout.ROLLOUT_EXECUTOR_POOL_ID]
        controller_host = STATIC_HOSTS[specs_inference.INFERENCE_CONTROLLER_POOL_ID]
        apps = {
            executor_host: create_rpc_app(executor),
            controller_host: create_rpc_app(controller),
        }
        transport = _PerHostTransport(apps)
        args = SimpleNamespace(
            cluster_backend="kubernetes",
            pin_rollout_manager_to_head=False,
            num_rollout=None,
            num_epoch=3,
            debug_train_only=True,
        )
        monkeypatch.setattr(placement_group, "get_backend_capability", lambda _args: capability)

        async def scenario():
            async with httpx.AsyncClient(transport=transport) as client:
                monkeypatch.setattr(http_utils.GeneralHttpClientProvider, "client", classmethod(lambda cls: client))
                for app in apps.values():
                    await app.router.lifespan_context(app).__aenter__()
                return await create_rollout_components(args)

        result = asyncio.run(scenario())

        assert isinstance(result.inference_controller, RpcWorkerHandle)
        assert isinstance(result.rollout_executor, RpcWorkerHandle)
        assert controller.initialized
        assert executor.initialized
        assert result.num_rollout_per_epoch == 7
        assert args.num_rollout == 21

    def test_the_trainer_controller_answers_over_rpc_with_no_actor_behind_it(self, monkeypatch: pytest.MonkeyPatch):
        """Under Kubernetes the trainer controller is a pod the driver addresses, not an object it builds."""
        capability = install(monkeypatch, pods=cell_pods(2))

        controller = FakeTrainerController()
        host = STATIC_HOSTS[specs_train.compute_trainer_controller_pool_id("actor")]
        app = create_rpc_app(controller)
        transport = _PerHostTransport({host: app})

        async def scenario():
            async with httpx.AsyncClient(transport=transport) as client:
                monkeypatch.setattr(http_utils.GeneralHttpClientProvider, "client", classmethod(lambda cls: client))
                await app.router.lifespan_context(app).__aenter__()
                handle = specs_train.create_trainer_controller_handle(capability=capability, role="actor")
                assert await handle.init(Namespace(num_rollout=7)) == [5]
                await handle.train(rollout_id=3, rollout_data_pack={"sample_indices": [0], "data_ref": "ref-3"})
                return handle, await handle.get_train_parallel_config()

        handle, parallel_config = asyncio.run(scenario())

        assert isinstance(handle, RpcWorkerHandle)
        assert controller.initialized.num_rollout == 7
        assert controller.trained == [(3, {"sample_indices": [0], "data_ref": "ref-3"})]
        assert parallel_config == {"dp_size": 2}
        assert set(transport.hosts_called) == {host}
