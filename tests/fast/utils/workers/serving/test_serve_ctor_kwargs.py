from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from tests.fast.fixtures.kubernetes_fixtures import _RELEASE, NAMESPACE, install_workers

from miles.utils.function_registry import function_registry
from miles.utils.workers.env_vars import CELL_INDEX_ENV_VAR
from miles.utils.workers.serving import serve_inner
from miles.utils.workers.serving.utils import compute_serve_worker_spec
from miles.utils.workers.serving.worker_identity import SUBPROCESS_INDEX_ENV_VAR, read_worker_in_pod_index
from miles.utils.workers.worker_provider.kubernetes.helm.env import NAMESPACE_ENV_VAR, RELEASE_ENV_VAR
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec

SPECS_FN = "test:specs"
WORKER_FN = "test:worker"

POOL_ID = "trainer-engine-actor"
RPC_PORT = 8000


class KeywordOnlyWorker:
    def __init__(self, *, args: str) -> None:
        self.args = args


def compute_specs(worker_argv: list[str]) -> list[ServeWorkerSpec]:
    return [
        ServeWorkerSpec(
            name=POOL_ID,
            port_infos=[PortInfo(name="rpc", static_port=RPC_PORT)],
            env_var=lambda context: {},
            scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
            worker_class=WORKER_FN,
            ctor_kwargs=lambda context: dict(args=f"{POOL_ID}:{' '.join(worker_argv)}"),
        )
    ]


@pytest.fixture(autouse=True)
def pod_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every pod is told which cell of its pool it serves before serve runs."""
    monkeypatch.setenv(CELL_INDEX_ENV_VAR, "0")


@pytest.fixture
def registered_functions():
    with function_registry.temporary(SPECS_FN, compute_specs):
        with function_registry.temporary(WORKER_FN, KeywordOnlyWorker):
            yield


def worker_of(worker_argv: list[str]) -> Any:
    spec = compute_serve_worker_spec(specs_fn=SPECS_FN, pool_id=POOL_ID, worker_argv=worker_argv)
    return serve_inner.create_worker(spec, specs_fn=SPECS_FN, worker_argv=worker_argv)


class TestCreateWorker:
    def test_builds_a_keyword_only_worker_from_the_computed_kwargs(self, registered_functions):
        """Every real served worker takes keyword arguments, so handing it the argv positionally is a TypeError."""
        worker = worker_of(["--rollout-num-gpus", "8"])

        assert isinstance(worker, KeywordOnlyWorker)
        assert worker.args == "trainer-engine-actor:--rollout-num-gpus 8"

    def test_refuses_a_pool_the_run_does_not_describe(self, registered_functions):
        """The pod and the launcher would otherwise disagree silently about what this process serves."""
        with pytest.raises(AssertionError, match="not one spec named"):
            compute_serve_worker_spec(specs_fn=SPECS_FN, pool_id="no-such-pool", worker_argv=[])


class TestDeferredCapability:
    def test_the_backend_is_built_only_when_a_spec_asks_for_a_provider(
        self, registered_functions, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every served worker is handed this capability, and most specs never look at it."""
        built: list[int] = []
        captured: dict[str, Any] = {}

        def _install(*, specs, cluster_backend):
            built.append(len(specs))
            return install_workers()

        monkeypatch.setenv(NAMESPACE_ENV_VAR, NAMESPACE)
        monkeypatch.setenv(RELEASE_ENV_VAR, _RELEASE)
        monkeypatch.setattr(serve_inner, "get_backend_capability", _install)
        monkeypatch.setattr(serve_inner, "parse_args", lambda: SimpleNamespace(cluster_backend="ray"))

        serve_inner.create_worker(_capturing_spec(captured), specs_fn=SPECS_FN, worker_argv=[])
        capability = captured["capability"]
        assert built == []

        capability.dynamic_worker_provider(pool_ids=["engine"])
        capability.dynamic_worker_provider(pool_ids=["engine"])

        assert built == [1]


class TestRpcPortOfARank:
    def test_the_workers_of_one_pod_listen_on_different_ports(self, registered_functions):
        """The supervisor runs them all in one network namespace, so a shared port is a bind failure."""
        spec = compute_serve_worker_spec(specs_fn=SPECS_FN, pool_id=POOL_ID, worker_argv=[])
        ports = [
            serve_inner._rpc_port_of(spec) + read_worker_in_pod_index({SUBPROCESS_INDEX_ENV_VAR: str(index)})
            for index in range(4)
        ]

        assert ports == [8000, 8001, 8002, 8003]

    def test_the_first_worker_keeps_the_port_the_address_book_predicts(self, registered_functions):
        """The provider addresses a pod at the spec's static rpc port, which worker zero has to answer on."""
        spec = compute_serve_worker_spec(specs_fn=SPECS_FN, pool_id=POOL_ID, worker_argv=[])

        assert serve_inner._rpc_port_of(spec) + read_worker_in_pod_index({}) == RPC_PORT


def _capturing_spec(captured: dict[str, Any]) -> ServeWorkerSpec:
    def ctor_kwargs(context) -> dict[str, Any]:
        captured["capability"] = context.capability
        return dict(args="captured")

    return ServeWorkerSpec(
        name=POOL_ID,
        port_infos=[PortInfo(name="rpc", static_port=RPC_PORT)],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
        worker_class=WORKER_FN,
        ctor_kwargs=ctor_kwargs,
    )
