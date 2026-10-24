from __future__ import annotations

import sys
from typing import Any

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.utils.workers.fake_specs import FakeServeSpec
from tests.fast.utils.workers.serving.registered_serve import pod_env, serve_config_argv
from tests.fast.utils.workers.serving.serve_smoke_worker import SmokeServeSpec, SmokeWorkerConfig

from miles.utils.function_registry import function_registry
from miles.utils.workers.connection_config import StaticConnConfig
from miles.utils.workers.serving import serve_inner
from miles.utils.workers.serving.worker_identity import SUBPROCESS_INDEX_ENV_VAR, read_worker_in_pod_index
from miles.utils.workers.worker_spec import BaseServeSpec, PortInfo, SchedulingSpec, WorkerCtorContext

WORKER_FN = "test:worker"

POOL_ID = "trainer-engine-actor"
RPC_PORT = 8000


class KeywordOnlyWorker:
    def __init__(self, *, args: str) -> None:
        self.args = args


class _DemoArgs:
    def __init__(self, *, flags: str, cluster_backend: str = "ray") -> None:
        self.flags = flags
        self.cluster_backend = cluster_backend


def _spec(*, args: Any, ctor_kwargs=None) -> BaseServeSpec:
    return FakeServeSpec(
        args=args,
        name=POOL_ID,
        port_infos=[PortInfo(name="rpc", static_port=RPC_PORT)],
        fixed_scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
        worker_class=WORKER_FN,
        make_ctor_kwargs=ctor_kwargs or (lambda context: dict(args=f"{POOL_ID}:{context.args.flags}")),
    )


def _serve_pod_of(spec: BaseServeSpec, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in pod_env(spec).items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def registered_functions():
    with function_registry.temporary(WORKER_FN, KeywordOnlyWorker):
        yield


class TestCreateWorker:
    def test_builds_a_keyword_only_worker_from_the_computed_kwargs(self, registered_functions, monkeypatch):
        """Every real served worker takes keyword arguments, and its spec computes them from the pool's config."""
        spec = _spec(args=_DemoArgs(flags="--rollout-num-gpus 8"))
        _serve_pod_of(spec, monkeypatch)

        worker = serve_inner.create_worker(spec, static_connections=StaticConnConfig(static_conn_infos={}))

        assert isinstance(worker, KeywordOnlyWorker)
        assert worker.args == "trainer-engine-actor:--rollout-num-gpus 8"

    def test_refuses_a_worker_type_the_run_does_not_describe(self, monkeypatch):
        """The pod and the launcher would otherwise disagree silently about what this process serves."""
        own_argv = serve_config_argv(
            spec_class=SmokeServeSpec, config=SmokeWorkerConfig(rpc_port=8000, worker_argv=[])
        )
        monkeypatch.setattr(sys, "argv", ["serve_inner", *own_argv])

        with pytest.raises(KeyError, match=SmokeServeSpec.worker_type):
            serve_inner.main()


class TestDeferredCapability:
    def test_the_backend_is_built_only_when_a_spec_asks_for_a_provider(
        self, registered_functions, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every served worker is handed this capability, and most specs never look at it."""
        built: list[tuple[str, StaticConnConfig]] = []
        captured: dict[str, Any] = {}
        static_connections = StaticConnConfig(static_conn_infos={})

        def _build(*, cluster_backend, static_connections):
            built.append((cluster_backend.value, static_connections))
            return FakeBackendCapability(cells_provider=object())

        def _capture(context: WorkerCtorContext) -> dict[str, Any]:
            captured["capability"] = context.capability
            return dict(args="captured")

        spec = _spec(args=_DemoArgs(flags=""), ctor_kwargs=_capture)
        _serve_pod_of(spec, monkeypatch)
        monkeypatch.setattr(serve_inner, "get_backend_capability", _build)

        serve_inner.create_worker(spec, static_connections=static_connections)
        capability = captured["capability"]
        assert built == []

        capability.dynamic_worker_provider(pool_ids=["engine"])
        capability.dynamic_worker_provider(pool_ids=["engine"])

        assert built == [("ray", static_connections)]


class TestRpcPortOfARank:
    def test_the_workers_of_one_pod_listen_on_different_ports(self, monkeypatch):
        """The supervisor runs them all in one network namespace, so a shared port is a bind failure."""
        spec = _spec(args=_DemoArgs(flags=""))
        _serve_pod_of(spec, monkeypatch)
        ports = [
            serve_inner._rpc_port_of(spec).effective_static_port(
                worker_in_pod_index=read_worker_in_pod_index({SUBPROCESS_INDEX_ENV_VAR: str(index)})
            )
            for index in range(4)
        ]

        assert ports == [8000, 8001, 8002, 8003]

    def test_the_first_worker_keeps_the_port_the_address_book_predicts(self, monkeypatch):
        """The provider addresses a pod at the spec's static rpc port, which worker zero has to answer on."""
        spec = _spec(args=_DemoArgs(flags=""))
        _serve_pod_of(spec, monkeypatch)

        assert (
            serve_inner._rpc_port_of(spec).effective_static_port(worker_in_pod_index=read_worker_in_pod_index({}))
            == RPC_PORT
        )
