from __future__ import annotations

import asyncio

import pytest

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_provider.kubernetes.helm.naming import static_worker_host
from miles.utils.workers.worker_provider.static import StaticWorkerProvider
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec, ServeWorkerSpec

RELEASE = "miles-run-c0ffee"


class FakeController:
    def health(self) -> int:
        return 1


def _served_spec(*, num_cells: int = 2) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name="trainer-controller",
        port_infos=[PortInfo(name="primary", static_port=7000), PortInfo(name="rpc", static_port=8000)],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(num_cells=num_cells, num_workers_per_cell=1, num_gpus_per_worker=0),
        worker_class=f"{__name__}.FakeController",
        ctor_kwargs=lambda context: {},
    )


def _command_spec() -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name="inference-router-0",
        port_infos=[PortInfo(name="primary", static_port=8000)],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
        launch_command=lambda context: "python -m router",
    )


def _provider(spec=None) -> StaticWorkerProvider:
    return StaticWorkerProvider(release=RELEASE, spec=spec or _served_spec())


class TestAddresses:
    def test_recomputes_the_host_the_chart_gave_the_cell(self):
        """The launcher and the pod run the same naming function, so no address has to be shipped between them."""
        addrs = asyncio.run(_provider().get_addrs("trainer-controller-0-0"))

        assert addrs["primary"].host == static_worker_host(RELEASE, "trainer-controller", 0)
        assert addrs["primary"].port == 7000

    def test_each_cell_answers_on_its_own_host(self):
        """Every cell of a static pool is its own workload object, so cell index selects the hostname."""
        second = asyncio.run(_provider().get_addrs("trainer-controller-1-0"))

        assert second["primary"].host == static_worker_host(RELEASE, "trainer-controller", 1)

    def test_carries_every_port_the_spec_declares(self):
        """A caller that asked for a port the spec declares must not have to know its number."""
        addrs = asyncio.run(_provider().get_addrs("trainer-controller-0-0"))

        assert sorted(addrs) == ["primary", "rpc"]

    def test_refuses_a_worker_of_another_pool(self):
        """One provider answers one pool, and guessing across pools would return a plausible wrong host."""
        with pytest.raises(AssertionError, match="answers for pool trainer-controller"):
            asyncio.run(_provider().get_addrs("inference-router-0-0-0"))

    def test_refuses_a_cell_the_run_never_deployed(self):
        """A cell beyond the replica count has no workload object, so its host would never resolve."""
        with pytest.raises(AssertionError, match="deploys 2 cells"):
            asyncio.run(_provider().get_addrs("trainer-controller-7-0"))


class TestHandles:
    def test_calls_a_served_worker_through_its_rpc_port(self):
        """A controller is reached by rpc, and the class it serves is named by its own spec."""
        handle = _provider().get_handle("trainer-controller-0-0")

        assert isinstance(handle, RpcWorkerHandle)
        assert handle._worker_cls_name == "FakeController"

    def test_refuses_to_call_a_pool_that_is_only_launched(self):
        """A command worker declares no worker class, so nothing describes its rpc methods."""
        with pytest.raises(AssertionError, match="launched as a command"):
            _provider(_command_spec()).get_handle("inference-router-0-0-0")


class TestEnumeration:
    def test_refuses_to_enumerate_workers(self):
        """Nothing observes a static pool, so a caller expecting live workers must fail loudly."""
        with pytest.raises(NotImplementedError, match="does not enumerate workers"):
            _provider().get_worker_infos(cell_ids=["trainer-controller-0"])
