from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_provider import static
from miles.utils.workers.worker_provider.kubernetes.helm import naming
from miles.utils.workers.worker_provider.static import StaticWorkerProvider, parse_host_and_port
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec, ServeWorkerSpec

_RELEASE = "miles-run-c0ffee"
_ADDR_POOL_ID = "trainer-controller-actor"


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
    return StaticWorkerProvider.of_release(release=_RELEASE, spec=spec or _served_spec())


def _addr_provider(*addrs: str) -> StaticWorkerProvider:
    return StaticWorkerProvider.of_rpc_addrs(
        pool_id=_ADDR_POOL_ID,
        addrs=[parse_host_and_port(addr) for addr in addrs or ("10.0.0.1:8000",)],
        worker_class=f"{__name__}.FakeController",
    )


class TestAddresses:
    def test_recomputes_the_host_the_chart_gave_the_cell(self):
        """The launcher and the pod run the same naming function, so no address has to be shipped between them."""
        addrs = asyncio.run(_provider().get_addrs("trainer-controller-00000-00000"))

        assert addrs["primary"].host == naming.static_worker_host(_RELEASE, "trainer-controller", 0)
        assert addrs["primary"].port == 7000

    def test_each_cell_answers_on_its_own_host(self):
        """Every cell of a static pool is its own workload object, so cell index selects the hostname."""
        second = asyncio.run(_provider().get_addrs("trainer-controller-00001-00000"))

        assert second["primary"].host == naming.static_worker_host(_RELEASE, "trainer-controller", 1)

    def test_carries_every_port_the_spec_declares(self):
        """A caller that asked for a port the spec declares must not have to know its number."""
        addrs = asyncio.run(_provider().get_addrs("trainer-controller-00000-00000"))

        assert sorted(addrs) == ["primary", "rpc"]

    def test_refuses_a_worker_of_another_pool(self):
        """One provider answers one pool, and guessing across pools would return a plausible wrong host."""
        with pytest.raises(AssertionError, match="inference-router-0-00000-00000"):
            asyncio.run(_provider().get_addrs("inference-router-0-00000-00000"))

    def test_refuses_a_worker_the_cell_never_holds(self):
        """A worker index beyond the cell size would be answered with cell zero's ports."""
        with pytest.raises(AssertionError, match="trainer-controller-00000-00003"):
            asyncio.run(_provider().get_addrs("trainer-controller-00000-00003"))

    def test_refuses_a_cell_the_run_never_deployed(self):
        """A cell beyond the replica count has no workload object, so its host would never resolve."""
        with pytest.raises(AssertionError, match="trainer-controller-00007-00000"):
            asyncio.run(_provider().get_addrs("trainer-controller-00007-00000"))


class TestReleaseNaming:
    def test_two_runs_that_share_a_truncated_prefix_get_different_objects(self):
        """The trailing digits are exactly what tells two runs of the same experiment apart."""
        long_prefix = "miles-run-a-rather-long-experiment-name-260810-123456"

        first = naming.component_name(f"{long_prefix}-001", "trainer-controller")
        second = naming.component_name(f"{long_prefix}-002", "trainer-controller")

        assert first != second
        assert len(first) <= naming.COMPONENT_NAME_BUDGET

    def test_the_same_release_always_names_the_same_object(self):
        """helm upgrade only adopts an object it can find again under the same name."""
        release = "miles-run-a-rather-long-experiment-name-260810-123456-001"

        assert naming.component_name(release, "trainer-controller") == naming.component_name(
            release, "trainer-controller"
        )

    def test_a_release_that_fits_is_left_alone(self):
        """A readable name is worth keeping, and nothing is lost when nothing is truncated."""
        assert naming.component_name(_RELEASE, "trainer-controller") == f"{_RELEASE}-trainer-controller"


class TestHandles:
    def test_calls_a_served_worker_through_its_rpc_port(self):
        """A controller is reached by rpc, and the class it serves is named by its own spec."""
        handle = _provider().get_handle("trainer-controller-00000-00000")

        assert isinstance(handle, RpcWorkerHandle)
        assert handle._worker_cls_name == "FakeController"

    def test_refuses_to_call_a_pool_that_is_only_launched(self):
        """A command worker declares no worker class, so nothing describes its rpc methods."""
        with pytest.raises(AssertionError, match="launched as a command"):
            _provider(_command_spec()).get_handle("inference-router-0-00000-00000")


class TestEnumeration:
    def test_refuses_to_enumerate_workers(self):
        """Nothing observes a static pool, so a caller expecting live workers must fail loudly."""
        with pytest.raises(NotImplementedError, match="does not enumerate workers"):
            _provider().get_worker_infos(cell_ids=["trainer-controller-00000"])


class TestAddressesGivenExplicitly:
    def test_answers_the_address_it_was_given(self):
        """The whole point of addressing another deployment is that nothing discovers its address at runtime."""
        addrs = asyncio.run(_addr_provider().get_addrs(f"{_ADDR_POOL_ID}-00000-00000"))

        assert addrs["rpc"].addr == "http://10.0.0.1:8000"

    def test_addresses_each_instance_by_its_own_entry(self):
        """A composite controller is given several addresses, and cell one is not cell zero."""
        provider = _addr_provider("10.0.0.1:8000", "10.0.0.2:9000")

        assert (
            asyncio.run(provider.get_addrs(f"{_ADDR_POOL_ID}-00001-00000")).get("rpc").addr == "http://10.0.0.2:9000"
        )

    def test_refuses_a_worker_nobody_named(self):
        """Answering an address for an instance that was never given would invent a host out of thin air."""
        with pytest.raises(AssertionError, match=f"{_ADDR_POOL_ID}-00003-00000"):
            asyncio.run(_addr_provider().get_addrs(f"{_ADDR_POOL_ID}-00003-00000"))

    def test_refuses_a_second_worker_in_a_cell(self):
        """A statically addressed controller is one process, so worker one of its cell does not exist."""
        with pytest.raises(AssertionError, match=f"{_ADDR_POOL_ID}-00000-00001"):
            asyncio.run(_addr_provider().get_addrs(f"{_ADDR_POOL_ID}-00000-00001"))

    def test_builds_an_rpc_handle_on_the_given_address(self):
        """The orchestration script reaches an independently deployed controller only over rpc."""
        assert isinstance(_addr_provider().get_handle(f"{_ADDR_POOL_ID}-00000-00000"), RpcWorkerHandle)


class TestWaitStaticAddrsReady:
    async def test_dials_every_address_before_the_run_calls_it(self):
        """A deployment installed a moment earlier is not yet listening, and the first call would simply fail."""
        dialled: list[tuple[str, int]] = []

        async def _dial(host, port, *, timeout):
            dialled.append((host, port))

        with patch.object(static, "wait_tcp_ready_async", _dial):
            await static.wait_static_addrs_ready([parse_host_and_port("a:1"), parse_host_and_port("b:2")])

        assert sorted(dialled) == [("a", 1), ("b", 2)]

    async def test_a_slow_address_does_not_hold_up_the_others(self):
        """Waiting serially would multiply one ready budget by the number of deployments a run reaches."""
        order: list[str] = []

        async def _dial(host, port, *, timeout):
            await asyncio.sleep(0.02 if host == "slow" else 0)
            order.append(host)

        with patch.object(static, "wait_tcp_ready_async", _dial):
            await static.wait_static_addrs_ready([parse_host_and_port("slow:1"), parse_host_and_port("fast:2")])

        assert order == ["fast", "slow"]


class TestParseHostAndPort:
    @pytest.mark.parametrize(
        ("addr", "expected"),
        [
            ("10.0.0.1:8000", ("10.0.0.1", 8000)),
            ("host.namespace.svc:8000", ("host.namespace.svc", 8000)),
            ("[2001:db8::1]:8000", ("[2001:db8::1]", 8000)),
        ],
    )
    def test_reads_the_forms_a_user_writes_on_the_command_line(self, addr, expected):
        """Every one of these is what somebody copies out of a log line into the next launch."""
        parsed = parse_host_and_port(addr)

        assert (parsed.host, parsed.port) == expected

    def test_an_ipv6_address_survives_being_rebuilt_into_a_url(self):
        """The rpc client dials `.addr`, and an unbracketed ipv6 host makes that url unparseable."""
        assert parse_host_and_port("[2001:db8::1]:8000").addr == "http://[2001:db8::1]:8000"

    def test_refuses_an_address_without_a_port(self):
        """A controller is reached at a port, and guessing one would fail much later and much less clearly."""
        with pytest.raises(AssertionError, match="host:port"):
            parse_host_and_port("10.0.0.1")

    @pytest.mark.parametrize("addr", ["http://10.0.0.1:8000", "http://host.namespace.svc:8000/"])
    def test_refuses_an_address_written_as_a_url(self, addr):
        """A scheme is not part of an address here, and silently dropping it would hide a wrong flag value."""
        with pytest.raises(AssertionError, match="host:port"):
            parse_host_and_port(addr)

    def test_refuses_an_address_that_names_no_host(self):
        """A port with no host parses cleanly and then dials whatever the empty host resolves to."""
        with pytest.raises(AssertionError, match="no host"):
            parse_host_and_port(":8000")

    @pytest.mark.parametrize("addr", ["2001:db8::1", "fd00::1"])
    def test_refuses_a_bare_ipv6_address_written_without_brackets(self, addr):
        """Its own colons parse as the port separator, so it becomes a truncated host and a one-digit port."""
        with pytest.raises(AssertionError, match="ipv6"):
            parse_host_and_port(addr)
