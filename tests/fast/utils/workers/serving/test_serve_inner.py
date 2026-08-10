import socket
from types import SimpleNamespace
from typing import Any

import pytest

from miles.utils.workers.serving import serve_inner
from miles.utils.workers.serving.serve_inner import _rpc_port_of, parse_own_args
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec

SPECS_PATH = "tests.fast.utils.workers.e2e.e2e_worker.compute_specs"
POOL_ID = "e2e-pool"


def _serve_spec() -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name=POOL_ID,
        port_infos=[PortInfo(name="rpc", static_port=8000)],
        env_var=lambda context: {},
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        worker_class="test.worker",
        ctor_kwargs=lambda context: {},
    )


class _FakeServerSocket:
    def __init__(self, address: tuple[str, int]) -> None:
        self.address = address
        self.closed = False

    def getsockname(self) -> tuple[str, int]:
        return self.address

    def __enter__(self) -> "_FakeServerSocket":
        return self

    def __exit__(self, *args: Any) -> None:
        self.closed = True


class TestParseOwnArgs:
    def test_the_spec_table_and_the_pool_it_serves_are_read(self) -> None:
        """These two are the whole of what the pod needs to find the one spec it is a worker of."""
        args = parse_own_args(["--specs", SPECS_PATH, "--pool-id", POOL_ID])

        assert (args.specs, args.pool_id) == (SPECS_PATH, POOL_ID)

    def test_an_omitted_pool_id_is_a_usage_error(self) -> None:
        """A process that does not know which pool it serves would pick a spec at random."""
        with pytest.raises(SystemExit) as exc_info:
            parse_own_args(["--specs", SPECS_PATH])

        assert exc_info.value.code == 2

    def test_an_omitted_spec_table_is_a_usage_error(self) -> None:
        """Without the run's spec table there is nothing to match the pool id against."""
        with pytest.raises(SystemExit) as exc_info:
            parse_own_args(["--pool-id", POOL_ID])

        assert exc_info.value.code == 2

    def test_unknown_inner_option_is_a_usage_error(self) -> None:
        """The inner entrypoint rejects an option it does not define instead of ignoring it."""
        with pytest.raises(SystemExit) as exc_info:
            parse_own_args(["--specs", SPECS_PATH, "--pool-id", POOL_ID, "--unknown-option", "1"])

        assert exc_info.value.code == 2


def _served(monkeypatch: pytest.MonkeyPatch, *, has_dualstack_ipv6: bool) -> dict[str, Any]:
    served: dict[str, Any] = {}
    monkeypatch.setattr(serve_inner.sys, "argv", ["serve_inner", "--specs", SPECS_PATH, "--pool-id", POOL_ID])
    monkeypatch.setattr(serve_inner.socket, "has_dualstack_ipv6", lambda: has_dualstack_ipv6)
    monkeypatch.setattr(serve_inner, "compute_serve_worker_spec", lambda **kwargs: SimpleNamespace(worker_class="w"))
    monkeypatch.setattr(serve_inner, "create_worker", lambda spec, **kwargs: object())
    monkeypatch.setattr(serve_inner, "create_rpc_app", lambda worker: "app")
    monkeypatch.setattr(serve_inner, "read_worker_in_pod_index", lambda environ: 0)
    monkeypatch.setattr(
        serve_inner,
        "_rpc_port_of",
        lambda spec: SimpleNamespace(effective_static_port=lambda worker_in_pod_index: 8123),
    )
    server_socket: _FakeServerSocket | None = None

    def create_server(address: tuple[str, int], **kwargs: Any) -> _FakeServerSocket:
        nonlocal server_socket
        served.update(host=address[0], port=address[1], socket_kwargs=kwargs)
        server_socket = _FakeServerSocket(address)
        return server_socket

    def create_uvicorn_server(config: Any) -> SimpleNamespace:
        served["config"] = config
        return SimpleNamespace(run=lambda *, sockets: served.update(sockets=sockets))

    monkeypatch.setattr(serve_inner.socket, "create_server", create_server)
    monkeypatch.setattr(serve_inner.uvicorn, "Config", lambda app: served.update(app=app) or "config")
    monkeypatch.setattr(serve_inner.uvicorn, "Server", create_uvicorn_server)

    serve_inner.main()
    assert server_socket is not None
    served["socket_closed"] = server_socket.closed
    return served


class TestTheAddressAWorkerIsServedOn:
    def test_binds_the_dual_stack_wildcard_where_the_platform_offers_one(self, monkeypatch):
        """The cell view publishes the pod ip, and on an ipv6-only cluster that is an ipv6 address."""
        served = _served(monkeypatch, has_dualstack_ipv6=True)

        assert served["host"] == serve_inner.IPV6_WILDCARD_HOST
        assert served["socket_kwargs"] == {"family": socket.AF_INET6, "dualstack_ipv6": True}

    def test_binds_the_ipv4_wildcard_where_ipv6_is_unavailable(self, monkeypatch):
        """Asking for the dual-stack wildcard where there is no ipv6 stack leaves the worker unserved."""
        served = _served(monkeypatch, has_dualstack_ipv6=False)

        assert served["host"] == serve_inner.IPV4_WILDCARD_HOST
        assert served["socket_kwargs"] == {"family": socket.AF_INET}

    def test_the_dual_stack_wildcard_is_the_unspecified_ipv6_address(self):
        """Only the unspecified address accepts the ipv4-mapped connections an ipv4 client makes."""
        assert (serve_inner.IPV6_WILDCARD_HOST, serve_inner.IPV4_WILDCARD_HOST) == ("::", "0.0.0.0")

    def test_serves_the_rpc_port_the_spec_declares_whichever_wildcard_it_binds(self, monkeypatch):
        """The address a client dials is the published pod ip and this port, so the port may not move."""
        dual_stack = _served(monkeypatch, has_dualstack_ipv6=True)
        ipv4 = _served(monkeypatch, has_dualstack_ipv6=False)

        assert dual_stack["port"] == 8123
        assert dual_stack["sockets"][0].address == (serve_inner.IPV6_WILDCARD_HOST, 8123)
        assert dual_stack["socket_closed"] is True
        assert ipv4["port"] == 8123
        assert ipv4["sockets"][0].address == (serve_inner.IPV4_WILDCARD_HOST, 8123)
        assert ipv4["socket_closed"] is True


class TestCreateServerSocket:
    def test_dual_stack_listener_accepts_ipv4_and_ipv6_when_ipv6_sockets_default_to_v6_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicitly dual-stack listener accepts both address families despite a v6-only default."""
        real_socket = socket.socket

        def create_v6_only_socket(*args: Any, **kwargs: Any) -> socket.socket:
            server_socket = real_socket(*args, **kwargs)
            if server_socket.family == socket.AF_INET6:
                server_socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            return server_socket

        monkeypatch.setattr(serve_inner.socket, "socket", create_v6_only_socket)
        monkeypatch.setattr(serve_inner.socket, "has_dualstack_ipv6", lambda: True)

        with serve_inner._create_server_socket(port=0) as server_socket:
            port = server_socket.getsockname()[1]

            assert server_socket.family == socket.AF_INET6
            assert server_socket.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 0
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                pass
            with socket.create_connection(("::1", port), timeout=1):
                pass

    def test_ipv4_listener_remains_reachable_without_dual_stack_support(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A host without dual-stack IPv6 still accepts worker RPC calls over IPv4."""
        monkeypatch.setattr(serve_inner.socket, "has_dualstack_ipv6", lambda: False)

        with serve_inner._create_server_socket(port=0) as server_socket:
            port = server_socket.getsockname()[1]

            assert server_socket.family == socket.AF_INET
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                pass


class TestRpcPortOf:
    @pytest.mark.parametrize(
        "port_infos, expected_count",
        [
            ([PortInfo(name="metrics", static_port=9000)], 0),
            ([PortInfo(name="rpc", static_port=8000), PortInfo(name="rpc", static_port=8001)], 2),
        ],
    )
    def test_a_spec_without_exactly_one_rpc_port_is_rejected(
        self, port_infos: list[PortInfo], expected_count: int
    ) -> None:
        """A served spec with a missing or ambiguous rpc port cannot choose a listening port."""
        spec = _serve_spec().model_copy(update={"port_infos": port_infos})

        with pytest.raises(AssertionError, match=rf"declares {expected_count} rpc ports"):
            _rpc_port_of(spec)
