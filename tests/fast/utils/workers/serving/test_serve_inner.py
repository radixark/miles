import socket
from types import SimpleNamespace
from typing import Any

import pytest
from tests.fast.utils.workers.serving.registered_serve import serve_config_argv
from tests.fast.utils.workers.serving.serve_smoke_worker import SmokeServeSpec, SmokeWorkerConfig

from miles.ray.specs.entrypoint import SERVE_SPEC_CLASSES
from miles.utils.workers.connection_config import WorkerPodMetadata
from miles.utils.workers.env_vars import WORKER_METADATA_ENV_VAR
from miles.utils.workers.serving import serve_inner
from miles.utils.workers.serving import utils as serving_utils
from miles.utils.workers.serving.serve_inner import _rpc_port_of, parse_own_args
from miles.utils.workers.worker_spec import PortInfo, StaticMeta

CONFIG_PAYLOAD = '{"worker_type": "demo"}'


def _smoke_spec() -> SmokeServeSpec:
    return SmokeServeSpec.create(SmokeWorkerConfig(rpc_port=8000, worker_argv=[]))


def _pod_metadata(port_infos: list[PortInfo]) -> str:
    return WorkerPodMetadata(
        workers_per_pod=1,
        pods_per_cell=1,
        gpu_slots_per_worker=0,
        dynamic_pool=False,
        worker_class="test.worker",
        port_infos=port_infos,
        static_meta=StaticMeta(),
    ).model_dump_json()


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
    def test_the_serialized_pool_config_is_read(self) -> None:
        """The config payload is the whole of what the pod needs to rebuild the one spec it is a worker of."""
        assert parse_own_args(["--config", CONFIG_PAYLOAD]).config == CONFIG_PAYLOAD

    def test_an_omitted_config_is_a_usage_error(self) -> None:
        """A process that is not told what it serves would have nothing to rebuild its spec from."""
        with pytest.raises(SystemExit) as exc_info:
            parse_own_args([])

        assert exc_info.value.code == 2

    def test_unknown_inner_option_is_a_usage_error(self) -> None:
        """The inner entrypoint rejects an option it does not define instead of ignoring it."""
        with pytest.raises(SystemExit) as exc_info:
            parse_own_args(["--config", CONFIG_PAYLOAD, "--unknown-option", "1"])

        assert exc_info.value.code == 2


def _served(monkeypatch: pytest.MonkeyPatch, *, has_dualstack_ipv6: bool) -> dict[str, Any]:
    served: dict[str, Any] = {}
    own_argv = serve_config_argv(spec_class=SmokeServeSpec, config=SmokeWorkerConfig(rpc_port=8000, worker_argv=[]))
    monkeypatch.setitem(SERVE_SPEC_CLASSES, SmokeServeSpec.worker_type, SmokeServeSpec)
    monkeypatch.setattr(serve_inner.sys, "argv", ["serve_inner", *own_argv])
    monkeypatch.setattr(serving_utils.socket, "has_dualstack_ipv6", lambda: has_dualstack_ipv6)
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

    monkeypatch.setattr(serving_utils.socket, "create_server", create_server)
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

        assert served["host"] == serving_utils.IPV6_WILDCARD_HOST
        assert served["socket_kwargs"] == {"family": socket.AF_INET6, "dualstack_ipv6": True}

    def test_binds_the_ipv4_wildcard_where_ipv6_is_unavailable(self, monkeypatch):
        """Asking for the dual-stack wildcard where there is no ipv6 stack leaves the worker unserved."""
        served = _served(monkeypatch, has_dualstack_ipv6=False)

        assert served["host"] == serving_utils.IPV4_WILDCARD_HOST
        assert served["socket_kwargs"] == {"family": socket.AF_INET}

    def test_the_dual_stack_wildcard_is_the_unspecified_ipv6_address(self):
        """Only the unspecified address accepts the ipv4-mapped connections an ipv4 client makes."""
        assert (serving_utils.IPV6_WILDCARD_HOST, serving_utils.IPV4_WILDCARD_HOST) == ("::", "0.0.0.0")

    def test_serves_the_rpc_port_the_spec_declares_whichever_wildcard_it_binds(self, monkeypatch):
        """The address a client dials is the published pod ip and this port, so the port may not move."""
        dual_stack = _served(monkeypatch, has_dualstack_ipv6=True)
        ipv4 = _served(monkeypatch, has_dualstack_ipv6=False)

        assert dual_stack["port"] == 8123
        assert dual_stack["sockets"][0].address == (serving_utils.IPV6_WILDCARD_HOST, 8123)
        assert dual_stack["socket_closed"] is True
        assert ipv4["port"] == 8123
        assert ipv4["sockets"][0].address == (serving_utils.IPV4_WILDCARD_HOST, 8123)
        assert ipv4["socket_closed"] is True


class TestRpcPortOf:
    @pytest.mark.parametrize(
        "port_infos, expected_count",
        [
            ([PortInfo(name="metrics", static_port=9000)], 0),
            ([PortInfo(name="rpc", static_port=8000), PortInfo(name="rpc", static_port=8001)], 2),
        ],
    )
    def test_a_pod_without_exactly_one_rpc_port_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, port_infos: list[PortInfo], expected_count: int
    ) -> None:
        """A served pod whose metadata has a missing or ambiguous rpc port cannot choose a listening port."""
        monkeypatch.setenv(WORKER_METADATA_ENV_VAR, _pod_metadata(port_infos))

        with pytest.raises(AssertionError, match=rf"declares {expected_count} rpc ports"):
            _rpc_port_of(_smoke_spec())

    def test_the_port_comes_from_the_pod_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The address book publishes the metadata's port, so the process must bind that one and not its own."""
        monkeypatch.setenv(WORKER_METADATA_ENV_VAR, _pod_metadata([PortInfo(name="rpc", static_port=8123)]))

        assert _rpc_port_of(_smoke_spec()).static_port == 8123
