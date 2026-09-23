from __future__ import annotations

import socket
from collections.abc import Callable, Iterator

import pytest
import requests
from fastapi import FastAPI

from miles.utils.misc import get_current_node_ip, get_free_port
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _is_reachable(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2.0)
        return sock.connect_ex((host, port)) == 0


@pytest.fixture
def start_server() -> Iterator[Callable[..., UvicornThreadServer]]:
    servers: list[UvicornThreadServer] = []

    def _start(**kwargs) -> UvicornThreadServer:
        server = UvicornThreadServer(_make_app(), port=get_free_port(start_port=21000), **kwargs)
        servers.append(server)
        server.start()
        return server

    yield _start

    for server in servers:
        server.stop()


class TestBindHost:
    def test_the_listener_defaults_to_the_advertised_host(self, start_server) -> None:
        """Without ``bind_host`` the server listens only on the host it advertises."""
        node_ip = get_current_node_ip()
        assert node_ip != "127.0.0.1"

        server = start_server(host="127.0.0.1")

        assert _is_reachable(host="127.0.0.1", port=server.port)
        assert not _is_reachable(host=node_ip, port=server.port)

    def test_an_explicit_bind_host_widens_the_listener_beyond_the_advertised_host(self, start_server) -> None:
        """``bind_host`` must reach uvicorn, else a loopback-advertising server stays unreachable elsewhere."""
        node_ip = get_current_node_ip()

        server = start_server(host="127.0.0.1", bind_host="0.0.0.0")

        assert requests.get(f"http://127.0.0.1:{server.port}/health", timeout=5).status_code == 200
        assert requests.get(f"http://{node_ip}:{server.port}/health", timeout=5).status_code == 200

    def test_the_url_advertises_the_host_and_never_the_wildcard_bind_address(self, start_server) -> None:
        """Peers dial ``url``, so it must carry the routable host even when the socket binds the wildcard address."""
        node_ip = get_current_node_ip()

        server = start_server(host=node_ip, bind_host="0.0.0.0")

        assert server.host == node_ip
        assert server.url == f"http://{node_ip}:{server.port}"
        assert requests.get(f"{server.url}/health", timeout=5).status_code == 200
