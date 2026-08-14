from __future__ import annotations

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import pytest

from miles.ray.multi_lora import controller as controller_module
from miles.ray.multi_lora.controller import MultiLoRAController

pytestmark = pytest.mark.asyncio


def _make_args(**overrides) -> SimpleNamespace:
    args = SimpleNamespace(
        sglang_router_ip=None,
        sglang_router_port=None,
        multi_lora_backend_path=None,
        multi_lora_http_server_path=None,
        multi_lora_api_port=0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def events(monkeypatch) -> list[str]:
    recorded: list[str] = []

    async def fake_resolve_router_addrs(args, *, router_providers):
        recorded.append("resolve_router_addrs")
        args.sglang_router_ip = "10.0.0.1"
        args.sglang_router_port = 4321

    class _Backend:
        def __init__(self, args, router_url):
            recorded.append(f"backend({router_url})")

        async def init(self):
            recorded.append("backend.init")

    class _Server:
        actual_api_port = 8123

        def __init__(self, backend, host, api_port):
            recorded.append(f"server({host}:{api_port})")

        async def start(self):
            recorded.append("server.start")

    monkeypatch.setattr(controller_module, "resolve_router_addrs", fake_resolve_router_addrs)
    monkeypatch.setattr(controller_module, "MultiLoRABackend", _Backend)
    monkeypatch.setattr(controller_module, "MultiLoRAHTTPServer", _Server)
    return recorded


class TestInit:
    async def test_the_constructor_starts_nothing(self, events):
        """The platform constructs the worker long before the router exists, so init() owns every side effect."""
        MultiLoRAController(args=_make_args(), router_providers=[object()])

        assert events == []

    async def test_it_resolves_the_router_on_its_own_args_before_building_the_backend(self, events):
        """The worker's args snapshot predates the driver's resolution, so a captured url would be empty."""
        args = _make_args()
        controller = MultiLoRAController(args=args, router_providers=[object()])

        await controller.init()

        assert events == [
            "resolve_router_addrs",
            "backend(http://10.0.0.1:4321)",
            "server(0.0.0.0:0)",
            "backend.init",
            "server.start",
        ]
        assert args.sglang_router_ip == "10.0.0.1"

    async def test_it_answers_the_port_the_server_actually_bound(self, events):
        """--multi-lora-api-port 0 asks the os for a port, so the caller can only learn it from the server."""
        controller = MultiLoRAController(args=_make_args(), router_providers=[object()])

        assert await controller.init() == 8123
