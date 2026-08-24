"""The frontend HTTP guard: SDK-key auth on /api/v1, the loopback-only
operator plane, readiness vs liveness probes, and the CLI flag contract —
all over ASGI so peer addresses can be faked."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu")

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from tests.fast.ray.tinker_frontend.fake_stack import FakeDriver, make_backend

from miles.ray.multi_lora.http_server import AdapterRunControlServer
from miles.ray.tinker_frontend.http_server import TinkerFrontendHTTPServer
from miles.utils.tinker import validate_tinker_args

API_KEY = "tml-test-key"


def test_frontend_extends_the_canonical_operation_control_server():
    assert issubclass(TinkerFrontendHTTPServer, AdapterRunControlServer)


def make_app(api_key=API_KEY, ready=True):
    backend = make_backend(tinker_api_key=api_key)
    if ready:
        FakeDriver(backend)
    server = TinkerFrontendHTTPServer(backend, host="127.0.0.1", api_port=0)
    app = server.create_app()
    server.add_routes(app)
    return app


def get(app, path, peer="127.0.0.1", **headers):
    async def go():
        transport = httpx.ASGITransport(app=app, client=(peer, 40000))
        async with httpx.AsyncClient(transport=transport, base_url="http://frontend") as client:
            return await client.get(path, headers=headers)

    return asyncio.run(go())


class TestGuard:
    def test_sdk_routes_require_the_key_from_any_peer(self):
        app = make_app()
        assert get(app, "/api/v1/get_server_capabilities").status_code == 401
        for peer in ("127.0.0.1", "203.0.113.9"):
            response = get(app, "/api/v1/get_server_capabilities", peer=peer, **{"x-api-key": API_KEY})
            assert response.status_code == 200, peer

    def test_operator_plane_is_loopback_only_even_with_the_sdk_key(self):
        app = make_app()
        for path in ("/adapter_runs", "/info"):
            assert get(app, path, peer="203.0.113.9", **{"x-api-key": API_KEY}).status_code == 403, path
            assert get(app, path, peer="127.0.0.1", **{"x-api-key": API_KEY}).status_code == 200, path
        assert get(app, "/adapter_runs").status_code == 401

    def test_health_probes_are_exempt_from_auth(self):
        app = make_app()
        assert get(app, "/health").status_code == 200
        assert get(app, "/api/v1/healthz").status_code == 200

    def test_healthz_is_503_until_the_trainer_is_ready(self):
        app = make_app(ready=False)
        assert get(app, "/health").status_code == 200
        assert get(app, "/api/v1/healthz").status_code == 503


class TestLaunchFlags:
    def args(self, **overrides):
        values = dict(tinker_backend=False, tinker_frontend=False, tinker_api_key=None)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_frontend_alone_fails_loud_instead_of_a_silent_noop(self):
        with pytest.raises(AssertionError, match="requires --tinker-backend"):
            validate_tinker_args(self.args(tinker_frontend=True))

    def test_api_key_requires_the_frontend(self):
        with pytest.raises(AssertionError, match="requires --tinker-frontend"):
            validate_tinker_args(self.args(tinker_api_key="tml-x"))

    def test_plain_run_still_validates(self):
        validate_tinker_args(self.args())
