"""CPU tests for SGLang control-plane / router bearer auth.

These exercise the real :class:`SGLangEngine` HTTP methods against a tiny FastAPI
server that records the ``Authorization`` header (and JSON body) of every request
and can be told to reply with arbitrary status codes. No CUDA / sglang server /
model is needed: the engine talks purely over ``requests``.

Pins the auth contract:
  1. ``_wait_server_healthy`` must not emit a literal ``"Bearer None"`` header when
     no api key is configured (the external-engine path passes ``api_key=None``).
  2. With an api key set, every control-plane call carries the bearer header, router
     (de)registration carries the router's key, and registration forwards the
     worker's own api key so the router can proxy to auth-enabled workers.
  3. With no keys, requests carry no ``Authorization`` header at all.
"""

from types import SimpleNamespace

import pytest
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from miles.backends.sglang_utils.sglang_engine import SGLangEngine, _bearer_auth_headers, _wait_server_healthy
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

_ENGINE_MODULE = "miles.backends.sglang_utils.sglang_engine"


class _AuthRecordingServer:
    """Records the ``Authorization`` header and JSON body of every request and
    replies with a configurable status code / content per path (default 200)."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        # path -> Authorization header value (or None if header absent)
        self.auth_by_path: dict[str, str | None] = {}
        # path -> parsed JSON body (or None if there was none)
        self.json_by_path: dict[str, dict | None] = {}
        self.status_by_path: dict[str, int] = {}
        self.content_by_path: dict[str, dict] = {}
        self.app = FastAPI()
        self._server: UvicornThreadServer | None = None
        self._setup_routes()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def set_status(self, path: str, status_code: int) -> None:
        self.status_by_path[path] = status_code

    def set_content(self, path: str, content: dict) -> None:
        self.content_by_path[path] = content

    def _setup_routes(self):
        async def handler(request: Request, full_path: str):
            path = f"/{full_path}"
            self.auth_by_path[path] = request.headers.get("authorization")
            try:
                self.json_by_path[path] = await request.json()
            except Exception:
                self.json_by_path[path] = None
            return JSONResponse(
                content=self.content_by_path.get(
                    path, {"weight_version": "v1", "remote_instance_transfer_engine_info": {}}
                ),
                status_code=self.status_by_path.get(path, 200),
            )

        self.app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"])(handler)

    def start(self):
        self._server = UvicornThreadServer(self.app, host=self.host, port=self.port)
        self._server.start()

    def stop(self):
        if self._server is not None:
            self._server.stop()


@pytest.fixture
def auth_server():
    server = _AuthRecordingServer(host="127.0.0.1", port=find_available_port(31000))
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _make_engine(server: _AuthRecordingServer, api_key: str | None) -> SGLangEngine:
    """Build an ``SGLangEngine`` without running the GPU-bound ``__init__``/``init``.

    Only the attributes the control-plane methods read are populated.
    """
    engine = object.__new__(SGLangEngine)
    engine.server_host = server.host
    engine.server_port = server.port
    engine.node_rank = 0
    engine.server_api_key = api_key
    return engine


def _make_registering_engine(router: _AuthRecordingServer, router_api_key: str | None) -> SGLangEngine:
    """Engine wired to register against / deregister from ``router``."""
    engine = object.__new__(SGLangEngine)
    engine.args = SimpleNamespace(
        use_miles_router=False,
        router_api_key=router_api_key,
        rollout_external=False,
    )
    engine.node_rank = 0
    engine.worker_type = "regular"
    engine.server_host = "127.0.0.1"
    engine.server_port = 12345
    engine.router_ip = router.host
    engine.router_port = router.port
    return engine


def _patch_server_launch(monkeypatch, router_version: str):
    """Make ``_init_normal`` runnable on CPU: skip the real server launch and pin
    the sglang_router version that selects the registration code path."""
    monkeypatch.setattr(f"{_ENGINE_MODULE}.sglang_router", SimpleNamespace(__version__=router_version))
    monkeypatch.setattr(f"{_ENGINE_MODULE}.ServerArgs", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(f"{_ENGINE_MODULE}.launch_server_process", lambda server_args: SimpleNamespace(pid=0))


# ---------------------------------------------------------------------------
# _bearer_auth_headers: the primitive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("falsy", [None, ""])
def test_bearer_auth_headers_none_for_falsy(falsy):
    # The Bearer-None bug: falsy keys must yield no header, not "Bearer None".
    assert _bearer_auth_headers(falsy) is None


# ---------------------------------------------------------------------------
# _wait_server_healthy: Bearer-None regression
# ---------------------------------------------------------------------------


def test_wait_server_healthy_no_bearer_none_header(auth_server):
    # External-engine path calls _wait_server_healthy(api_key=None); the health
    # probe must NOT send a literal "Bearer None".
    _wait_server_healthy(base_url=auth_server.url, api_key=None, is_process_alive=lambda: True)
    assert auth_server.auth_by_path.get("/health_generate") is None
    assert auth_server.auth_by_path.get("/flush_cache") is None


def test_wait_server_healthy_sends_bearer_when_key_set(auth_server):
    _wait_server_healthy(base_url=auth_server.url, api_key="secret", is_process_alive=lambda: True)
    assert auth_server.auth_by_path.get("/health_generate") == "Bearer secret"
    assert auth_server.auth_by_path.get("/flush_cache") == "Bearer secret"


# ---------------------------------------------------------------------------
# Control-plane methods carry auth when a key is set
# ---------------------------------------------------------------------------


def test_control_plane_calls_send_bearer_when_key_set(auth_server):
    engine = _make_engine(auth_server, api_key="secret")

    engine.get_server_info()
    engine.health_generate()
    engine.get_weight_version()
    engine.get_remote_instance_transfer_engine_info(rank=0)
    engine.get_parallelism_info(rank=0)
    engine.flush_cache()
    engine.continue_generation()
    engine.pause_generation(mode="retract")
    engine.stop_profile()
    # _make_request path (weight sync et al.)
    engine.unload_lora_adapter("adapter")

    expected = "Bearer secret"
    assert auth_server.auth_by_path.get("/server_info") == expected
    assert auth_server.auth_by_path.get("/health_generate") == expected
    assert auth_server.auth_by_path.get("/model_info") == expected
    assert auth_server.auth_by_path.get("/get_remote_instance_transfer_engine_info") == expected
    assert auth_server.auth_by_path.get("/parallelism_config") == expected
    assert auth_server.auth_by_path.get("/flush_cache") == expected
    assert auth_server.auth_by_path.get("/continue_generation") == expected
    assert auth_server.auth_by_path.get("/pause_generation") == expected
    assert auth_server.auth_by_path.get("/stop_profile") == expected
    assert auth_server.auth_by_path.get("/unload_lora_adapter") == expected


# ---------------------------------------------------------------------------
# Backward compatibility: no key => no Authorization header at all
# ---------------------------------------------------------------------------


def test_control_plane_calls_omit_auth_when_no_key(auth_server):
    engine = _make_engine(auth_server, api_key=None)

    engine.get_server_info()
    engine.health_generate()
    engine.flush_cache()
    engine.continue_generation()
    engine.stop_profile()
    engine.unload_lora_adapter("adapter")

    for path in (
        "/server_info",
        "/health_generate",
        "/flush_cache",
        "/continue_generation",
        "/stop_profile",
        "/unload_lora_adapter",
    ):
        assert auth_server.auth_by_path.get(path) is None, f"{path} should carry no auth header"


# ---------------------------------------------------------------------------
# Router registration / deregistration
# ---------------------------------------------------------------------------


def test_router_registration_sends_bearer_and_forwards_worker_key(auth_server, monkeypatch):
    _patch_server_launch(monkeypatch, router_version="0.3.0")
    engine = _make_registering_engine(auth_server, router_api_key="router-key")

    engine._init_normal({"api_key": "worker-key"})

    assert auth_server.auth_by_path.get("/workers") == "Bearer router-key"
    assert auth_server.json_by_path.get("/workers")["api_key"] == "worker-key"


def test_router_registration_omits_auth_and_worker_key_when_no_keys(auth_server, monkeypatch):
    _patch_server_launch(monkeypatch, router_version="0.3.0")
    engine = _make_registering_engine(auth_server, router_api_key=None)

    engine._init_normal({})

    assert auth_server.auth_by_path.get("/workers") is None
    assert "api_key" not in auth_server.json_by_path.get("/workers")


def test_router_registration_legacy_add_worker_sends_bearer(auth_server, monkeypatch):
    _patch_server_launch(monkeypatch, router_version="0.2.1")
    engine = _make_registering_engine(auth_server, router_api_key="router-key")

    engine._init_normal({})

    assert auth_server.auth_by_path.get("/add_worker") == "Bearer router-key"


def test_shutdown_deregistration_sends_bearer(auth_server, monkeypatch):
    monkeypatch.setattr(f"{_ENGINE_MODULE}.sglang_router", SimpleNamespace(__version__="0.3.0"))
    monkeypatch.setattr(f"{_ENGINE_MODULE}.kill_process_tree", lambda pid: None)
    engine = _make_registering_engine(auth_server, router_api_key="router-key")
    engine.process = SimpleNamespace(pid=0)
    auth_server.set_content("/workers", {"workers": [{"url": "http://127.0.0.1:12345", "id": "w1"}]})

    engine.shutdown()

    assert auth_server.auth_by_path.get("/workers") == "Bearer router-key"
    assert auth_server.auth_by_path.get("/workers/w1") == "Bearer router-key"


# ---------------------------------------------------------------------------
# flush_cache fails fast on 401/403 instead of retrying for 60s
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [401, 403])
def test_flush_cache_fails_fast_on_auth_error(auth_server, status_code):
    auth_server.set_status("/flush_cache", status_code)
    engine = _make_engine(auth_server, api_key="wrong-key")

    with pytest.raises(requests.exceptions.HTTPError):
        engine.flush_cache()
