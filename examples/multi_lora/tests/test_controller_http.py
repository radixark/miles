"""HTTP tests for the MultiLoRAHTTPServer control plane with a mock router
(no Ray, no SGLang)."""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web

from miles.utils.adapter_config import AdapterRunConfig
from miles.utils.multi_lora import RID_SEPARATOR, MultiLoRABackend, MultiLoRAHTTPServer


def minimal_config(name: str) -> dict:
    return {"data": f"/data/{name}.parquet", "save": f"/tmp/adapters/{name}"}


class ControllerHarness:
    """Running control plane (backend + API listener) against a mock router
    that serves /list_workers and records /abort_request posts."""

    def __init__(self, session: aiohttp.ClientSession, backend: MultiLoRABackend, srv: MultiLoRAHTTPServer):
        self.session = session
        self.backend = backend
        self.srv = srv
        self.aborts: list[dict] = []

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self.srv.actual_api_port}"

    async def api_post(self, path: str, payload: dict) -> tuple[int, dict]:
        async with self.session.post(f"{self.api_base}{path}", json=payload) as resp:
            return resp.status, await resp.json()

    async def api_get(self, path: str) -> tuple[int, dict, dict]:
        async with self.session.get(f"{self.api_base}{path}") as resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, await resp.json(), headers

    async def api_delete(self, path: str) -> tuple[int, dict]:
        async with self.session.delete(f"{self.api_base}{path}") as resp:
            return resp.status, await resp.json()

    async def register(self, name: str) -> tuple[int, dict]:
        status, body = await self.api_post("/adapter_runs", {"name": name, "config": minimal_config(name)})
        # Registered adapters start pending; a weight push promotes them.
        self.backend.registry.record_weight_update([name])
        return status, body

    async def deregister(self, name: str) -> tuple[int, dict]:
        return await self.api_delete(f"/adapter_runs/{name}")

    async def active(self) -> dict:
        _, body, _ = await self.api_get("/adapter_runs")
        return {
            s["name"]: {"slot": s["slot"], "version": s["version"], "step": s["step"]}
            for s in body["adapters"]
            if s["state"] == "ACTIVE"
        }


@asynccontextmanager
async def running_controller(server_cls=MultiLoRAHTTPServer):
    router_url = ""
    harness: ControllerHarness | None = None

    async def router_handler(request):
        if request.path == "/list_workers":
            return web.json_response({"urls": [router_url]})
        if request.path == "/abort_request":
            harness.aborts.append(json.loads(await request.read()))
            return web.json_response({})
        return web.json_response({}, status=404)

    app = web.Application()
    app.router.add_resource("/{tail:.*}").add_route("*", router_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    router_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"

    backend = MultiLoRABackend(SimpleNamespace(multi_lora_n_adapters=4, save=None), router_url)
    srv = server_cls(backend)
    await backend.init()
    await srv.start()
    try:
        async with aiohttp.ClientSession() as session:
            harness = ControllerHarness(session, backend, srv)
            yield harness
    finally:
        await srv.stop()
        await backend.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_register_and_active_view():
    async with running_controller() as ctl:
        status, body = await ctl.register("A")
        assert status == 200
        assert body["slot"] == 0
        assert await ctl.active() == {"A": {"slot": 0, "version": 1, "step": 0}}


@pytest.mark.asyncio
async def test_deregister_marks_and_retire_adapters_aborts():
    """Deregistration only marks; the driver-synced apply performs the
    demotion and fans out one prefix abort per worker."""
    async with running_controller() as ctl:
        await ctl.register("A")
        status, _ = await ctl.deregister("A")
        assert status == 200
        assert ctl.aborts == []  # still serving until the sync point
        assert "A" in ctl.backend.registry.active_adapters()

        applied = await ctl.backend.retire_adapters()
        assert applied == ["A"]
        assert ctl.aborts == [{"rid": f"A{RID_SEPARATOR}", "prefix": True}]
        assert ctl.backend.registry.active_adapters() == {}


@pytest.mark.asyncio
async def test_register_json_config_validates_to_adapter_config():
    """FastAPI validates the JSON body straight into AdapterRunConfig (422 on bad
    payloads)."""
    async with running_controller() as ctl:
        config = {
            "rank": 8,
            "data": "/data/train.parquet",
            "save": "/tmp/adapters/A",
            "rm_type": "math",
        }
        status, _ = await ctl.api_post("/adapter_runs", {"name": "A", "config": config})
        assert status == 200
        record = ctl.backend.registry.find("A")
        assert isinstance(record.config, AdapterRunConfig)
        assert record.config.data == "/data/train.parquet"
        assert Path(record.config.save) == Path("/tmp/adapters/A")
        assert record.config.input_key == "text"  # dataclass default

        status, _ = await ctl.api_post("/adapter_runs", {"name": "B", "config": {"rank": 8}})
        assert status == 422  # data is required

        status, _ = await ctl.api_post("/adapter_runs", {"name": "C"})
        assert status == 400  # exactly one of config/yaml_path


@pytest.mark.asyncio
async def test_state_endpoint_reports_lifecycle_and_completed():
    """States walk PENDING -> ACTIVE -> RETIRING -> CLEANUP -> COMPLETED;
    unknown names report null; COMPLETED is retained after free_slot."""
    async with running_controller() as ctl:
        await ctl.api_post("/adapter_runs", {"name": "A", "config": minimal_config("A")})

        async def state_of(name):
            _, body, _ = await ctl.api_get(f"/adapter_runs/state?names={name}")
            return body["states"][name]

        assert await state_of("A") == "PENDING"
        ctl.backend.registry.record_weight_update(["A"])
        assert await state_of("A") == "ACTIVE"

        await ctl.deregister("A")
        assert await state_of("A") == "RETIRING"
        await ctl.backend.retire_adapters()
        assert await state_of("A") == "CLEANUP"

        ctl.backend.registry.free_slot("A")
        assert await state_of("A") == "COMPLETED"
        assert await state_of("nope") is None

        # GET by name serves the completed record; DELETE of unknown 404s.
        status, body, _ = await ctl.api_get("/adapter_runs/A")
        assert status == 200 and body["state"] == "COMPLETED"
        status, _ = await ctl.api_delete("/adapter_runs/nope")
        assert status == 404

        # Re-registration reclaims the name; the completed record is dropped.
        status, _ = await ctl.api_post(
            "/adapter_runs", {"name": "A", "config": {"data": "/data/A2.parquet", "save": "/tmp/adapters/A2"}}
        )
        assert status == 200
        assert await state_of("A") == "PENDING"


@pytest.mark.asyncio
async def test_custom_server_subclass_adds_routes():
    class CustomServer(MultiLoRAHTTPServer):
        def create_app(self):
            app = super().create_app()

            @app.middleware("http")
            async def tag_response(request, call_next):
                response = await call_next(request)
                response.headers["X-Custom-Server"] = "1"
                return response

            return app

        def add_routes(self, app):
            super().add_routes(app)
            app.get("/custom_status")(self.custom_status)

        async def custom_status(self):
            return {"custom": True, "active": sorted(self.backend.registry.active_adapters())}

    async with running_controller(server_cls=CustomServer) as ctl:
        _, body, headers = await ctl.api_get("/custom_status")
        assert headers.get("x-custom-server") == "1"
        assert body == {"custom": True, "active": []}
