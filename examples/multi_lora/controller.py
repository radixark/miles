"""Multi-LoRA controller for the fully-async design.

Self-contained (example-level, no library changes). The controller is a Ray
async actor that also runs an HTTP server (Ray out-of-band communication), so:
  - trainer / datasource reach it via Ray (efficient, in-process state),
  - rollout requests reach it via HTTP (the proxy), with the active-adapter set
    and in-flight rid table in-process.

Correctness for adapter replacement: each rollout request carries
``rid = f"{adapter_name}_{uuid4().hex}"``. The controller blocks forwards for
adapters no longer active and dummies responses whose adapter was retired while
the request was in flight (covers the mid-send transit stragglers that abort
can't bound). No SGLang changes, no versions, no drain.

This module factors the testable logic into a plain class; the Ray + HTTP
wrapper is added on top.
"""

import json
import uuid
from typing import Any, Optional

import aiohttp
from aiohttp import web

__all__ = [
    "MultiLoRAControllerLogic",
    "MultiLoRAHTTPServer",
    "make_rid",
    "parse_adapter",
    "dummy_response_body",
]


def make_rid(adapter_name: str) -> str:
    """Build a request id encoding the adapter name (names may contain '_')."""
    return f"{adapter_name}_{uuid.uuid4().hex}"


def parse_adapter(rid: str) -> str:
    """Recover the adapter name from a rid built by ``make_rid``.

    ``uuid4().hex`` has no underscores, so the adapter name is everything before
    the final ``_``.
    """
    return rid.rsplit("_", 1)[0]


class MultiLoRAControllerLogic:
    """Adapter registry + in-flight rid tracking, no I/O. Testable directly.

    The controller is the source of truth for the active-adapter set. The data
    source reads it to decide what to generate for, and calls ``deregister`` when an
    adapter reaches its num_row; the trainer reads it to decide what to train.
    Both reconcile to it. No step tracking, no drain, no rollout id.
    """

    def __init__(self) -> None:
        self.active_slots: dict[str, int] = {}  # adapter name -> slot
        self.configs: dict[str, Any] = {}  # adapter name -> opaque AdapterConfig
        self.in_flight: dict[str, str] = {}  # rid -> adapter name

    def register(self, name: str, slot: int, config: Any = None) -> None:
        self.active_slots[name] = slot
        self.configs[name] = config

    def deregister(self, name: str) -> None:
        self.active_slots.pop(name, None)
        self.configs.pop(name, None)

    def active(self) -> dict[str, int]:
        return dict(self.active_slots)

    def active_adapters(self) -> dict[str, dict]:
        """name -> {'slot', 'config'} for readers (data source, trainer)."""
        return {name: {"slot": slot, "config": self.configs[name]} for name, slot in self.active_slots.items()}

    def on_forward(self, rid: str) -> bool:
        """Decide whether to forward a request. Returns True iff the adapter is
        active (and records it as in-flight). False => block, return dummy."""
        name = parse_adapter(rid)
        if name not in self.active_slots:
            return False
        self.in_flight[rid] = name
        return True

    def on_response(self, rid: str) -> bool:
        """Decide whether to dummy a completed response. Returns True iff the
        adapter this request was sent for is no longer active."""
        name = self.in_flight.pop(rid, None)
        if name is None:
            return True  # unknown rid (e.g. blocked, or already retired) -> dummy
        return name not in self.active_slots


def dummy_response_body(rid: str) -> dict:
    """A normal-shaped SGLang generate response that yields an ABORTED sample.

    The producer reads ``text`` and ``meta_info`` (incl. ``finish_reason.type``
    and optional ``output_token_logprobs``); ``"abort"`` makes
    ``Sample.update_from_meta_info`` set ``Sample.Status.ABORTED``, which the
    existing fully-async recycle path already handles. No extra marker fields,
    so it's indistinguishable in shape from a real SGLang abort response.
    """
    return {
        "text": "",
        "meta_info": {"finish_reason": {"type": "abort"}},
        "rid": rid,
    }


class MultiLoRAHTTPServer:
    """aiohttp server wrapping MultiLoRAControllerLogic. Plain asyncio (no Ray),
    so it can be smoke-tested with a mock upstream. The Ray actor wraps this."""

    def __init__(
        self,
        logic: MultiLoRAControllerLogic,
        upstream_url: str,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.logic = logic
        self.upstream_url = upstream_url.rstrip("/")
        self.host = host
        self.port = port
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.client: Optional[aiohttp.ClientSession] = None

    @property
    def actual_port(self) -> int:
        return self.site._server.sockets[0].getsockname()[1] if self.site else self.port

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/register_adapter", self.register_handler)
        app.router.add_post("/deregister_adapter", self.deregister_handler)
        app.router.add_get("/active_adapters", self.active_handler)
        app.router.add_resource("/{tail:.*}").add_route("*", self.proxy_handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        self.client = aiohttp.ClientSession()

    async def stop(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None

    async def register_handler(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.logic.register(body["name"], body["slot"], body.get("config"))
        return web.json_response({"ok": True, "active": self.logic.active()})

    async def deregister_handler(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.logic.deregister(body["name"])
        return web.json_response({"ok": True, "active": self.logic.active()})

    async def active_handler(self, request: web.Request) -> web.Response:
        return web.json_response(self.logic.active())

    async def proxy_handler(self, request: web.Request) -> web.Response:
        body = await request.read()
        rid = extract_rid(body)

        if rid is not None and not self.logic.on_forward(rid):
            return web.json_response(dummy_response_body(rid), status=200)

        url = f"{self.upstream_url}/{request.match_info['tail']}"
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("content-length", "transfer-encoding", "host")}
        assert self.client is not None
        async with self.client.request(request.method, url, data=body, headers=headers) as upstream:
            content = await upstream.read()
            if rid is not None and self.logic.on_response(rid):
                return web.json_response(dummy_response_body(rid), status=200)
            out_headers = {k: v for k, v in upstream.headers.items()
                           if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")}
            return web.Response(body=content, status=upstream.status, headers=out_headers)


def extract_rid(body: bytes) -> Optional[str]:
    if not body:
        return None
    try:
        obj = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if isinstance(obj, dict):
        return obj.get("rid")
    return None
