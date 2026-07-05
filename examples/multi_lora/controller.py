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

import asyncio
import json
from typing import Optional

import aiohttp
from aiohttp import web

import uuid

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
    """Adapter registry + in-flight rid tracking, no I/O. Testable directly."""

    def __init__(self) -> None:
        self._active: dict[str, int] = {}  # adapter name -> slot
        self._in_flight: dict[str, str] = {}  # rid -> adapter name

    def register(self, name: str, slot: int) -> None:
        self._active[name] = slot

    def retire(self, name: str) -> None:
        self._active.pop(name, None)

    def active(self) -> dict[str, int]:
        return dict(self._active)

    def on_forward(self, rid: str) -> bool:
        """Decide whether to forward a request. Returns True iff the adapter is
        active (and records it as in-flight). False => block, return dummy."""
        name = parse_adapter(rid)
        if name not in self._active:
            return False
        self._in_flight[rid] = name
        return True

    def on_response(self, rid: str) -> bool:
        """Decide whether to dummy a completed response. Returns True iff the
        adapter this request was sent for is no longer active."""
        name = self._in_flight.pop(rid, None)
        if name is None:
            return True  # unknown rid (e.g. blocked, or already retired) -> dummy
        return name not in self._active


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
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._client: Optional[aiohttp.ClientSession] = None

    @property
    def actual_port(self) -> int:
        return self._site._server.sockets[0].getsockname()[1] if self._site else self.port

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/register_adapter", self._register)
        app.router.add_post("/retire_adapter", self._retire)
        app.router.add_get("/active_adapters", self._active)
        app.router.add_resource("/{tail:.*}").add_route("*", self._proxy)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        self._client = aiohttp.ClientSession()

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _register(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.logic.register(body["name"], body["slot"])
        return web.json_response({"ok": True, "active": self.logic.active()})

    async def _retire(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.logic.retire(body["name"])
        return web.json_response({"ok": True, "active": self.logic.active()})

    async def _active(self, request: web.Request) -> web.Response:
        return web.json_response(self.logic.active())

    async def _proxy(self, request: web.Request) -> web.Response:
        body = await request.read()
        rid = self._extract_rid(body)

        if rid is not None and not self.logic.on_forward(rid):
            return web.json_response(dummy_response_body(rid), status=200)

        url = f"{self.upstream_url}/{request.match_info['tail']}"
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("content-length", "transfer-encoding", "host")}
        assert self._client is not None
        async with self._client.request(request.method, url, data=body, headers=headers) as upstream:
            content = await upstream.read()
            if rid is not None and self.logic.on_response(rid):
                return web.json_response(dummy_response_body(rid), status=200)
            out_headers = {k: v for k, v in upstream.headers.items()
                           if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")}
            return web.Response(body=content, status=upstream.status, headers=out_headers)

    @staticmethod
    def _extract_rid(body: bytes) -> Optional[str]:
        if not body:
            return None
        try:
            obj = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return None
        if isinstance(obj, dict):
            return obj.get("rid")
        return None
