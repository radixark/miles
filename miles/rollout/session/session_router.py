"""Thin client-facing router process for the multi-process session server.

The router is the SOLE HTTP listener. It owns no session state and no
tokenizer: it holds one client-side :class:`IpcChannel` per worker, maps each
request's ``session_id`` to its owning worker with
:func:`worker_index_for_session`, ships an op envelope over IPC, and renders the
worker's :class:`CoreResponse` back into a ``starlette`` response WITHOUT
re-parsing the body (chat AND GET-records bodies are relayed as opaque bytes).

Design (m3-design-contract §"Router"):

* Route order: the chat route is registered BEFORE the catch-all proxy route so
  chat keeps record-keeping (the catch-all would otherwise swallow it).
* Catch-all keeps NO-404 proxy semantics: it routes by session_id but never
  looks the session up / 404s on an unknown id.
* ``/health`` pings every worker (bounded timeout) and is healthy iff all
  respond — this is also the readiness signal the supervisor waits on.
* Cancellation: a client disconnect must NOT cancel the worker's in-flight chat
  task (today's race-test semantics: the handler continues and may commit). The
  IPC request runs as a router-owned task and the handler ``await``\\s it under
  :func:`asyncio.shield`, so a handler cancel still drains the IPC reply rather
  than abandoning the worker mid-send.
* Per-worker backpressure: a bounded in-flight counter returns 503 when a worker
  is saturated. The slot is freed in the IPC task's done-callback (when the
  worker's reply actually arrives), NOT on handler cancellation — so a
  disconnected request keeps counting against the cap while it still consumes
  worker/httpx resources, instead of leaking the count.

Stdlib + FastAPI/uvicorn only; the parent spawns this as a process target.
"""

from __future__ import annotations

import asyncio
import json
import logging

import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import Response

from miles.rollout.session.routing import new_session_id, worker_index_for_session
from miles.rollout.session.session_core import CoreResponse
from miles.rollout.session.session_ipc import IpcChannelClosed, IpcError, open_unix_channel
from miles.rollout.session.session_worker import (
    OP_CHAT,
    OP_CREATE_ID,
    OP_DELETE,
    OP_GET,
    OP_HEALTH,
    OP_PROXY,
    _set_pdeathsig,
    decode_core_response,
    encode_request,
)

logger = logging.getLogger(__name__)

# Per-worker in-flight cap at the router: a saturated worker rejects with 503
# rather than letting router futures grow unbounded (the worker enforces its own
# admission cap too; this is the front-door guard).
DEFAULT_ROUTER_MAX_INFLIGHT = 512
# Health ping timeout: a worker mid-parse can be briefly unresponsive, so the
# readiness/health probe must not block forever on a busy worker.
DEFAULT_HEALTH_TIMEOUT = 5.0


def _to_starlette_response(core_response: CoreResponse) -> Response:
    return Response(
        content=core_response.body,
        status_code=core_response.status_code,
        headers=core_response.headers,
        media_type=core_response.media_type,
    )


def _overloaded_response() -> Response:
    return Response(
        content=b'{"error":"session router overloaded (per-worker in-flight cap reached)"}',
        status_code=503,
        media_type="application/json",
    )


def _channel_closed_response() -> Response:
    # A worker channel torn down (worker death / fail-fast in progress): the
    # supervisor is killing the group, so surface a 503 rather than a 500.
    return Response(
        content=b'{"error":"session worker unavailable"}',
        status_code=503,
        media_type="application/json",
    )


class SessionRouter:
    """Holds one client channel per worker and dispatches ops by session_id."""

    def __init__(self, args, channels, *, max_inflight: int = DEFAULT_ROUTER_MAX_INFLIGHT):
        self._args = args
        self._channels = list(channels)
        self._n_worker = len(self._channels)
        self._instance_id = getattr(args, "session_server_instance_id", None)
        self._health_timeout = getattr(args, "session_router_health_timeout", DEFAULT_HEALTH_TIMEOUT)
        self._max_inflight = max_inflight
        self._inflight = [0] * self._n_worker
        # Router-owned IPC request tasks: a client disconnect cancels the HTTP
        # handler but NOT these, so the worker still drains its reply. Held here
        # so they are never an unobserved/GC'd task; each removes itself + frees
        # its in-flight slot in its done-callback.
        self._ipc_tasks: set[asyncio.Task] = set()

    def _channel_for(self, session_id: str):
        return self._channels[worker_index_for_session(session_id, self._n_worker)]

    def _spawn_ipc_request(self, worker_idx: int, payload: bytes) -> asyncio.Task[bytes]:
        """Start a worker IPC request as a router-owned task and account for it.

        The in-flight slot is taken here and released in the done-callback (NOT
        when the awaiting HTTP handler is cancelled), so a disconnected client's
        request keeps counting against the per-worker cap until the worker's
        reply actually arrives — disconnect no longer leaks the count nor frees
        worker/httpx resources early.
        """
        self._inflight[worker_idx] += 1
        task = asyncio.create_task(self._channels[worker_idx].request(payload))
        self._ipc_tasks.add(task)

        def _done(t: asyncio.Task, _idx=worker_idx) -> None:
            self._ipc_tasks.discard(t)
            self._inflight[_idx] -= 1
            if not t.cancelled():
                t.exception()  # retrieve to avoid "exception never retrieved" if abandoned

        task.add_done_callback(_done)
        return task

    async def _dispatch(self, worker_idx: int, payload: bytes) -> Response:
        """Send *payload* to one worker and render its reply.

        The IPC request runs as a router-owned task (see ``_spawn_ipc_request``);
        a cancel of THIS coroutine (client disconnect) is shielded so it does
        not cancel the worker's chat task — the worker still drains its reply.

        Backpressure is admission-checked per worker before the task is started,
        so a saturated worker fast-fails 503 here.
        """
        if self._inflight[worker_idx] >= self._max_inflight:
            return _overloaded_response()
        task = self._spawn_ipc_request(worker_idx, payload)
        try:
            reply = await asyncio.shield(task)
        except IpcChannelClosed:
            return _channel_closed_response()
        except IpcError as exc:
            logger.error("IPC error dispatching to worker %d: %s", worker_idx, exc)
            return Response(
                content=f'{{"error":"session worker error: {exc}"}}'.encode(),
                status_code=502,
                media_type="application/json",
            )
        return _to_starlette_response(decode_core_response(reply))

    async def _dispatch_session(self, session_id: str, payload: bytes) -> Response:
        return await self._dispatch(worker_index_for_session(session_id, self._n_worker), payload)

    # ---- routes ---------------------------------------------------------

    async def create_session(self) -> Response:
        sid = new_session_id()
        worker_idx = worker_index_for_session(sid, self._n_worker)
        return await self._dispatch(worker_idx, encode_request(OP_CREATE_ID, session_id=sid))

    async def get_session(self, session_id: str) -> Response:
        return await self._dispatch_session(session_id, encode_request(OP_GET, session_id=session_id))

    async def delete_session(self, session_id: str) -> Response:
        return await self._dispatch_session(session_id, encode_request(OP_DELETE, session_id=session_id))

    async def chat_completions(self, request: Request, session_id: str) -> Response:
        body = await request.body()
        return await self._dispatch_session(
            session_id,
            encode_request(
                OP_CHAT,
                session_id=session_id,
                method=request.method,
                query=request.url.query,
                headers=dict(request.headers),
                body=body,
            ),
        )

    async def session_proxy(self, request: Request, session_id: str, path: str) -> Response:
        # NO-404 proxy semantics: route by session_id but never look it up.
        body = await request.body()
        return await self._dispatch_session(
            session_id,
            encode_request(
                OP_PROXY,
                session_id=session_id,
                method=request.method,
                query=request.url.query,
                headers=dict(request.headers),
                path=path,
                body=body,
            ),
        )

    async def health(self) -> Response:
        """Healthy iff every worker answers a health ping within the timeout.

        Each worker is pinged with a per-worker timeout (a worker mid-parse may
        be briefly slow); any timeout / channel error makes the router unhealthy.
        """
        if not await self.all_workers_healthy():
            return Response(
                content=b'{"status":"unavailable"}',
                status_code=503,
                media_type="application/json",
            )
        body = {"status": "ok"}
        if self._instance_id is not None:
            body["session_server_instance_id"] = self._instance_id
        return _to_starlette_response(
            CoreResponse(
                status_code=200,
                headers={"content-type": "application/json"},
                body=_json_bytes(body),
                media_type="application/json",
            )
        )

    async def all_workers_healthy(self) -> bool:
        ping = encode_request(OP_HEALTH)

        async def _ping(channel) -> bool:
            # Own the request task so a timeout here does not abandon an
            # unobserved IPC task (whose later exception would be GC-logged); on
            # timeout we stop waiting but the task drains and self-cleans below.
            task = asyncio.create_task(channel.request(ping))
            self._ipc_tasks.add(task)

            def _done(t: asyncio.Task) -> None:
                self._ipc_tasks.discard(t)
                if not t.cancelled():
                    t.exception()

            task.add_done_callback(_done)
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=self._health_timeout)
                return True
            except (TimeoutError, asyncio.TimeoutError, IpcError, IpcChannelClosed):
                return False

        results = await asyncio.gather(*(_ping(ch) for ch in self._channels))
        return all(results)


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def build_app(router: SessionRouter) -> FastAPI:
    """Wire the router's methods onto a FastAPI app.

    Route order matters: the chat route MUST precede the catch-all proxy route
    so a chat request keeps record-keeping instead of being swallowed by the
    proxy (which skips the session core's chat path).
    """
    app = FastAPI()

    app.get("/health")(router.health)
    app.post("/sessions")(router.create_session)
    app.get("/sessions/{session_id}")(router.get_session)
    app.delete("/sessions/{session_id}")(router.delete_session)
    app.post("/sessions/{session_id}/v1/chat/completions")(router.chat_completions)
    app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])(
        router.session_proxy
    )
    return app


async def _open_channels(socks):
    """Open one client-side channel per router-end socket.

    No ``request_handler`` (the router only sends ``request()``); ``on_close``
    is left to the channel's own teardown — worker death surfaces as the per-
    request ``IpcChannelClosed`` and the supervisor's process monitor drives
    fail-fast.
    """
    return [await open_unix_channel(sock) for sock in socks]


def run_router(args, socks, host: str, port: int) -> None:
    """``multiprocessing.Process`` target for the router.

    Parameters
    ----------
    args : the Miles args namespace (carries session_server_instance_id, …).
    socks : the N router-end sockets (one per worker), in worker-index order.
    host, port : where the router's uvicorn binds (the client-facing endpoint).

    The parent passes ONLY the router-end sockets here and closes the worker
    ends, so EOF is observable on worker death.
    """
    import setproctitle

    setproctitle.setproctitle("miles-session-router")
    _set_pdeathsig()

    async def _main() -> None:
        channels = await _open_channels(socks)
        router = SessionRouter(args, channels)
        app = build_app(router)
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    try:
        asyncio.run(_main())
    except (KeyboardInterrupt, IpcChannelClosed):
        pass
