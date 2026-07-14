"""Multi-process load generator for the session-server HTTP benchmark.

Lives beside `bench_session_server_overhead` so it is importable by qualified
module name in spawn children (`sys.path[0]` is the script dir, inherited by
the spawn bootstrap) — the same mechanism `_mock_r3_backend` relies on.

Why a separate, multi-process driver
------------------------------------
A single asyncio event loop + one httpx client cannot cleanly drive ~1000
concurrent sessions over loopback: it cycles the ephemeral-port range and resets
pooled connections under its own scheduling pressure, surfacing as
`httpx.ReadError` on a chat POST — which is NOT retried (chat mutates session
state) — and aborting the whole run, even though the server stays healthy
(every answered request is 200, no worker death, no OOM). Sharding the sessions
across `--bench-driver-procs` OS processes (each its own loop + client) keeps
per-loop connection volume low and mirrors the real rollout, where many workers
drive sessions rather than one process.

Robust per-session driving
---------------------------
Each session is driven defensively: a failure is CLASSIFIED (server non-2xx vs
client transport reset, for chat vs the GET-records read) and tallied, never
raised. One reset therefore does not abort a generator — it becomes a measurable
rate. That is what lets the bench answer "does it stay stable" instead of dying
on the first hiccup.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

_STATUS_KEYS = (
    "completed_turns",
    "chat_server_errors",
    "chat_transport_errors",
    "get_ok",
    "get_server_errors",
    "get_transport_errors",
)


def _transport_error_types() -> tuple:
    """Client-side connection/transport faults (NOT server non-2xx responses)."""
    import httpx

    return (
        httpx.ReadError,
        httpx.WriteError,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
        httpx.PoolTimeout,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
    )


async def _idempotent_request(client, method: str, url: str):
    """Issue an idempotent control-plane request, retrying ONCE on a connection-
    level transport fault with a fresh connection.

    A reset surfaces client-side BEFORE any handler ran, so the request had no
    effect — reissuing it on a new connection is correct transport-fault
    handling, not retrying a server-side failure (a non-2xx is surfaced, not
    retried). Only create/get/delete come through here; chat is never retried
    (it mutates session state).
    """
    import httpx

    try:
        return await client.request(method, url)
    except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError):
        return await client.request(method, url)


def _empty_status() -> dict[str, Any]:
    st: dict[str, Any] = {k: 0 for k in _STATUS_KEYS}
    st["first_error"] = None
    return st


async def drive_one_session(
    client, base_url, request_bodies, samples, *, get_records, tool_interval=0.0
) -> dict[str, Any]:
    """create -> N x chat (-> GET records) -> delete for one session, robustly.

    Never raises: returns a per-session status dict (completed turns + classified
    error counts). Records per-chat reply latency in `samples['reply_latency_ms']`
    and, when `get_records`, the GET latency in `samples['get_records_ms']`.
    A chat transport reset / non-2xx ends that session (its state is unknown / a
    server error is real) but does not abort the generator. `tool_interval` is
    the idle a session waits after each chat (except the last) before its next
    turn (simulated tool/env step).
    """
    transport_errors = _transport_error_types()
    st = _empty_status()

    try:
        r = await _idempotent_request(client, "POST", f"{base_url}/sessions")
    except transport_errors as e:
        st["chat_transport_errors"] += 1
        st["first_error"] = f"create transport {type(e).__name__}"
        return st
    if r.status_code != 200:
        st["chat_server_errors"] += 1
        st["first_error"] = f"create {r.status_code}"
        return st
    sid = r.json()["session_id"]

    last_idx = len(request_bodies) - 1
    for idx, body in enumerate(request_bodies):
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"{base_url}/sessions/{sid}/v1/chat/completions",
                content=body,
                headers={"content-type": "application/json"},
            )
        except transport_errors as e:
            st["chat_transport_errors"] += 1
            st["first_error"] = st["first_error"] or f"chat[{idx}] transport {type(e).__name__}"
            return st
        samples["reply_latency_ms"].append((time.perf_counter() - t0) * 1000)
        if resp.status_code != 200:
            st["chat_server_errors"] += 1
            st["first_error"] = st["first_error"] or f"chat[{idx}] {resp.status_code} {resp.text[:120]}"
            return st
        st["completed_turns"] += 1
        if tool_interval > 0 and idx != last_idx:
            await asyncio.sleep(tool_interval)

    if get_records:
        t0 = time.perf_counter()
        g = None
        try:
            g = await _idempotent_request(client, "GET", f"{base_url}/sessions/{sid}")
        except transport_errors as e:
            st["get_transport_errors"] += 1
            st["first_error"] = st["first_error"] or f"get transport {type(e).__name__}"
        if g is not None:
            if g.status_code == 200:
                samples["get_records_ms"].append((time.perf_counter() - t0) * 1000)
                st["get_ok"] += 1
            else:
                st["get_server_errors"] += 1
                st["first_error"] = st["first_error"] or f"get {g.status_code}"

    try:
        await _idempotent_request(client, "DELETE", f"{base_url}/sessions/{sid}")
    except transport_errors:
        pass
    return st


def _empty_agg() -> dict[str, Any]:
    agg: dict[str, Any] = {k: 0 for k in _STATUS_KEYS}
    agg["sessions_ok"] = 0
    agg["first_errors"] = []
    return agg


async def lg_drive_all(base_urls, request_bodies, num_sessions, get_records, tool_interval):
    """Drive `num_sessions` sessions concurrently in ONE event loop/client.

    Each session picks one instance URL at create time and stays on it for its
    whole lifecycle — the URL is the router, mirroring
    `OpenAIEndpointTracer.create`. Returns (samples, agg). Used directly for the
    single-process path and as the body of each spawned load generator.
    """
    import httpx

    samples: dict[str, list] = {"reply_latency_ms": []}
    if get_records:
        samples["get_records_ms"] = []
    # keepalive_expiry < the server's keep-alive window so the CLIENT drops an
    # idle connection first (a fresh connection, not a reset) rather than reusing
    # one the server already closed.
    limits = httpx.Limits(
        max_connections=max(2, num_sessions * 2),
        max_keepalive_connections=max(2, num_sessions * 2),
        keepalive_expiry=2.0,
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), limits=limits) as client:
        for base_url in base_urls:
            await client.get(f"{base_url}/health")
        statuses = await asyncio.gather(
            *(
                drive_one_session(
                    client,
                    random.choice(base_urls),
                    request_bodies,
                    samples,
                    get_records=get_records,
                    tool_interval=tool_interval,
                )
                for _ in range(num_sessions)
            )
        )

    agg = _empty_agg()
    for s in statuses:
        for k in _STATUS_KEYS:
            agg[k] += s[k]
        if s["chat_server_errors"] == 0 and s["chat_transport_errors"] == 0:
            agg["sessions_ok"] += 1
        if s["first_error"] and len(agg["first_errors"]) < 5:
            agg["first_errors"].append(s["first_error"])
    return samples, agg


def load_generator_entry(base_urls, request_bodies, num_sessions, get_records, tool_interval, result_queue) -> None:
    """spawn `Process` target: drive a shard of sessions, put (samples, agg) on
    `result_queue`. Drives no tokenizer — it only replays pre-built request
    bodies — so generator startup is cheap."""
    try:
        import setproctitle

        setproctitle.setproctitle("miles-bench-loadgen")
    except Exception:
        pass
    samples, agg = asyncio.run(lg_drive_all(base_urls, request_bodies, num_sessions, get_records, tool_interval))
    result_queue.put((samples, agg))
