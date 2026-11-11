"""An agent worker: runs agent-function calls through ``subproc`` on this host, for rollouts on another one.

A rollout with ``--custom-agent-function-mode worker`` sends each call here (protocol: ``miles.rollout.agentic.remote``).
The agent function must be importable here under the name the rollout loaded it by, and the paths in its forwarded
env must exist here.

    python -m miles.rollout.agentic.worker --host 0.0.0.0 --port 11100
"""

import argparse
import asyncio
import fnmatch
import hmac
import importlib
import json
import logging
import time
import traceback
from collections.abc import Sequence
from typing import Any

from aiohttp import web

from miles.rollout.agentic import rollout_limits
from miles.rollout.agentic.agent_function import call_agent_function
from miles.rollout.agentic.errors import InfraAbort
from miles.rollout.agentic.remote import MAX_POLL_WAIT_S, encode_report, worker_secret

logger = logging.getLogger(__name__)

_REAP_INTERVAL_S = 10.0
_ORPHAN_TIMEOUT_S = 120.0
_RETENTION_S = 3600.0


def _new_call() -> dict[str, Any]:
    return {
        "reports": [],
        "outcome": None,
        "changed": asyncio.Event(),
        "last_polled": time.monotonic(),
        "done_at": None,
        "task": None,
    }


def _add_reports(call: dict[str, Any], items: list[Any]) -> None:
    call["reports"].extend(encode_report(item) for item in items)
    call["changed"].set()


def _finish(call: dict[str, Any], outcome: dict[str, Any]) -> None:
    call["outcome"] = outcome
    call["done_at"] = time.monotonic()
    call["changed"].set()


def _call_env(forwarded: dict[str, str], keep_env: Sequence[str]) -> dict[str, str]:
    return {k: v for k, v in forwarded.items() if not any(fnmatch.fnmatchcase(k, p) for p in keep_env)}


def _resolve(path: str) -> Any:
    module_name, _, qualname = path.partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)  # config-access-exempt: resolves the qualname the caller's function was sent by
    return obj


async def _run(call: dict[str, Any], fn_path: str, kwargs: dict[str, Any], env: dict[str, str]) -> None:
    try:
        value = await call_agent_function(
            _resolve(fn_path),
            mode="subproc",
            on_child_reports=lambda items: _add_reports(call, items),
            extra_env=env,
            **kwargs,
        )
        json.dumps(value)
        _finish(call, {"kind": "ok", "value": value})
    except InfraAbort as exc:
        _finish(call, {"kind": "infra_abort", "exit_status": exc.exit_status, "message": str(exc)})
    except asyncio.CancelledError:
        _finish(call, {"kind": "cancelled"})
    except Exception as exc:
        _finish(
            call,
            {"kind": "error", "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        )


def create_app(
    *,
    orphan_timeout_s: float = _ORPHAN_TIMEOUT_S,
    retention_s: float = _RETENTION_S,
    keep_env: Sequence[str] = (),
    secret: str | None = None,
) -> web.Application:
    calls: dict[str, dict[str, Any]] = {}
    secret = secret or worker_secret()
    expected_authorization = f"Bearer {secret}".encode() if secret else None

    @web.middleware
    async def require_secret(request: web.Request, handler):
        if not hmac.compare_digest(request.headers.get("Authorization", "").encode(), expected_authorization):
            raise web.HTTPUnauthorized(text="missing or wrong agent worker secret")
        return await handler(request)

    async def submit(request: web.Request) -> web.Response:
        body = await request.json()
        call_id = str(body["call_id"])
        if call_id not in calls:
            call = calls[call_id] = _new_call()
            env = {**_call_env(body["env"], keep_env), rollout_limits.SCOPE_ENV_VAR: str(body["scope"])}
            call["task"] = asyncio.create_task(_run(call, body["fn"], body["kwargs"], env))
        return web.json_response({"call_id": call_id})

    async def poll(request: web.Request) -> web.Response:
        call = calls.get(request.match_info["call_id"])
        if call is None:
            raise web.HTTPNotFound(text="unknown call")
        after = int(request.query.get("after", 0))
        wait = min(max(float(request.query.get("wait", 0)), 0.0), MAX_POLL_WAIT_S)
        call["last_polled"] = time.monotonic()
        call["changed"].clear()
        if len(call["reports"]) <= after and call["outcome"] is None:
            try:
                await asyncio.wait_for(call["changed"].wait(), timeout=wait)
            except TimeoutError:
                pass
        call["last_polled"] = time.monotonic()
        return web.json_response(
            {
                "state": "running" if call["outcome"] is None else "done",
                "reports": call["reports"][after:],
                "next": len(call["reports"]),
                "outcome": call["outcome"],
            }
        )

    async def cancel(request: web.Request) -> web.Response:
        call_id = request.match_info["call_id"]
        call = calls.get(call_id)
        if call is None:
            calls[call_id] = tombstone = _new_call()
            _finish(tombstone, {"kind": "cancelled"})
        elif call["task"] is not None and not call["task"].done():
            call["task"].cancel()
        return web.json_response({"cancelled": True})

    async def health(request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "running": sum(c["outcome"] is None for c in calls.values())})

    async def reap() -> None:
        while True:
            await asyncio.sleep(_REAP_INTERVAL_S)
            now = time.monotonic()
            for call_id, call in list(calls.items()):
                if call["outcome"] is None and now - call["last_polled"] > orphan_timeout_s:
                    logger.warning(f"cancelling call {call_id}: no poll for {orphan_timeout_s:.0f}s")
                    call["task"].cancel()
                elif call["done_at"] is not None and now - call["done_at"] > retention_s:
                    del calls[call_id]

    async def start_reaper(app: web.Application):
        task = asyncio.create_task(reap())
        yield
        task.cancel()
        for call in calls.values():
            if call["task"] is not None and not call["task"].done():
                call["task"].cancel()

    app = web.Application(
        client_max_size=64 * 1024 * 1024, middlewares=[require_secret] if expected_authorization else []
    )
    app.add_routes(
        [
            web.post("/calls", submit),
            web.get("/calls/{call_id}", poll),
            web.delete("/calls/{call_id}", cancel),
            web.get("/health", health),
        ]
    )
    app.cleanup_ctx.append(start_reaper)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11100)
    parser.add_argument(
        "--orphan-timeout",
        type=float,
        default=_ORPHAN_TIMEOUT_S,
        help="Cancel a call nobody has polled for this many seconds (its caller is gone).",
    )
    parser.add_argument(
        "--keep-env",
        nargs="*",
        default=[],
        help="Env var names (fnmatch patterns) whose value on this host wins over the one a call forwards, "
        "such as a task directory that lives at another path here.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    web.run_app(
        create_app(orphan_timeout_s=args.orphan_timeout, keep_env=args.keep_env), host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
