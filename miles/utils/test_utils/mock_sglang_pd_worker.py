"""Scripted SGLang HTTP worker for sgl-model-gateway e2e tests on CPU.

Extends :class:`MockSGLangServer` with the discovery endpoints the router reads when a worker registers
(``/server_info``, ``/model_info`` and their ``/get_*`` fallbacks), a ``/flush_cache`` recorder, and a per-request
script hook so a prefill/decode pair can stage PD-router scenarios: expert-row envelopes, missing fields, a slow
error body, and client-disconnect timing. Responses still come from the tokenizer-driven base class; no model
inference happens.

Each worker runs its own uvicorn thread and event loop, so cross-worker signalling uses ``threading.Event`` polled
from the handlers rather than ``asyncio.Event``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessFn, default_process_fn

WorkerSide = Literal["regular", "prefill", "decode"]

_POLL_INTERVAL_S = 0.005


async def _wait_event(event: threading.Event, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while not event.is_set():
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_POLL_INTERVAL_S)
    return True


@dataclass
class PDScenario:
    """Events and timestamps shared by one prefill/decode pair for a single staged request.

    Use a fresh instance per staged request (handlers keep a reference to the one they started with).
    """

    decode_started: threading.Event = field(default_factory=threading.Event)
    prefill_headers_sent: threading.Event = field(default_factory=threading.Event)
    prefill_headers_sent_at: float | None = None
    prefill_body_done_at: float | None = None
    decode_disconnected_at: float | None = None


@dataclass
class ErrorScript:
    """Answer ``/generate`` with an error status whose body arrives late (prefill side of PR #17)."""

    status: int = 500
    body: bytes = b"original prefill failure"
    body_delay_s: float = 1.5
    # Wait for the decode request to be in flight before sending the error headers, so the router
    # really has a decode side to cancel.
    wait_for_decode_s: float = 5.0


@dataclass
class HoldScript:
    """Keep the ``/generate`` request open until the router cancels it (decode side of PR #17)."""

    mode: Literal["await_headers", "holding_response"]
    disconnect_timeout_s: float = 3.0


@dataclass
class WorkerScript:
    """Per-worker, per-case response mutations. ``reset()`` restores the plain mock behaviour."""

    meta_info_overrides: dict[str, Any] = field(default_factory=dict)
    drop_meta_keys: set[str] = field(default_factory=set)
    error: ErrorScript | None = None
    hold: HoldScript | None = None

    def reset(self) -> None:
        self.meta_info_overrides = {}
        self.drop_meta_keys = set()
        self.error = None
        self.hold = None


class ScriptedSGLangWorker(MockSGLangServer):
    def __init__(
        self,
        *,
        side: WorkerSide,
        script: WorkerScript | None = None,
        scenario: PDScenario | None = None,
        served_model_name: str = "chat-model",
        advertised_model_path: str | None = None,
        advertised_tokenizer_path: str | None = None,
        model_name: str = "Qwen/Qwen3-0.6B",
        process_fn: ProcessFn = default_process_fn,
        host: str | None = "127.0.0.1",
        port: int | None = None,
        latency: float = 0.0,
    ):
        # The base constructor calls _setup_routes(), so everything the routes read must exist first.
        self.side: WorkerSide = side
        self.script = script if script is not None else WorkerScript()
        self.scenario = scenario if scenario is not None else PDScenario()
        self.served_model_name = served_model_name
        self.advertised_model_path = advertised_model_path
        self.advertised_tokenizer_path = advertised_tokenizer_path
        self.flush_cache_calls: list[dict[str, str]] = []
        # /generate handlers (including streaming bodies) still running; tests wait for 0 between cases so a
        # late disconnect from a previous request cannot leak into the next scenario.
        self.inflight = 0
        super().__init__(model_name=model_name, process_fn=process_fn, host=host, port=port, latency=latency)

    # ------------------------------------------------------------------ routes

    def _setup_routes(self) -> None:
        super()._setup_routes()

        @self.app.get("/server_info")
        @self.app.get("/get_server_info")
        async def server_info():
            return JSONResponse(content=self.server_info())

        @self.app.get("/model_info")
        @self.app.get("/get_model_info")
        async def model_info():
            return JSONResponse(content=self.model_info())

        @self.app.get("/health_generate")
        async def health_generate():
            return JSONResponse(content={"status": "ok"})

        @self.app.post("/flush_cache")
        @self.app.get("/flush_cache")
        async def flush_cache(request: Request):
            self.flush_cache_calls.append(dict(request.query_params))
            return JSONResponse(content={"status": "ok"})

    def server_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "served_model_name": self.served_model_name,
            "tp_size": 1,
            "dp_size": 1,
            "version": "mock",
        }
        if self.advertised_model_path is not None:
            info["model_path"] = self.advertised_model_path
        return info

    def model_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"is_generation": True}
        if self.advertised_model_path is not None:
            info["model_path"] = self.advertised_model_path
        if self.advertised_tokenizer_path is not None:
            info["tokenizer_path"] = self.advertised_tokenizer_path
        return info

    # ------------------------------------------------------------ generate hook

    async def _handle_generate_like_request(self, request: Request, compute_fn):
        payload = await request.json()
        self.request_log.append(payload)
        script = self.script
        self.inflight += 1
        streaming = False
        try:
            if script.error is not None:
                streaming = True
                return await self._error_response(script.error)
            if script.hold is not None:
                response = await self._hold_response(request, script.hold, compute_fn, payload)
                streaming = isinstance(response, StreamingResponse)
                return response
            with self._concurrency.track():
                if self.latency > 0:
                    await asyncio.sleep(self.latency)
                body = self._compute(compute_fn, payload)
            return JSONResponse(content=body)
        finally:
            if not streaming:
                self.inflight -= 1  # streaming bodies decrement when their generator finishes

    def _compute(self, compute_fn, payload: dict) -> Any:
        input_ids = payload.get("input_ids")
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            # Batched /generate: one response object per prompt, as SGLang does.
            return [self._apply_meta_script(compute_fn({**payload, "input_ids": ids})) for ids in input_ids]
        return self._apply_meta_script(compute_fn(payload))

    def _apply_meta_script(self, body: dict) -> dict:
        meta = body.get("meta_info")
        if meta is None and body.get("choices"):
            meta = body["choices"][0].get("meta_info")
        if isinstance(meta, dict):
            meta.update(self.script.meta_info_overrides)
            for key in self.script.drop_meta_keys:
                meta.pop(key, None)
        return body

    async def _error_response(self, error: ErrorScript):
        scenario = self.scenario
        await _wait_event(scenario.decode_started, error.wait_for_decode_s)

        async def body():
            try:
                scenario.prefill_headers_sent_at = time.monotonic()
                scenario.prefill_headers_sent.set()
                await asyncio.sleep(error.body_delay_s)
                yield error.body
                scenario.prefill_body_done_at = time.monotonic()
            finally:
                self.inflight -= 1

        headers = {"content-length": str(len(error.body))}
        return StreamingResponse(body(), status_code=error.status, media_type="text/plain", headers=headers)

    async def _hold_response(self, request: Request, hold: HoldScript, compute_fn, payload: dict):
        scenario = self.scenario
        scenario.decode_started.set()
        if hold.mode == "await_headers":
            await _wait_event(scenario.prefill_headers_sent, 5.0)
            deadline = time.monotonic() + hold.disconnect_timeout_s
            while time.monotonic() < deadline:
                if await request.is_disconnected():
                    scenario.decode_disconnected_at = time.monotonic()
                    break
                await asyncio.sleep(0.01)
            return JSONResponse(content=self._compute(compute_fn, payload))

        async def body():
            try:
                yield b" "
                await _wait_event(scenario.prefill_headers_sent, 5.0)
                deadline = time.monotonic() + hold.disconnect_timeout_s
                while time.monotonic() < deadline and scenario.prefill_body_done_at is None:
                    await asyncio.sleep(0.01)
                    yield b" "
            except (asyncio.CancelledError, GeneratorExit):
                # uvicorn cancels the body iteration when the client goes away.
                scenario.decode_disconnected_at = time.monotonic()
                raise
            finally:
                self.inflight -= 1

        return StreamingResponse(body(), media_type="application/json")
