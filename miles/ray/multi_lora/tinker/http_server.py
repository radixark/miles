"""Tinker wire layer; mount via --multi-lora-http-server-path miles.ray.multi_lora.tinker.http_server.TinkerHTTPServer."""

import asyncio
import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from miles.ray.multi_lora.http_server import MultiLoRAHTTPServer
from miles.ray.multi_lora.operations import BadRequest, OperationQueue, QueueFull, payload_fingerprint
from miles.utils.adapter_config import AdapterRunConfig

KIND_BY_PATH = {
    "/api/v1/forward_backward": "forward_backward",
    "/api/v1/forward": "forward",
    "/api/v1/optim_step": "optim_step",
    "/api/v1/save_weights": "save_state",
    "/api/v1/load_weights": "load_state",
    "/api/v1/save_weights_for_sampler": "save_weights_for_sampler",
}
# Excluded from retry identity: the SDK re-mints this inside every retry attempt.
FP_EXCLUDE = {"sampling_session_seq_id"}

logger = logging.getLogger(__name__)


class TinkerHTTPServer(MultiLoRAHTTPServer):
    """Serves the tinker SDK REST surface (tinker==0.24.1 wire contract)."""

    def __init__(self, backend, host="127.0.0.1", api_port=0):
        super().__init__(backend, host, api_port)
        assert hasattr(
            backend, "operation_queue"
        ), "TinkerHTTPServer needs MultiLoRAOperationBackend (--multi-lora-backend-path)"
        self._sessions: dict[str, dict] = {}
        self._models: dict[str, dict] = {}  # model_id -> {"name", "rank"}
        self._queues: dict[str, OperationQueue] = {}  # model_id -> training-plane queue
        self._ready_futures: dict[str, dict] = {}  # request_id -> terminal body
        self.poll_window_s = 15.0
        self.poll_interval_s = 0.1

    def create_app(self) -> FastAPI:
        app = super().create_app()

        @app.exception_handler(QueueFull)
        async def queue_full_handler(request: Request, exc: QueueFull):
            return JSONResponse({"detail": exc.client_detail}, status_code=429, headers={"Retry-After": "1"})

        @app.exception_handler(RuntimeError)
        async def runtime_error_handler(request: Request, exc: RuntimeError):
            # The SDK retries 409 forever, so capacity maps to 429; everything else is a real 500.
            if "No free adapter slots" in str(exc):
                detail = "no free adapter slots; retry shortly"
                return JSONResponse({"detail": detail}, status_code=429, headers={"Retry-After": "1"})
            logger.exception("unhandled server error")  # internals stay in the server log, never in the response
            return JSONResponse({"detail": "internal server error"}, status_code=500)

        return app

    def add_routes(self, app: FastAPI) -> None:
        super().add_routes(app)
        app.get("/api/v1/get_server_capabilities")(self.get_server_capabilities)
        app.post("/api/v1/client/config")(self.client_config)
        app.post("/api/v1/create_session")(self.create_session)
        app.post("/api/v1/session_heartbeat")(self.session_heartbeat)
        app.post("/api/v1/telemetry")(self.telemetry)
        app.post("/api/v1/create_model")(self.create_model)
        app.post("/api/v1/get_info")(self.get_info)
        app.post("/api/v1/retrieve_future")(self.retrieve_future)
        for path in KIND_BY_PATH:
            app.post(path)(self.enqueue_operation_for_trainer)

    async def get_server_capabilities(self) -> dict:
        # One trainer serves one base model; adapters register on top of it.
        return {"supported_models": [{"model_name": self.backend.args.hf_checkpoint}]}

    # ------------------------------ session bootstrap ------------------------------

    async def client_config(self, request: Request) -> dict:
        # Empty response selects every SDK default: api-key auth, JSON wire, parallel chunks.
        return {}

    async def create_session(self, request: Request) -> dict:
        session_id = f"sess-{uuid.uuid4().hex[:16]}"
        self._sessions[session_id] = {"last_heartbeat": time.monotonic()}
        return {"type": "create_session", "session_id": session_id}

    async def session_heartbeat(self, request: Request) -> dict:
        session = self._sessions.get((await request.json()).get("session_id"))
        if session is not None:
            session["last_heartbeat"] = time.monotonic()
        return {"type": "session_heartbeat"}

    async def telemetry(self, request: Request) -> dict:
        return {"status": "accepted"}

    # ------------------------------ model lifecycle ------------------------------

    async def create_model(self, request: Request):
        body = await request.json()
        session_id = body.get("session_id")
        if session_id not in self._sessions:
            return JSONResponse({"detail": f"unknown session '{session_id}'"}, status_code=404)
        model_id = f"{session_id}:train:{body['model_seq_id']}"
        request_id = f"{model_id}:create"
        if model_id not in self._models:  # an SDK retry replays the same ack
            name = f"tinker-{session_id.removeprefix('sess-')[:8]}-t{body['model_seq_id']}"
            lora = body.get("lora_config") or {}
            await self.backend.register(
                name, AdapterRunConfig(data="", rank=lora.get("rank"), alpha=lora.get("alpha"))
            )
            self._models[model_id] = {"name": name, "rank": lora.get("rank")}
            self._queues[model_id] = self.backend.operation_queue(name)
            self._ready_futures[request_id] = {"type": "create_model", "model_id": model_id}
        return {"request_id": request_id, "model_id": model_id}

    async def get_info(self, request: Request):
        model_id = (await request.json()).get("model_id")
        model = self._models.get(model_id)
        if model is None:
            return JSONResponse({"detail": f"unknown model '{model_id}'"}, status_code=404)
        base = self.backend.args.hf_checkpoint
        # arch/tokenizer_id are opaque strings to the SDK; the base checkpoint names both.
        return {
            "type": "get_info",
            "model_id": model_id,
            "model_name": base,
            "model_data": {"arch": base, "model_name": base, "tokenizer_id": base},
            "is_lora": True,
            "lora_rank": model["rank"],
        }

    # ------------------------------ training operations ------------------------------

    async def enqueue_operation_for_trainer(self, request: Request):
        body = await request.json()
        model_id = body.get("model_id")
        seq_id = body.get("seq_id")
        queue = self._queues.get(model_id)
        if queue is None:
            return JSONResponse({"detail": f"unknown model '{model_id}'"}, status_code=404)
        if not isinstance(seq_id, int) or seq_id < 1:
            return JSONResponse({"detail": "seq_id must be an integer >= 1"}, status_code=400)
        request_id = f"{model_id}:op{seq_id}"
        kind = KIND_BY_PATH[request.url.path]
        # Identity hashes the body MINUS volatile fields; the stored execution payload keeps the full body.
        fingerprint = payload_fingerprint(kind, {k: v for k, v in body.items() if k not in FP_EXCLUDE})
        try:
            queue.enqueue(seq_id, request_id, kind, body, fingerprint=fingerprint)
        except BadRequest as exc:
            # 422: same identity retried with different content; the SDK treats it as fatal.
            return JSONResponse({"detail": exc.client_detail}, status_code=422)
        return {"request_id": request_id}

    async def retrieve_future(self, request: Request):
        request_id = (await request.json()).get("request_id")
        result = self._ready_futures.get(request_id)
        if result is not None:
            return result
        model_id, _, seq = request_id.rpartition(":op") if request_id else ("", "", "")
        queue = self._queues.get(model_id)
        if queue is None or not seq.isdigit():
            # 410 marks a broken/unknown promise; the SDK treats it as retryable, never fatal.
            return JSONResponse({"detail": f"no result for request '{request_id}'"}, status_code=410)
        deadline = time.monotonic() + self.poll_window_s
        while True:
            try:
                kind, payload = queue.poll(int(seq))
            except BadRequest:
                return JSONResponse({"detail": f"no record of request '{request_id}'"}, status_code=410)
            if kind != "try_again":
                return payload
            if time.monotonic() >= deadline:
                return {"type": "try_again", "queue_state": "active"}
            await asyncio.sleep(self.poll_interval_s)
