"""HTTP surface for the tinker frontend: /api/v1 as ``tinker==0.24.1`` speaks it.

Extends the controller's registration server (selected via
``--tinker-frontend`` / ``--multi-lora-http-server-path``), so the SDK
protocol and the operator plane share one uvicorn on the head node — but not
one trust domain: the operator routes (/adapter_runs*, /info) accept
loopback peers only, whatever the bind. The SDK ``X-API-Key`` authenticates
/api/v1/* and never grants the operator plane (which reads server-local
yaml_path files, chooses save paths, and deregisters tenants) to a remote
caller. When a key is configured, every route except the health probes
additionally requires it; a non-loopback bind without a key refuses to
start (fail closed).

Error mapping (what the 0.24.1 SDK does with each status, observed):
- 429 + Retry-After  <- backend backpressure (SDK retries with backoff)
- 422                <- same-identity/different-payload conflicts (fatal to
                        the SDK; 409 must never be used — the SDK retries it)
- 400/404/401        <- malformed/unknown/unauthenticated (fatal)
- 410                <- expired/unknown future. The SDK does NOT re-run the
                        original training request: it raises a retryable
                        "promise expired/broken" toward the caller. Delivered
                        results answer 410 from a fingerprint tombstone, so
                        an identical late retry is typed instead of silently
                        re-executing.
- payload rejections on a spent seq_id are NOT HTTP errors: they become
  terminal FAILED(user) futures so the ordinal stays consumed.
"""

import hmac
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from miles.ray.multi_lora.http_server import AdapterRunControlServer
from miles.ray.multi_lora.operations import OperationBackpressure
from miles.ray.tinker_frontend import wire
from miles.ray.tinker_frontend.service import ApiError, TinkerFrontend

AUTH_EXEMPT_PATHS = ("/health", "/api/v1/healthz")
API_KEY_ENV = "MILES_TINKER_API_KEY"
# The operator plane stays node-local even on a public bind; the SDK key is
# a client credential, not an operator one.
LOOPBACK_PEERS = ("127.0.0.1", "::1", "localhost")


def is_sdk_path(path: str) -> bool:
    """/api/v1/* plus the base liveness probe; everything else is operator."""
    return path.startswith("/api/v1/") or path == "/health"


def resolve_api_key(args: Any) -> str | None:
    return getattr(args, "tinker_api_key", None) or os.environ.get(API_KEY_ENV) or None


def resolve_sampling_max_context(args: Any) -> int | None:
    """Static engine context limit for the sampling preflight: the explicit
    tinker flag wins, else the context length this deployment itself launched
    its engines with (--sglang-context-length). None defers to lazy discovery
    from the router's /get_server_info on the first sample."""
    return getattr(args, "tinker_sampling_max_context", None) or getattr(args, "sglang_context_length", None) or None


class TinkerFrontendHTTPServer(AdapterRunControlServer):
    """The registration server + the official tinker SDK protocol."""

    def __init__(self, backend, host="127.0.0.1", api_port=0):
        super().__init__(backend, host, api_port)
        args = backend.args
        self.frontend = TinkerFrontend(
            backend,
            # Aggregate sampling cap across ALL SDK clients, in sub-generation
            # units (the per-client SDK limit of 64 never bounded the sum).
            sampling_max_active_subgenerations=getattr(args, "tinker_sampling_max_active_subgenerations", 64),
            sampling_max_context=resolve_sampling_max_context(args),
            session_idle_ttl_s=getattr(args, "tinker_session_idle_ttl", 3600.0),
            future_unpolled_ttl_s=getattr(args, "tinker_future_unpolled_ttl", 900.0),
            future_undelivered_ttl_s=getattr(args, "tinker_future_undelivered_ttl", 3600.0),
        )
        self.api_key = resolve_api_key(backend.args)

    async def start(self) -> None:
        if self.host not in ("127.0.0.1", "localhost", "::1") and not self.api_key:
            raise RuntimeError(
                f"refusing to bind the tinker frontend to '{self.host}' without an API key: "
                f"pass --tinker-api-key or set {API_KEY_ENV}"
            )
        await super().start()
        # The reaper + metrics-summary loop lives with the serving surface:
        # started only once the server accepts traffic, torn down by stop()
        # through frontend.close().
        self.frontend.start_maintenance()

    async def stop(self) -> None:
        # Order matters: stop ACCEPTING first (uvicorn), then drain the
        # frontend (cancel + await in-flight samples, close the transport).
        # Closing the frontend first would let a late request lazily reopen
        # the transport it just closed. Idempotent: a second stop is a no-op.
        if getattr(self, "_stopped", False):
            return
        self._stopped = True
        await super().stop()
        await self.frontend.close()

    def create_app(self) -> FastAPI:
        app = super().create_app()

        @app.exception_handler(ApiError)
        async def api_error_handler(request: Request, exc: ApiError):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

        @app.exception_handler(OperationBackpressure)
        async def backpressure_handler(request: Request, exc: OperationBackpressure):
            # Retryable by contract: the SDK backs off and resends the same
            # request, which the deterministic request ids dedupe.
            return JSONResponse({"detail": str(exc)}, status_code=429, headers={"Retry-After": "1"})

        key = self.api_key.encode() if self.api_key is not None else None

        @app.middleware("http")
        async def guard(request: Request, call_next):
            path = request.url.path
            if key is not None and path not in AUTH_EXEMPT_PATHS:
                supplied = request.headers.get("x-api-key", "").encode()
                if not hmac.compare_digest(supplied, key):
                    return JSONResponse({"detail": "invalid or missing X-API-Key"}, status_code=401)
            if not is_sdk_path(path):
                # Operator plane: node-local only, key or no key. A missing
                # peer identity fails closed.
                client = request.client
                if client is None or client.host not in LOOPBACK_PEERS:
                    return JSONResponse(
                        {"detail": "operator routes are loopback-only; the SDK surface is /api/v1/*"},
                        status_code=403,
                    )
            return await call_next(request)

        return app

    def add_routes(self, app: FastAPI) -> None:
        super().add_routes(app)
        frontend = self.frontend

        # -------- bootstrap / session --------
        @app.get("/api/v1/healthz")
        async def healthz() -> dict:
            return frontend.health()

        @app.get("/api/v1/get_server_capabilities")
        async def get_server_capabilities() -> dict:
            return frontend.capabilities()

        @app.post("/api/v1/client/config")
        async def client_config(request: wire.ClientConfigRequest) -> dict:
            return frontend.client_config(request)

        @app.post("/api/v1/create_session")
        async def create_session(request: wire.CreateSessionRequest) -> dict:
            return frontend.create_session(request)

        @app.post("/api/v1/session_heartbeat")
        async def session_heartbeat(request: wire.SessionHeartbeatRequest) -> dict:
            return frontend.session_heartbeat(request)

        @app.post("/api/v1/telemetry")
        async def telemetry(request: Request) -> dict:
            return frontend.telemetry(await request.body())

        # -------- models --------
        @app.post("/api/v1/create_model")
        async def create_model(request: wire.CreateModelRequest) -> dict:
            return await frontend.create_model(request)

        @app.post("/api/v1/get_info")
        async def get_info(request: wire.GetInfoRequest) -> dict:
            return frontend.get_info(request)

        @app.post("/api/v1/unload_model")
        async def unload_model(request: wire.UnloadModelRequest) -> dict:
            return await frontend.unload_model(request)

        # -------- training --------
        @app.post("/api/v1/forward_backward")
        async def forward_backward(request: wire.ForwardBackwardRequest) -> dict:
            return frontend.forward_backward(request)

        @app.post("/api/v1/forward")
        async def forward(request: wire.ForwardRequest) -> dict:
            return frontend.forward(request)

        @app.post("/api/v1/optim_step")
        async def optim_step(request: wire.OptimStepRequest) -> dict:
            return frontend.optim_step(request)

        # -------- checkpoints --------
        @app.post("/api/v1/save_weights")
        async def save_weights(request: wire.SaveWeightsRequest) -> dict:
            return frontend.save_weights(request)

        @app.post("/api/v1/load_weights")
        async def load_weights(request: wire.LoadWeightsRequest) -> dict:
            return frontend.load_weights(request)

        @app.post("/api/v1/weights_info")
        async def weights_info(request: wire.WeightsInfoRequest) -> dict:
            return frontend.weights_info(request)

        # -------- sampling --------
        @app.post("/api/v1/save_weights_for_sampler")
        async def save_weights_for_sampler(request: wire.SaveWeightsForSamplerRequest) -> dict:
            return frontend.save_weights_for_sampler(request)

        @app.post("/api/v1/create_sampling_session")
        async def create_sampling_session(request: wire.CreateSamplingSessionRequest) -> dict:
            return frontend.create_sampling_session(request)

        @app.get("/api/v1/samplers/{sampler_id}")
        async def get_sampler(sampler_id: str) -> dict:
            return frontend.get_sampler(sampler_id)

        @app.post("/api/v1/asample")
        async def asample(request: wire.SampleRequest) -> dict:
            return frontend.sample(request)

        # -------- futures --------
        @app.post("/api/v1/retrieve_future")
        async def retrieve_future(request: wire.FutureRetrieveRequest) -> dict:
            return await frontend.retrieve_future(request)
