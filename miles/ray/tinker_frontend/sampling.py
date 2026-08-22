"""Sampling transport for the tinker frontend
(codex-rollout-fullparameter-design-0810 §4.6).

The sampling hot path stays frontend -> router: /asample answers with a
future immediately and a background task posts the generation itself. This
port isolates WHERE that post goes — the SGLang router today, whatever
endpoint the InferenceController advertises after PR #1842 — without ever
proxying per-sample traffic through a rollout component. Serving identity,
versions, and session invalidation stay in the tinker backend/frontend:
only the HTTP hop lives here."""

import asyncio
from typing import Protocol

import httpx


class SamplingTransport(Protocol):
    async def generate(self, payload: dict) -> dict: ...

    async def server_info(self) -> dict: ...

    async def close(self) -> None: ...


class SGLangRouterSamplingTransport:
    """Direct router transport with an explicit hard bound on in-flight
    generations (lazy client creation on the first request, like before).

    The previous default-configured client carried an implicit
    ``max_connections=100`` pool with a 10-second pool timeout: above 100
    concurrent generations (2 SDK clients x 64, before ``num_samples``
    fan-out) request #101 died waiting for a connection — an empty-message
    ``PoolTimeout`` the frontend turned into a terminal server failure the
    SDK never retries (the Tau 100/28 sampling cliff). The bound here is the
    transport-level invariant behind the frontend's weighted admission: even
    a caller that bypasses admission cannot stampede the router."""

    def __init__(self, base_url: str, max_inflight: int = 64) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_inflight = max_inflight
        # Acquired INSIDE each per-sample generation task (not at submit):
        # `async with` guarantees a sibling-cancelled or shutdown-cancelled
        # generation releases its permit on the way out.
        self._gate = asyncio.Semaphore(max_inflight)
        # The pool matches the gate, and pool=None removes the 10s pool
        # deadline. That is safe ONLY because the semaphore keeps in-flight
        # requests <= max_connections, so a request never actually queues on
        # the pool: legal, bounded waiting happens on the gate instead of
        # being misclassified as a terminal PoolTimeout. Read stays at 600s
        # (the value this frontend always used) — deriving it from router
        # config is deliberately out of scope here.
        self.limits = httpx.Limits(max_connections=max_inflight, max_keepalive_connections=max_inflight)
        self.timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=None)
        self._http: httpx.AsyncClient | None = None

    async def generate(self, payload: dict) -> dict:
        async with self._gate:
            if self._http is None:
                self._http = httpx.AsyncClient(limits=self.limits, timeout=self.timeout)
            response = await self._http.post(f"{self.base_url}/generate", json=payload)
            response.raise_for_status()
            return response.json()

    async def server_info(self) -> dict:
        """Engine server info for the frontend's context-limit discovery,
        via a dedicated short-timeout client — an info probe must neither
        take a generation permit nor wait behind a saturated pool.

        Two shapes exist behind one URL (verified live on H200): a bare
        SGLang engine answers /get_server_info with its ServerArgs +
        scheduler_info (context_length / max_req_input_len present), while
        sglang-router >= 0.3 answers with router metadata
        ({"router_manager": true, ...}) and keeps the engines one hop away
        behind /workers. When the first answer carries no engine fields,
        hop to the first healthy worker — miles deployments run homogeneous
        engines, so any worker's limit is the deployment's limit."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/get_server_info")
            response.raise_for_status()
            info = response.json()
            if isinstance(info, dict) and ("context_length" in info or "max_req_input_len" in info):
                return info
            workers_response = await client.get(f"{self.base_url}/workers")
            workers_response.raise_for_status()
            workers = (workers_response.json() or {}).get("workers") or []
            urls = [row.get("url") for row in workers if row.get("url") and row.get("is_healthy", True)]
            if not urls:
                return info if isinstance(info, dict) else {}
            response = await client.get(f"{urls[0].rstrip('/')}/get_server_info")
            response.raise_for_status()
            return response.json()

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
