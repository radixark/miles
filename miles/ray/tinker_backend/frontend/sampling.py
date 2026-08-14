"""Sampling transport for the tinker frontend
(codex-rollout-fullparameter-design-0810 §4.6).

The sampling hot path stays frontend -> router: /asample answers with a
future immediately and a background task posts the generation itself. This
port isolates WHERE that post goes — the SGLang router today, whatever
endpoint the InferenceController advertises after PR #1842 — without ever
proxying per-sample traffic through a rollout component. Serving identity,
versions, and session invalidation stay in the tinker backend/frontend:
only the HTTP hop lives here."""

from typing import Protocol

import httpx


class SamplingTransport(Protocol):
    async def generate(self, payload: dict) -> dict: ...

    async def close(self) -> None: ...


class SGLangRouterSamplingTransport:
    """Direct router transport: the exact client configuration, timeouts, and
    ``/generate`` URL the frontend always used (lazy client creation on the
    first request, like before)."""

    def __init__(
        self,
        base_url: str,
        *,
        max_connections: int = 100,
        pool_timeout_s: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_connections = max(100, max_connections)
        self.pool_timeout_s = pool_timeout_s
        self._http: httpx.AsyncClient | None = None

    async def generate(self, payload: dict) -> dict:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    10.0,
                    read=600.0,
                    write=60.0,
                    pool=self.pool_timeout_s,
                ),
                limits=httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=min(self.max_connections, 256),
                ),
            )
        response = await self._http.post(f"{self.base_url}/generate", json=payload)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
