import asyncio
import logging
from typing import Protocol

import httpx

from miles.utils.http_utils import router_worker_base_urls

logger = logging.getLogger(__name__)


class InferenceAdminPort(Protocol):
    async def init(self) -> None:
        """Open the transport. The backend's lifecycle calls this — it is
        part of the declared contract, so a fake implementing the port never
        surprises the backend with an AttributeError."""
        ...

    async def close(self) -> None:
        """Release the transport (idempotent)."""
        ...

    async def abort_registration(self, rid_prefix: str) -> None:
        """Abort every in-flight engine request whose rid carries this
        registration's prefix (anti-ABA: the prefix embeds the registration
        id, so a retiring tenant can never abort a same-name successor)."""
        ...


class RouterInferenceAdmin:
    """Current adapter: worker discovery via the router's
    ``/list_workers``|``/workers`` and per-worker ``/abort_request`` posts."""

    def __init__(self, router_url: str) -> None:
        self.router_url = router_url.rstrip("/")
        self.client: httpx.AsyncClient | None = None

    async def init(self) -> None:
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def worker_urls(self) -> list[str]:
        assert self.client is not None
        for endpoint, extract in (
            ("/list_workers", lambda body: body["urls"]),
            ("/workers", lambda body: [worker["url"] for worker in body["workers"]]),
        ):
            try:
                resp = await self.client.get(f"{self.router_url}{endpoint}")
                if resp.status_code == 200:
                    return router_worker_base_urls(extract(resp.json()))
            except Exception:
                continue
        return []

    async def abort_registration(self, rid_prefix: str) -> None:
        urls = await self.worker_urls()
        if not urls:
            logger.warning(f"[tinker] abort for '{rid_prefix}': no workers discovered at {self.router_url}")
            return
        results = await asyncio.gather(
            *(self.client.post(f"{url}/abort_request", json={"rid": rid_prefix, "prefix": True}) for url in urls),
            return_exceptions=True,
        )
        if failures := sum(isinstance(r, Exception) for r in results):
            logger.warning(f"[tinker] abort for '{rid_prefix}': {failures}/{len(results)} posts failed")
