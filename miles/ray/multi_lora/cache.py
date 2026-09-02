"""Cached resident-adapter projection for rollout request routing."""

import time

from miles.utils.misc import SingletonMeta


class AdaptersCache(metaclass=SingletonMeta):
    """TTL-cache the controller's ready and retiring adapter registrations."""

    def __init__(self, ttl_s: float = 1.0) -> None:
        self.ttl_s = ttl_s
        self.snapshot: dict = {"pending": {}, "ready": {}, "retiring": {}, "cleanup": []}
        self.last_refresh: float | None = None

    async def get_snapshot(self) -> dict:
        from miles.ray.multi_lora.controller import get_multi_lora_controller

        now = time.monotonic()
        if self.last_refresh is None or now - self.last_refresh >= self.ttl_s:
            try:
                self.snapshot = await get_multi_lora_controller().snapshot.remote()
                self.last_refresh = now
            except Exception:
                pass
        return self.snapshot

    async def get_all(self) -> dict:
        snapshot = await self.get_snapshot()
        return {**snapshot.get("ready", {}), **snapshot.get("retiring", {})}

    async def get(self, adapter_name: str):
        return (await self.get_all()).get(adapter_name)
