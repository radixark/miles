"""Pin-and-verify a set of inference engines to an HF checkpoint snapshot.

Both the in-job dedicated eval fleet (Ray actors) and external eval backends
(bare sglang HTTP servers, e.g. ``examples/fully_async``) need the same
sequence: load a snapshot from disk, then confirm every target actually
reports the expected ``weight_version`` before trusting it for eval. This
module is the one place that sequence is implemented.
"""

import asyncio
import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class WeightTarget(Protocol):
    """Something that can load an HF snapshot from disk and report the
    weight_version it currently has loaded."""

    async def load_from_disk(self, hf_dir: str, weight_version: str) -> None: ...

    async def read_version(self) -> str | None: ...


class RayEngineTarget:
    """Adapts a Ray sglang engine actor handle to ``WeightTarget``."""

    def __init__(self, actor_handle):
        self._actor = actor_handle

    async def load_from_disk(self, hf_dir: str, weight_version: str) -> None:
        await self._actor.update_weights_from_disk.remote(hf_dir, weight_version=weight_version)

    async def read_version(self) -> str | None:
        return await self._actor.get_weight_version.remote()


class HttpServerTarget:
    """Adapts a bare sglang HTTP server to ``WeightTarget``."""

    def __init__(self, base_url: str):
        self._url = base_url

    async def load_from_disk(self, hf_dir: str, weight_version: str) -> None:
        from miles.utils.http_utils import post

        await post(f"{self._url}/update_weights_from_disk", {"model_path": hf_dir, "weight_version": weight_version})

    async def read_version(self) -> str | None:
        from miles.utils.http_utils import get

        info = await get(f"{self._url}/model_info")
        return info.get("weight_version")


async def pin_and_verify(
    targets: list[WeightTarget],
    hf_dir: str,
    weight_version: str,
    *,
    timeout: float = 600.0,
    retries: int = 2,
) -> bool:
    """Load ``hf_dir`` into every target and confirm all report ``weight_version``.

    Never raises: transient failures (timeout, RPC error, mismatched version)
    are retried up to ``retries`` times, then this returns ``False`` so the
    caller can decide what a failed pin means for it (skip a point, raise, ...).
    """
    versions: list = []
    for attempt in range(retries):
        try:
            await asyncio.wait_for(
                asyncio.gather(*[t.load_from_disk(hf_dir, weight_version) for t in targets]), timeout=timeout
            )
            versions = await asyncio.wait_for(asyncio.gather(*[t.read_version() for t in targets]), timeout=timeout)
        except Exception as e:
            logger.warning(f"Weight pin to {hf_dir} failed (attempt {attempt + 1}/{retries}): {e}")
            continue
        if versions and all(str(v) == weight_version for v in versions):
            return True
    logger.warning(f"Failed to pin weight_version={weight_version} to {hf_dir} (got {versions})")
    return False
