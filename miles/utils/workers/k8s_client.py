from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kubernetes_asyncio.client import CoreV1Api


@asynccontextmanager
async def core_v1_api() -> AsyncIterator[CoreV1Api]:
    from kubernetes_asyncio import client, config

    try:
        config.load_incluster_config()
    except config.ConfigException:
        await config.load_kube_config()
    async with client.ApiClient() as api_client:
        yield client.CoreV1Api(api_client)
