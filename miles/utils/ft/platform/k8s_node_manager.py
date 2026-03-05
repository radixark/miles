from __future__ import annotations

import logging
import time

from kubernetes_asyncio import config as k8s_config
from kubernetes_asyncio.client import ApiClient, CoreV1Api

logger = logging.getLogger(__name__)

LABEL_KEY = "ft.miles/disabled"
REASON_LABEL_KEY = "ft.miles/disabled-reason"


class K8sNodeManager:
    """Manage bad-node labels on Kubernetes nodes via the K8s API.

    Implements NodeManagerProtocol using kubernetes_asyncio for async calls.
    """

    def __init__(self, api_client: ApiClient | None = None) -> None:
        self._api_client: ApiClient | None = api_client
        self._core_v1: CoreV1Api | None = None

    async def _ensure_client(self) -> CoreV1Api:
        if self._core_v1 is not None:
            return self._core_v1

        if self._api_client is None:
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                await k8s_config.load_kube_config()

            self._api_client = ApiClient()

        self._core_v1 = CoreV1Api(self._api_client)
        return self._core_v1

    async def mark_node_bad(self, node_id: str, reason: str) -> None:
        elapsed = await self._patch_node_labels(
            node_id=node_id,
            labels={LABEL_KEY: "true", REASON_LABEL_KEY: reason},
        )
        logger.info(
            "mark_node_bad node_id=%s reason=%s elapsed_seconds=%.3f",
            node_id, reason, elapsed,
        )

    async def unmark_node_bad(self, node_id: str) -> None:
        elapsed = await self._patch_node_labels(
            node_id=node_id,
            labels={LABEL_KEY: None, REASON_LABEL_KEY: None},
        )
        logger.info(
            "unmark_node_bad node_id=%s elapsed_seconds=%.3f",
            node_id, elapsed,
        )

    async def _patch_node_labels(
        self,
        node_id: str,
        labels: dict[str, str | None],
    ) -> float:
        core_v1 = await self._ensure_client()
        body = {"metadata": {"labels": labels}}

        start = time.monotonic()
        await core_v1.patch_node(name=node_id, body=body)
        return time.monotonic() - start

    async def aclose(self) -> None:
        if self._api_client is not None:
            await self._api_client.close()
            self._api_client = None
            self._core_v1 = None

    async def get_bad_nodes(self) -> list[str]:
        core_v1 = await self._ensure_client()

        start = time.monotonic()
        node_list = await core_v1.list_node(
            label_selector=f"{LABEL_KEY}=true",
        )
        elapsed = time.monotonic() - start

        names = [node.metadata.name for node in node_list.items]
        logger.info(
            "get_bad_nodes count=%d elapsed_seconds=%.3f",
            len(names), elapsed,
        )
        return names


def query_bad_nodes() -> list[str] | None:
    """Synchronous helper: query K8s for bad-node names.

    Returns a list of node names on success, or None if the query failed.
    Raises RuntimeError if called from within an async context (running event
    loop) since asyncio.run() cannot be nested.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "query_bad_nodes() cannot be called from within a running event loop; "
            "use 'await K8sNodeManager().get_bad_nodes()' instead"
        )

    async def _query() -> list[str]:
        manager = K8sNodeManager()
        try:
            return await manager.get_bad_nodes()
        finally:
            await manager.aclose()

    try:
        return asyncio.run(_query())
    except Exception:
        logger.warning("Failed to query K8s bad nodes", exc_info=True)
        return None
