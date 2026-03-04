from __future__ import annotations

import time

import structlog
from kubernetes_asyncio import config as k8s_config
from kubernetes_asyncio.client import ApiClient, CoreV1Api

from miles.utils.ft.platform.protocols import NodeManagerProtocol

log = structlog.get_logger(__name__)

LABEL_KEY = "ft.miles/disabled"
REASON_LABEL_KEY = "ft.miles/disabled-reason"


class K8sNodeManager:
    """Manage bad-node labels on Kubernetes nodes via the K8s API.

    Implements NodeManagerProtocol using kubernetes_asyncio for async calls.
    """

    def __init__(self, api_client: ApiClient | None = None) -> None:
        self._external_client = api_client
        self._api_client: ApiClient | None = api_client

    async def _ensure_client(self) -> CoreV1Api:
        if self._api_client is None:
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                await k8s_config.load_kube_config()
            self._api_client = ApiClient()
        return CoreV1Api(self._api_client)

    async def mark_node_bad(self, node_id: str, reason: str) -> None:
        core_v1 = await self._ensure_client()
        body = {
            "metadata": {
                "labels": {
                    LABEL_KEY: "true",
                    REASON_LABEL_KEY: reason,
                }
            }
        }

        start = time.monotonic()
        await core_v1.patch_node(name=node_id, body=body)
        elapsed = time.monotonic() - start

        log.info(
            "mark_node_bad",
            node_id=node_id,
            reason=reason,
            elapsed_seconds=round(elapsed, 3),
        )

    async def unmark_node_bad(self, node_id: str) -> None:
        core_v1 = await self._ensure_client()
        body = {
            "metadata": {
                "labels": {
                    LABEL_KEY: None,
                    REASON_LABEL_KEY: None,
                }
            }
        }

        start = time.monotonic()
        await core_v1.patch_node(name=node_id, body=body)
        elapsed = time.monotonic() - start

        log.info(
            "unmark_node_bad",
            node_id=node_id,
            elapsed_seconds=round(elapsed, 3),
        )

    async def get_bad_nodes(self) -> list[str]:
        core_v1 = await self._ensure_client()

        start = time.monotonic()
        node_list = await core_v1.list_node(
            label_selector=f"{LABEL_KEY}=true",
        )
        elapsed = time.monotonic() - start

        names = [node.metadata.name for node in node_list.items]
        log.info(
            "get_bad_nodes",
            count=len(names),
            elapsed_seconds=round(elapsed, 3),
        )
        return names


def _check_protocol_conformance() -> None:
    _: type[NodeManagerProtocol] = K8sNodeManager  # type: ignore[assignment]
