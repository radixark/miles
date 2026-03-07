from __future__ import annotations

import logging
import time

from kubernetes_asyncio import config as k8s_config
from kubernetes_asyncio.client import ApiClient, CoreV1Api

from miles.utils.ft.protocols.platform import NodeManagerProtocol
from miles.utils.ft.utils.retry import retry_async_or_raise

logger = logging.getLogger(__name__)

_LABEL_DOMAIN = "ft.miles.io"
_BASE_DISABLED = "disabled"
_BASE_REASON = "disabled-reason"

_BASE_LABEL_KEY = f"{_LABEL_DOMAIN}/{_BASE_DISABLED}"
_BASE_REASON_LABEL_KEY = f"{_LABEL_DOMAIN}/{_BASE_REASON}"

_K8S_API_TIMEOUT_SECONDS = 30
_K8S_API_MAX_RETRIES = 3
_K8S_API_BACKOFF_BASE = 1.0

LABEL_KEY = _BASE_LABEL_KEY
REASON_LABEL_KEY = _BASE_REASON_LABEL_KEY


class K8sNodeManager(NodeManagerProtocol):
    """Manage bad-node labels on Kubernetes nodes via the K8s API.

    Implements NodeManagerProtocol using kubernetes_asyncio for async calls.
    """

    def __init__(
        self,
        api_client: ApiClient | None = None,
        label_prefix: str = "",
    ) -> None:
        self._api_client: ApiClient | None = api_client
        self._core_v1: CoreV1Api | None = None
        self._label_key, self._reason_label_key = _build_label_keys(label_prefix)

    async def mark_node_bad(self, node_id: str, reason: str) -> None:
        elapsed = await self._patch_node_labels(
            node_id=node_id,
            labels={self._label_key: "true", self._reason_label_key: reason},
        )
        logger.info(
            "mark_node_bad node_id=%s reason=%s elapsed_seconds=%.3f",
            node_id, reason, elapsed,
        )

    async def unmark_node_bad(self, node_id: str) -> None:
        elapsed = await self._patch_node_labels(
            node_id=node_id,
            labels={self._label_key: None, self._reason_label_key: None},
        )
        logger.info(
            "unmark_node_bad node_id=%s elapsed_seconds=%.3f",
            node_id, elapsed,
        )

    async def aclose(self) -> None:
        if self._api_client is not None:
            await self._api_client.close()
            self._api_client = None
            self._core_v1 = None

    async def get_bad_nodes(self) -> list[str]:
        core_v1 = await self._ensure_client()

        start = time.monotonic()
        node_list = await retry_async_or_raise(
            lambda: core_v1.list_node(label_selector=f"{self._label_key}=true"),
            description="list_node(bad)",
            max_retries=_K8S_API_MAX_RETRIES,
            per_call_timeout=_K8S_API_TIMEOUT_SECONDS,
            backoff_base=_K8S_API_BACKOFF_BASE,
        )
        elapsed = time.monotonic() - start

        names = [node.metadata.name for node in node_list.items]
        logger.info(
            "get_bad_nodes count=%d elapsed_seconds=%.3f",
            len(names), elapsed,
        )
        return names

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

    async def _patch_node_labels(
        self,
        node_id: str,
        labels: dict[str, str | None],
    ) -> float:
        core_v1 = await self._ensure_client()
        body = {"metadata": {"labels": labels}}

        start = time.monotonic()
        await retry_async_or_raise(
            lambda: core_v1.patch_node(name=node_id, body=body),
            description=f"patch_node({node_id})",
            max_retries=_K8S_API_MAX_RETRIES,
            per_call_timeout=_K8S_API_TIMEOUT_SECONDS,
            backoff_base=_K8S_API_BACKOFF_BASE,
        )
        return time.monotonic() - start


def _build_label_keys(label_prefix: str) -> tuple[str, str]:
    if label_prefix:
        return (
            f"{_LABEL_DOMAIN}/{label_prefix}-{_BASE_DISABLED}",
            f"{_LABEL_DOMAIN}/{label_prefix}-{_BASE_REASON}",
        )
    return _BASE_LABEL_KEY, _BASE_REASON_LABEL_KEY
