from __future__ import annotations

import logging
import os
import time

from kubernetes_asyncio import config as k8s_config
from kubernetes_asyncio.client import ApiClient, CoreV1Api

from miles.utils.ft.adapters.types import NodeManagerProtocol
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
    Auto-detects ray_cluster_name from the current pod's labels on first use,
    then deletes the Ray worker pod on the labeled node so the RayCluster
    operator reschedules it.
    """

    def __init__(
        self,
        api_client: ApiClient | None = None,
        label_prefix: str = "",
        namespace: str = "",
    ) -> None:
        self._api_client: ApiClient | None = api_client
        self._core_v1: CoreV1Api | None = None
        self._label_key, self._reason_label_key = _build_label_keys(label_prefix)
        self._ray_cluster_name: str = ""
        self._namespace = namespace
        self._reverse_map: dict[str, str] = {}
        self._affinity_validated: bool = False

    async def mark_node_bad(self, node_id: str, reason: str, node_metadata: dict[str, str] | None = None) -> None:
        await self._ensure_ray_cluster_name()

        if not self._affinity_validated:
            await self.assert_worker_node_affinity()
            self._affinity_validated = True

        k8s_name = node_id
        if node_metadata and "k8s_node_name" in node_metadata:
            k8s_name = node_metadata["k8s_node_name"]
            self._reverse_map[k8s_name] = node_id

        elapsed = await self._patch_node_labels(
            node_id=k8s_name,
            labels={self._label_key: "true", self._reason_label_key: reason},
        )
        logger.info(
            "mark_node_bad node_id=%s k8s_name=%s reason=%s elapsed_seconds=%.3f",
            node_id,
            k8s_name,
            reason,
            elapsed,
        )

        try:
            await self._delete_ray_worker_pod_on_node(k8s_name)
        except Exception:
            logger.error(
                "failed to delete ray worker pod on node %s, " "pod will remain until RayCluster operator handles it",
                k8s_name,
                exc_info=True,
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

        k8s_names = [node.metadata.name for node in node_list.items]

        result: list[str] = []
        for k8s_name in k8s_names:
            if k8s_name in self._reverse_map:
                result.append(self._reverse_map[k8s_name])
            else:
                logger.debug("get_bad_nodes: no reverse mapping for k8s_name=%s, skipping", k8s_name)

        logger.info(
            "get_bad_nodes k8s_count=%d mapped_count=%d elapsed_seconds=%.3f",
            len(k8s_names),
            len(result),
            elapsed,
        )
        return result

    async def assert_worker_node_affinity(self) -> None:
        """Validate that ray worker pods have a nodeAffinity NotIn rule for the label key.

        Raises RuntimeError if no worker pods are found or if the expected
        anti-affinity rule is missing.
        """
        core_v1 = await self._ensure_client()

        label_selector = f"ray.io/cluster={self._ray_cluster_name},ray.io/node-type=worker"
        pod_list = await retry_async_or_raise(
            lambda: core_v1.list_namespaced_pod(
                namespace=self._namespace,
                label_selector=label_selector,
            ),
            description="list_worker_pods(affinity_check)",
            max_retries=_K8S_API_MAX_RETRIES,
            per_call_timeout=_K8S_API_TIMEOUT_SECONDS,
            backoff_base=_K8S_API_BACKOFF_BASE,
        )

        if not pod_list.items:
            raise RuntimeError("no ray worker pods found")

        pod = pod_list.items[0]
        affinity = getattr(pod.spec, "affinity", None)
        if affinity is None:
            raise RuntimeError(f"worker pod missing nodeAffinity NotIn rule for {self._label_key}")

        node_affinity = getattr(affinity, "node_affinity", None)
        required = (
            getattr(node_affinity, "required_during_scheduling_ignored_during_execution", None)
            if node_affinity
            else None
        )
        terms = getattr(required, "node_selector_terms", None) if required else None

        if terms:
            for term in terms:
                for expr in getattr(term, "match_expressions", []) or []:
                    if (
                        getattr(expr, "key", None) == self._label_key
                        and getattr(expr, "operator", None) == "NotIn"
                        and "true" in (getattr(expr, "values", []) or [])
                    ):
                        logger.info(
                            "assert_worker_node_affinity passed label_key=%s pod=%s",
                            self._label_key,
                            pod.metadata.name,
                        )
                        return

        raise RuntimeError(f"worker pod missing nodeAffinity NotIn rule for {self._label_key}")

    async def _ensure_ray_cluster_name(self) -> str:
        if self._ray_cluster_name:
            return self._ray_cluster_name

        pod_name = os.environ.get("K8S_POD_NAME", "")
        if not pod_name:
            raise RuntimeError("K8S_POD_NAME env var not set. " "Configure Kubernetes Downward API in pod spec.")

        core_v1 = await self._ensure_client()
        pod = await retry_async_or_raise(
            lambda: core_v1.read_namespaced_pod(
                name=pod_name,
                namespace=self._namespace,
            ),
            description=f"read_pod({pod_name})",
            max_retries=_K8S_API_MAX_RETRIES,
            per_call_timeout=_K8S_API_TIMEOUT_SECONDS,
            backoff_base=_K8S_API_BACKOFF_BASE,
        )
        labels = pod.metadata.labels or {}
        cluster_name = labels.get("ray.io/cluster", "")
        if not cluster_name:
            raise RuntimeError(f"Pod {pod_name} missing ray.io/cluster label. " "Not running in a RayCluster?")

        self._ray_cluster_name = cluster_name
        logger.info(
            "auto_detected ray_cluster_name=%s from pod=%s",
            cluster_name,
            pod_name,
        )
        return cluster_name

    async def _delete_ray_worker_pod_on_node(self, node_id: str) -> None:
        core_v1 = await self._ensure_client()

        label_selector = f"ray.io/cluster={self._ray_cluster_name},ray.io/node-type=worker"
        field_selector = f"spec.nodeName={node_id}"

        start = time.monotonic()
        pod_list = await retry_async_or_raise(
            lambda: core_v1.list_namespaced_pod(
                namespace=self._namespace,
                label_selector=label_selector,
                field_selector=field_selector,
            ),
            description=f"list_pods_on_node({node_id})",
            max_retries=_K8S_API_MAX_RETRIES,
            per_call_timeout=_K8S_API_TIMEOUT_SECONDS,
            backoff_base=_K8S_API_BACKOFF_BASE,
        )

        for pod in pod_list.items:
            pod_name = pod.metadata.name
            await retry_async_or_raise(
                lambda name=pod_name: core_v1.delete_namespaced_pod(
                    name=name,
                    namespace=self._namespace,
                ),
                description=f"delete_pod({pod_name})",
                max_retries=_K8S_API_MAX_RETRIES,
                per_call_timeout=_K8S_API_TIMEOUT_SECONDS,
                backoff_base=_K8S_API_BACKOFF_BASE,
            )
            logger.info("deleted_ray_worker_pod pod=%s node=%s", pod_name, node_id)

        elapsed = time.monotonic() - start
        logger.info(
            "delete_ray_worker_pods_on_node node_id=%s count=%d elapsed_seconds=%.3f",
            node_id,
            len(pod_list.items),
            elapsed,
        )

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
