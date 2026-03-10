from __future__ import annotations

from miles.utils.ft.adapters.types import AgentMetadataProvider
from miles.utils.ft.utils.env import get_k8s_node_name, get_k8s_pod_name


class K8sMetadataProvider(AgentMetadataProvider):
    """Reads K8s node/pod names from Downward API environment variables."""

    def get_metadata(self) -> dict[str, str]:
        metadata: dict[str, str] = {}

        k8s_node_name = get_k8s_node_name()
        if k8s_node_name:
            metadata["k8s_node_name"] = k8s_node_name

        k8s_pod_name = get_k8s_pod_name()
        if k8s_pod_name:
            metadata["k8s_pod_name"] = k8s_pod_name

        return metadata
