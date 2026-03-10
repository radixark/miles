from __future__ import annotations

import os

from miles.utils.ft.adapters.types import AgentMetadataProvider


class K8sMetadataProvider(AgentMetadataProvider):
    """Reads K8s node/pod names from Downward API environment variables."""

    def get_metadata(self) -> dict[str, str]:
        metadata: dict[str, str] = {}

        k8s_node_name = os.environ.get("K8S_NODE_NAME", "")
        if k8s_node_name:
            metadata["k8s_node_name"] = k8s_node_name

        k8s_pod_name = os.environ.get("K8S_POD_NAME", "")
        if k8s_pod_name:
            metadata["k8s_pod_name"] = k8s_pod_name

        return metadata
