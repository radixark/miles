from __future__ import annotations

import os

from miles.utils.ft.adapters.types import AgentMetadataProvider


class K8sMetadataProvider(AgentMetadataProvider):
    """Reads K8s node/pod names from Downward API environment variables."""

    def get_metadata(self) -> dict[str, str]:
        node_name = os.environ.get("K8S_NODE_NAME", "")
        pod_name = os.environ.get("K8S_POD_NAME", "")

        if not node_name:
            raise RuntimeError(
                "K8S_NODE_NAME env var not set. "
                "Configure Kubernetes Downward API in pod spec."
            )
        if not pod_name:
            raise RuntimeError(
                "K8S_POD_NAME env var not set. "
                "Configure Kubernetes Downward API in pod spec."
            )

        return {
            "k8s_node_name": node_name,
            "k8s_pod_name": pod_name,
        }
