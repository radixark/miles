from __future__ import annotations

from unittest.mock import patch

from miles.utils.ft.adapters.impl.k8s_metadata_provider import K8sMetadataProvider


class TestK8sMetadataProvider:
    def test_returns_both_env_vars_when_set(self) -> None:
        with patch.dict("os.environ", {"K8S_NODE_NAME": "gke-node-01", "K8S_POD_NAME": "trainer-pod-abc"}):
            provider = K8sMetadataProvider()
            metadata = provider.get_metadata()

        assert metadata == {"k8s_node_name": "gke-node-01", "k8s_pod_name": "trainer-pod-abc"}

    def test_returns_empty_dict_when_no_env_vars(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            provider = K8sMetadataProvider()
            metadata = provider.get_metadata()

        assert metadata == {}

    def test_returns_only_node_name_when_pod_unset(self) -> None:
        with patch.dict("os.environ", {"K8S_NODE_NAME": "gke-node-01"}, clear=True):
            provider = K8sMetadataProvider()
            metadata = provider.get_metadata()

        assert metadata == {"k8s_node_name": "gke-node-01"}

    def test_returns_only_pod_name_when_node_unset(self) -> None:
        with patch.dict("os.environ", {"K8S_POD_NAME": "trainer-pod-abc"}, clear=True):
            provider = K8sMetadataProvider()
            metadata = provider.get_metadata()

        assert metadata == {"k8s_pod_name": "trainer-pod-abc"}

    def test_skips_empty_string_values(self) -> None:
        with patch.dict("os.environ", {"K8S_NODE_NAME": "", "K8S_POD_NAME": ""}):
            provider = K8sMetadataProvider()
            metadata = provider.get_metadata()

        assert metadata == {}

    def test_reads_fresh_env_on_each_call(self) -> None:
        """Metadata is not cached — each call reads current env vars."""
        provider = K8sMetadataProvider()

        with patch.dict("os.environ", {"K8S_NODE_NAME": "node-1"}, clear=True):
            first = provider.get_metadata()

        with patch.dict("os.environ", {"K8S_NODE_NAME": "node-2"}, clear=True):
            second = provider.get_metadata()

        assert first == {"k8s_node_name": "node-1"}
        assert second == {"k8s_node_name": "node-2"}
