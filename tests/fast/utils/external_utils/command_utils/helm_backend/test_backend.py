import pytest

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.backend import KubernetesCommandBackend


class TestApiServerHost:
    @pytest.mark.parametrize(
        ("run_id", "namespace"),
        [
            ("", "rl"),
            ("260101-000000-000", ""),
        ],
    )
    def test_an_api_server_host_requires_both_run_id_and_namespace(self, run_id: str, namespace: str) -> None:
        """An api server host requires both parts that identify its Kubernetes service."""
        config = ExecuteTrainConfig(run_id=run_id, namespace=namespace)
        backend = KubernetesCommandBackend(config)

        with pytest.raises(AssertionError, match="run_id and namespace"):
            backend.api_server_host(config)
