import pytest

from miles.utils.external_utils.command_utils.api_server import api_server_url
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.workers.types import ClusterBackend

PORT = 18080


class TestATestFindsTheApiServerOfTheRunItLaunched:
    def test_a_ray_run_answers_where_it_always_did(self):
        """The orchestrator is the launching process itself, so the fault injector keeps talking to localhost."""
        config = ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY.value)

        assert api_server_url(config, port=PORT) == f"http://localhost:{PORT}"

    def test_a_kubernetes_run_answers_on_its_orchestrator_service(self):
        """The orchestrator is a pod now, and localhost there is whichever pod asked."""
        config = ExecuteTrainConfig(
            cluster_backend=ClusterBackend.KUBERNETES.value, run_id="soak-1", namespace="miles-e2e"
        )

        url = api_server_url(config, port=PORT)

        assert url == f"http://miles-run-soak-1-orchestrator.miles-e2e.svc.cluster.local:{PORT}"

    @pytest.mark.parametrize("missing", ["run_id", "namespace"])
    def test_a_kubernetes_run_without_an_identity_is_refused(self, missing: str):
        """Guessing would poll a url nobody serves and report a fault injector that never injected."""
        fields = {"run_id": "soak-1", "namespace": "miles-e2e", missing: ""}
        config = ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES.value, **fields)

        with pytest.raises(AssertionError, match="run_id"):
            api_server_url(config, port=PORT)
