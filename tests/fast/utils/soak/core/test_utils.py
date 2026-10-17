from dataclasses import replace

from tests.utils.soak.core.utils import compute_release_of_config, create_soak_config

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.workers.types import ClusterBackend, DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.naming import CHART_NAME

_RUN_ID = "260926-120000-000"


class TestComputeReleaseOfConfig:
    def test_the_release_is_named_by_run_component_and_instance(self) -> None:
        """Cleanup targets the release this config launched and no other."""
        config = ExecuteTrainConfig(
            cluster_backend=ClusterBackend.KUBERNETES,
            namespace="rl",
            run_id=_RUN_ID,
            deploy_component=DeployComponent.INFERENCE,
            deploy_instance_id="b",
        )

        assert compute_release_of_config(config) == f"{CHART_NAME}-{_RUN_ID}-inference-b"
        assert compute_release_of_config(replace(config, deploy_instance_id="c")) != compute_release_of_config(config)


class TestCreateSoakConfig:
    def test_every_ray_soak_owns_a_fresh_submission_id(self) -> None:
        """Each Ray soak gets its own submission id so teardown only stops its own job."""
        config = ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY, run_id=_RUN_ID, ray_submission_id="theirs")

        first, second = create_soak_config(config), create_soak_config(config)

        assert first.ray_submission_id.startswith("miles-soak-")
        assert first.ray_submission_id != second.ray_submission_id
        assert replace(first, ray_submission_id=None) == replace(config, ray_submission_id=None)

    def test_a_kubernetes_config_is_returned_unchanged(self) -> None:
        """A Kubernetes soak is owned through its release, not a Ray submission."""
        config = ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, namespace="rl", run_id=_RUN_ID)

        assert create_soak_config(config) is config
