from collections.abc import Callable

from tests.fast.cluster_backends import RUN_NAMESPACE_ENV_VAR, compute_missing_namespace_reason, create_backend_for_run

from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend


def run_on_cluster(run_ci: Callable[[str | None], None]) -> Callable[[], None]:
    def run_ci_on_deployable_cluster() -> None:
        assert_cluster_can_deploy_runs(command_utils.default_config())
        run_ci(None)

    return run_ci_on_deployable_cluster


def assert_cluster_can_deploy_runs(config: command_utils.ExecuteTrainConfig) -> None:
    assert (reason := _compute_unconfigured_reason(config)) is None, reason

    create_backend_for_run(config)


def _compute_unconfigured_reason(config: command_utils.ExecuteTrainConfig) -> str | None:
    if (backend := config.cluster_backend) is not ClusterBackend.KUBERNETES:
        return (
            f"these tests install a run as helm releases, which only the {ClusterBackend.KUBERNETES.value} "
            f"backend does, and this environment declares the {backend.value} backend"
        )
    if not config.namespace:
        return compute_missing_namespace_reason(RUN_NAMESPACE_ENV_VAR)
    return None
