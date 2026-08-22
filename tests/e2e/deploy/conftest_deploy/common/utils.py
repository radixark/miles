from collections.abc import Callable

from tests.e2e.ft.conftest_ft import comparisons
from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE
from tests.fast.cluster_backends import RUN_NAMESPACE_ENV_VAR, compute_missing_namespace_reason, create_backend_for_run

from miles.utils.external_utils import command_utils
from miles.utils.test_utils.comparisons.inference_engine_checksums import assert_engine_count
from miles.utils.workers.types import ClusterBackend


# ============================ the cluster to run on ===========================


def run_on_a_cluster(run_ci: Callable[[str | None], None]) -> Callable[[], None]:
    def run_ci_where_a_run_can_be_deployed() -> None:
        assert_the_cluster_can_deploy_runs(command_utils.default_config())
        run_ci(None)

    return run_ci_where_a_run_can_be_deployed


def assert_the_cluster_can_deploy_runs(config: command_utils.ExecuteTrainConfig) -> None:
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


# =========================== comparing the two sides ==========================


def compare_deterministic_sides(
    *,
    baseline_dir: str,
    target_dir: str,
    expected_engine_count: int,
    min_trained_rollouts: int,
    expected_metric_deltas: dict[str, list[float]] | None = None,
) -> None:
    comparisons.compare_deterministic_sides(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        min_trained_rollouts=min_trained_rollouts,
        expected_metric_deltas=expected_metric_deltas,
    )

    for side, side_dir in ((BASELINE_SIDE, baseline_dir), (TARGET_SIDE, target_dir)):
        assert_engine_count(side=side, dump_dir=side_dir, expected=expected_engine_count)
