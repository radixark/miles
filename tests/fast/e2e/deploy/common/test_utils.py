from typing import Any

import pytest
from tests.e2e.deploy.conftest_deploy.common import utils
from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.workers.types import ClusterBackend

_BASELINE_DIR: str = "/dumps/baseline"
_TARGET_DIR: str = "/dumps/target"
_MIN_TRAINED_ROLLOUTS: int = 2
_EXPECTED_ENGINE_COUNT: int = 2


def _config(*, cluster_backend: ClusterBackend, namespace: str) -> ExecuteTrainConfig:
    return ExecuteTrainConfig(cluster_backend=cluster_backend, namespace=namespace, run_id="demo")


class TestComputeUnconfiguredReason:
    def test_an_environment_on_another_backend_declares_no_kubernetes(self):
        """Only kubernetes installs one release per deployment, so any other backend is a declared absence."""
        reason = utils._compute_unconfigured_reason(_config(cluster_backend=ClusterBackend.RAY, namespace="rl"))

        assert reason is not None and ClusterBackend.RAY.value in reason

    def test_a_kubernetes_environment_without_a_namespace_is_unconfigured(self):
        """An empty namespace is how the environment says it configured no cluster of its own."""
        reason = utils._compute_unconfigured_reason(_config(cluster_backend=ClusterBackend.KUBERNETES, namespace=""))

        assert reason is not None and utils.RUN_NAMESPACE_ENV_VAR in reason

    def test_a_configured_kubernetes_environment_is_not_excused(self):
        """A declared namespace commits the environment to actually running these tests."""
        assert (
            utils._compute_unconfigured_reason(_config(cluster_backend=ClusterBackend.KUBERNETES, namespace="rl"))
            is None
        )


class TestAssertTheClusterCanDeployRuns:
    def test_a_non_kubernetes_backend_fails_rather_than_skips(self, monkeypatch):
        """A run that quietly does nothing reports green for a test that never installed anything."""
        monkeypatch.setattr(utils, "create_backend_for_run", _refuse_to_probe)

        with pytest.raises(AssertionError, match=ClusterBackend.KUBERNETES.value):
            utils.assert_the_cluster_can_deploy_runs(_config(cluster_backend=ClusterBackend.RAY, namespace="rl"))

    def test_an_environment_that_named_no_namespace_fails_rather_than_skips(self, monkeypatch):
        """The reason names the variable to set, which is the whole value of failing here rather than in helm."""
        monkeypatch.setattr(utils, "create_backend_for_run", _refuse_to_probe)

        with pytest.raises(AssertionError, match=utils.RUN_NAMESPACE_ENV_VAR):
            utils.assert_the_cluster_can_deploy_runs(_config(cluster_backend=ClusterBackend.KUBERNETES, namespace=""))

    def test_a_declared_cluster_that_cannot_be_reached_fails_rather_than_skips(self, monkeypatch):
        """Exiting 0 here would report green for a test that never installed anything."""
        monkeypatch.setattr(utils, "create_backend_for_run", _refuse_to_probe)

        with pytest.raises(AssertionError):
            utils.assert_the_cluster_can_deploy_runs(
                _config(cluster_backend=ClusterBackend.KUBERNETES, namespace="rl")
            )

    def test_a_reachable_cluster_lets_the_test_run(self, monkeypatch):
        """The check exists to fail unconfigured environments early, not to narrow configured ones."""
        monkeypatch.setattr(utils, "create_backend_for_run", lambda config: None)

        utils.assert_the_cluster_can_deploy_runs(_config(cluster_backend=ClusterBackend.KUBERNETES, namespace="rl"))

    def test_the_entry_wrapper_checks_the_cluster_before_it_runs_anything(self, monkeypatch):
        """Every deploy entry goes through this wrapper, so the check may not be skippable by forgetting it."""
        ran: list[str | None] = []
        monkeypatch.setattr(
            utils.command_utils,
            "default_config",
            lambda: _config(cluster_backend=ClusterBackend.KUBERNETES, namespace="rl"),
        )
        monkeypatch.setattr(utils, "assert_the_cluster_can_deploy_runs", _refuse_the_cluster)

        with pytest.raises(AssertionError, match="no cluster here"):
            utils.run_on_a_cluster(ran.append)()

        assert ran == []


def _refuse_to_probe(config: ExecuteTrainConfig) -> None:
    raise AssertionError(f"the {config.cluster_backend.value} backend is not reachable")


def _refuse_the_cluster(config: ExecuteTrainConfig) -> None:
    raise AssertionError("no cluster here")


@pytest.fixture
def recorded_calls(monkeypatch) -> dict[str, list[dict[str, Any]]]:
    calls: dict[str, list[dict[str, Any]]] = {"compare_deterministic_sides": [], "assert_engine_count": []}

    def record(name: str):
        def recorder(**kwargs: Any) -> None:
            calls[name].append(kwargs)

        return recorder

    monkeypatch.setattr(utils.comparisons, "compare_deterministic_sides", record("compare_deterministic_sides"))
    monkeypatch.setattr(utils, "assert_engine_count", record("assert_engine_count"))

    return calls


def _compare(*, expected_metric_deltas: dict[str, list[float]] | None = None) -> None:
    utils.compare_deterministic_sides(
        baseline_dir=_BASELINE_DIR,
        target_dir=_TARGET_DIR,
        expected_engine_count=_EXPECTED_ENGINE_COUNT,
        min_trained_rollouts=_MIN_TRAINED_ROLLOUTS,
        expected_metric_deltas=expected_metric_deltas,
    )


class TestCompareDeterministicSides:
    def test_the_two_sides_are_compared_exactly_as_an_unsplit_pair_is(self, recorded_calls):
        """A deploy comparison that dropped one of ft's checks would pass a split run ft would refuse."""
        _compare()

        assert recorded_calls["compare_deterministic_sides"] == [
            dict(
                baseline_dir=_BASELINE_DIR,
                target_dir=_TARGET_DIR,
                min_trained_rollouts=_MIN_TRAINED_ROLLOUTS,
                expected_metric_deltas=None,
            )
        ]

    def test_a_deploy_scenario_can_keep_a_proven_nonzero_metric_delta_in_the_exact_comparison(self, recorded_calls):
        """The deploy wrapper forwards semantic counter deltas instead of omitting those metrics."""
        expected = {"rollout/weight_version/max": [2.0]}

        _compare(expected_metric_deltas=expected)

        assert recorded_calls["compare_deterministic_sides"][0]["expected_metric_deltas"] == expected

    def test_both_sides_are_required_to_have_served_the_engines_the_run_declared(self, recorded_calls):
        """This is what the unsplit comparison cannot check: a split run losing an engine deployment."""
        _compare()

        assert recorded_calls["assert_engine_count"] == [
            dict(side=BASELINE_SIDE, dump_dir=_BASELINE_DIR, expected=_EXPECTED_ENGINE_COUNT),
            dict(side=TARGET_SIDE, dump_dir=_TARGET_DIR, expected=_EXPECTED_ENGINE_COUNT),
        ]
