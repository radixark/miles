import pytest
from examples.infra_features.split_deployment.address_book import DEFAULT_TRAINER_ID
from tests.e2e.deploy.conftest_deploy.hot_restart import scenario_hot_restart_realistic_gsm8k as scenario
from tests.fast.e2e.scenario_harness import SCENARIO_RUN_ID, ScenarioHarness, parse_fault_tolerance_args
from tests.utils.deploy.hot_restart.evidence import evidence_path
from tests.utils.soak.core.config import SoakTailConfig, SoakTargetConfig
from tests.utils.soak.core.events import LaunchOutcome
from tests.utils.soak.core.utils import compute_release_of_config
from tests.utils.soak.deploy.actions.hot_restart import HotRestartForm
from tests.utils.soak.deploy.checkers.checkpoint_progress import MIN_HOT_RESTARTS, SAVE_INTERVAL
from tests.utils.soak.deploy.observers import DeploymentObserver
from tests.utils.soak.deploy.types import DEPLOYMENT_TARGET_KIND
from tests.utils.soak.deploy.utils import compute_checkpoint_dir
from tests.utils.soak.recipes import gsm8k

from miles.utils.workers.types import ClusterBackend

_NAMESPACE = "rl"
_CHECKERS: tuple[str, ...] = (
    "assert_hot_restart_launches_finished",
    "assert_checkpoints_advanced_between_takeovers",
    "assert_take_overs_replaced_only_script",
    "assert_take_over_loss_within_save_interval",
    "assert_publications_after_take_overs",
    "assert_take_overs_resumed_within_save_interval",
)


@pytest.fixture
def kubernetes_harness(scenario_harness: ScenarioHarness, monkeypatch: pytest.MonkeyPatch) -> ScenarioHarness:
    monkeypatch.setenv("MILES_SCRIPT_CLUSTER_BACKEND", ClusterBackend.KUBERNETES.value)
    monkeypatch.setenv("MILES_SCRIPT_NAMESPACE", _NAMESPACE)
    monkeypatch.setattr(gsm8k, "create_backend_for_run", lambda config: config.create_backend())
    monkeypatch.setattr(gsm8k, "prepare_gsm8k", scenario_harness.record_backend_prepare)
    return scenario_harness


@pytest.fixture
def harness(kubernetes_harness: ScenarioHarness, monkeypatch: pytest.MonkeyPatch) -> ScenarioHarness:
    for name in _CHECKERS:
        monkeypatch.setattr(scenario, name, kubernetes_harness.recorder(name))
    monkeypatch.setattr(
        scenario,
        "project_hot_restart_evidence",
        kubernetes_harness.spy("project_hot_restart_evidence", scenario.project_hot_restart_evidence),
    )
    return kubernetes_harness


class TestWhereAHotRestartSoakMayRun:
    def test_a_ray_run_is_refused_before_anything_is_prepared(self, scenario_harness: ScenarioHarness) -> None:
        """A take-over replaces a helm release, so on ray it would have nothing to replace."""
        with pytest.raises(AssertionError, match="Hot restart needs Kubernetes"):
            scenario.run_ci(seed=5, num_rollout=40, hot_restart_interval_seconds=17.0)

        assert scenario_harness.prepared == []
        assert scenario_harness.launches == []

    def test_a_kubernetes_run_without_a_namespace_is_refused(
        self, scenario_harness: ScenarioHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a namespace the observer and the take-over would address no release at all."""
        monkeypatch.setenv("MILES_SCRIPT_CLUSTER_BACKEND", ClusterBackend.KUBERNETES.value)

        with pytest.raises(AssertionError, match="Hot restart needs Kubernetes"):
            scenario.run_ci(seed=5, num_rollout=40, hot_restart_interval_seconds=17.0)

        assert scenario_harness.launches == []


class TestTheSoakOfOneDeployment:
    def test_the_one_deployment_is_soaked_at_the_hot_restart_cadence(self, harness: ScenarioHarness) -> None:
        """More than one expected target, or another cadence, would misjudge the single release the run has."""
        scenario.run_ci(seed=5, num_rollout=40, hot_restart_interval_seconds=17.0)

        (soak,) = harness.soaks
        runner_config = soak["runner_config"]
        assert runner_config.seed == 5
        assert runner_config.target_configs == {
            DEPLOYMENT_TARGET_KIND: SoakTargetConfig(expected_count=1, mean_interval_seconds=17.0)
        }
        assert runner_config.tail == SoakTailConfig.create(num_rollout=40)

    def test_the_take_over_form_relaunches_the_spec_and_log_of_this_run(self, harness: ScenarioHarness) -> None:
        """A take-over launching another spec, or logging elsewhere, would replace a run the soak never watched."""
        scenario.run_ci(seed=5, num_rollout=40, hot_restart_interval_seconds=17.0)

        (soak,) = harness.soaks
        (launch,) = harness.launches
        (form,) = soak["forms"][DEPLOYMENT_TARGET_KIND]
        assert isinstance(form, HotRestartForm)
        assert form.event_log is soak["event_log"]
        assert launch.config is soak["config"]
        assert form.launch_spec.config == soak["config"]
        assert (
            f"{form.launch_spec.train_args} --deploy-component {launch.config.deploy_component.value}"
            == launch.request.train_args
        )

    def test_the_observer_watches_the_release_checkpoints_and_events_of_this_run(
        self, harness: ScenarioHarness
    ) -> None:
        """Observing another release or directory would draw take-overs from progress this run never made."""
        scenario.run_ci(seed=5, num_rollout=40, hot_restart_interval_seconds=17.0)

        (soak,) = harness.soaks
        observer = soak["observer"]
        dump_dir = harness.dumps_root / SCENARIO_RUN_ID / "hot_restart_realistic_gsm8k"
        assert soak["dump_dir"] == dump_dir
        assert isinstance(observer, DeploymentObserver)
        assert observer.namespace == _NAMESPACE
        assert observer.release == compute_release_of_config(soak["config"])
        assert observer.trainer_id == DEFAULT_TRAINER_ID
        assert observer.checkpoint_dir == compute_checkpoint_dir(str(dump_dir))
        assert observer.events_dir == dump_dir / "events"


class TestTheLaunchATakeOverResumes:
    def test_the_first_launch_is_unguarded_on_kubernetes_under_the_run_config(self, harness: ScenarioHarness) -> None:
        """Only a take-over has an observed generation to guard, so the first install must not be guarded."""
        scenario.run_ci(seed=5, num_rollout=40, hot_restart_interval_seconds=17.0)

        (launch,) = harness.launches
        (prepared,) = harness.prepared
        assert launch.guard is None
        assert launch.config.cluster_backend is ClusterBackend.KUBERNETES
        assert prepared["config"] == launch.config

    def test_the_run_saves_and_resumes_from_its_own_checkpoint_directory(self, harness: ScenarioHarness) -> None:
        """A take-over resumes only from what the run saved, at the interval the loss bound assumes."""
        scenario.run_ci(seed=5, num_rollout=40, hot_restart_interval_seconds=17.0)

        (launch,) = harness.launches
        (soak,) = harness.soaks
        checkpoint_dir = str(compute_checkpoint_dir(str(soak["dump_dir"])))
        assert launch.value_of("--save") == checkpoint_dir
        assert launch.value_of("--load") == checkpoint_dir
        assert launch.value_of("--save-interval") == str(SAVE_INTERVAL)

    def test_the_run_keeps_one_wandb_run_and_publishes_checksums_without_training_ft(
        self, harness: ScenarioHarness
    ) -> None:
        """A take-over is not a cell fault, so cell ft stays off while the publication checker needs checksums."""
        scenario.run_ci(seed=5, num_rollout=40, hot_restart_interval_seconds=17.0)

        (launch,) = harness.launches
        parsed = parse_fault_tolerance_args(launch.request.train_args)
        assert launch.value_of("--wandb-run-id") == SCENARIO_RUN_ID
        assert "--save-inference-engine-weight-checksum" in launch.argv
        assert "--ci-disable-weight-update-checker" in launch.argv
        assert parsed.ft_components == []
        assert not parsed.mini_ft_controller_enable
        assert not parsed.namespace.colocate


class TestWhatAHotRestartSoakIsJudgedBy:
    def test_every_take_over_checker_reads_the_events_and_evidence_of_this_run(self, harness: ScenarioHarness) -> None:
        """Projecting another release or reading other events would pass take-overs this run never had."""
        scenario.run_ci(seed=5, num_rollout=40, hot_restart_interval_seconds=17.0)

        (soak,) = harness.soaks
        events = soak["event_log"].events
        assert [name for name in harness.checker_names if name in _CHECKERS] == list(_CHECKERS)
        ((project_args, project_kwargs),) = harness.calls_of("project_hot_restart_evidence")
        assert project_args == (events,)
        assert project_kwargs == {"release": compute_release_of_config(soak["config"])}
        ((_, scope_kwargs),) = harness.calls_of("assert_take_overs_replaced_only_script")
        assert scope_kwargs == {"num_restarts": 0, "minimum_restarts": MIN_HOT_RESTARTS}
        ((_, publication_kwargs),) = harness.calls_of("assert_publications_after_take_overs")
        assert publication_kwargs["source"] == soak["dump_dir"] / "events"
        ((resumed_args, _),) = harness.calls_of("assert_take_overs_resumed_within_save_interval")
        assert resumed_args == (str(soak["dump_dir"]),)

    def test_the_projected_evidence_is_kept_beside_the_soak_evidence(self, harness: ScenarioHarness) -> None:
        """The evidence of the take-overs is what a failed run is diagnosed from, so it must be archived."""
        scenario.run_ci(seed=5, num_rollout=40, hot_restart_interval_seconds=17.0)

        (soak,) = harness.soaks
        assert evidence_path(dump_dir=str(soak["evidence_dir"])).is_file()

    def test_a_failed_launch_is_raised_recorded_and_never_graded(self, harness: ScenarioHarness) -> None:
        """A run whose first install failed has no take-overs to judge and must fail as a launch failure."""
        harness.launch_error = RuntimeError("helm upgrade failed")

        with pytest.raises(RuntimeError, match="helm upgrade failed"):
            scenario.run_ci(seed=5, num_rollout=40, hot_restart_interval_seconds=17.0)

        (soak,) = harness.soaks
        (event,) = soak["event_log"].events
        assert event.outcome is LaunchOutcome.FAILED
        assert harness.checks == []

    def test_a_run_nothing_took_over_fails_the_real_checkers(self, kubernetes_harness: ScenarioHarness) -> None:
        """A soak that never replaced the script ran an ordinary training, and passing it would prove nothing."""
        with pytest.raises(AssertionError, match=f"Expected at least {MIN_HOT_RESTARTS} applied takeovers, got 0"):
            scenario.run_ci(seed=5, num_rollout=40, hot_restart_interval_seconds=17.0)
