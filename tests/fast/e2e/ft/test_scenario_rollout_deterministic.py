import json
from pathlib import Path

import pytest
from tests.e2e.ft.conftest_ft import app as comparison_app
from tests.e2e.ft.conftest_ft import scenario_rollout_deterministic
from tests.e2e.ft.conftest_ft.modes import MODES
from tests.fast.e2e.scenario_harness import SCENARIO_RUN_ID, ScenarioHarness, parse_fault_tolerance_args
from tests.utils.ft.launch import DETERMINISTIC_ENV_VARS
from tests.utils.soak.core.config import QUIESCENT_POLLS_REQUIRED, SoakTailConfig, SoakTargetConfig
from tests.utils.soak.ft.actions.factory import create_cell_fault_forms
from tests.utils.soak.ft.types import ROLLOUT_CELL_TYPE, FaultTrigger

from miles.utils.workers.types import ClusterBackend

_MODE = "kill_rollout__dp4"
_CHECKERS: tuple[str, ...] = (
    "assert_min_injections",
    "assert_injections_recovered",
    "assert_faults_span_progress_windows",
)


@pytest.fixture
def harness(scenario_harness: ScenarioHarness, monkeypatch: pytest.MonkeyPatch) -> ScenarioHarness:
    monkeypatch.setattr(comparison_app, "prepare", scenario_harness.record_prepare)
    monkeypatch.setattr(
        scenario_rollout_deterministic, "compare_deterministic_sides", scenario_harness.recorder("compare")
    )
    for name in _CHECKERS:
        monkeypatch.setattr(scenario_rollout_deterministic, name, scenario_harness.recorder(name))
    return scenario_harness


class TestTheTwoSidesOfTheComparison:
    def test_the_baseline_side_is_soaked_with_no_target_and_no_form(self, harness: ScenarioHarness) -> None:
        """A baseline that injected faults would compare a faulted run against a faulted run and prove nothing."""
        scenario_rollout_deterministic.run_ci(_MODE)

        baseline, _ = harness.soaks
        assert baseline["runner_config"].target_configs == {}
        assert baseline["forms"] == {}

    def test_the_target_side_soaks_every_rollout_engine_at_the_crash_interval(self, harness: ScenarioHarness) -> None:
        """The end-state check counts engines, and a smaller count would accept a run that lost one for good."""
        scenario_rollout_deterministic.run_ci(_MODE)

        _, target = harness.soaks
        assert target["runner_config"].target_configs == {
            ROLLOUT_CELL_TYPE: SoakTargetConfig(expected_count=4, mean_interval_seconds=30.0)
        }
        expected = create_cell_fault_forms(target["config"], triggers=frozenset({FaultTrigger.TIMER}))
        assert [form.name for form in target["forms"][ROLLOUT_CELL_TYPE]] == [
            form.name for form in expected[ROLLOUT_CELL_TYPE]
        ]
        assert set(target["forms"]) == {ROLLOUT_CELL_TYPE}

    def test_both_sides_share_the_seed_start_tail_and_ray_polling(self, harness: ScenarioHarness) -> None:
        """The sides differ only by injection, so any other difference in the runner would bias the comparison."""
        scenario_rollout_deterministic.run_ci(_MODE)

        for soak in harness.soaks:
            runner_config = soak["runner_config"]
            assert runner_config.seed == 42
            assert runner_config.start_after_rollout_id == 0
            assert runner_config.tail == SoakTailConfig.create(num_rollout=8)
            assert runner_config.poll_interval_seconds == 0.2
            assert runner_config.quiescent_polls_required == 1

    def test_each_side_runs_under_its_own_submission_dumps_and_evidence(self, harness: ScenarioHarness) -> None:
        """Sharing any of them would let the target stop, overwrite or archive the baseline it is compared to."""
        scenario_rollout_deterministic.run_ci(_MODE)

        baseline, target = harness.soaks
        root = _dump_root(harness)
        assert (baseline["dump_dir"], target["dump_dir"]) == (root / "baseline", root / "target")
        assert baseline["config"].ray_submission_id != target["config"].ray_submission_id
        assert baseline["evidence_dir"] != target["evidence_dir"]
        assert baseline["event_log"].path != target["event_log"].path
        assert [launch.config for launch in harness.launches] == [baseline["config"], target["config"]]

    def test_both_sides_launch_the_same_arguments_apart_from_their_dump_directory(
        self, harness: ScenarioHarness
    ) -> None:
        """Deterministic comparison needs identical runs, so only the side's own paths may differ."""
        scenario_rollout_deterministic.run_ci(_MODE)

        baseline, target = harness.launches
        assert baseline.request.train_args.replace("/baseline", "/target") == target.request.train_args
        assert "/baseline" in baseline.request.train_args


class TestTheDisaggregatedDeterministicLaunch:
    def test_the_arguments_pass_the_rollout_fault_tolerance_parser_gate(self, harness: ScenarioHarness) -> None:
        """A colocated or broadcast launch is refused by the real gate, so the soak could never start."""
        scenario_rollout_deterministic.run_ci(_MODE)

        for launch in harness.launches:
            parsed = parse_fault_tolerance_args(launch.request.train_args)
            assert parsed.ft_components == ["rollout"]
            assert parsed.partial_target_weight_update
            assert parsed.mini_ft_controller_enable
            assert not parsed.namespace.colocate

    def test_trainer_and_engines_get_disjoint_gpus_that_fill_the_node(self, harness: ScenarioHarness) -> None:
        """The rollout soak runs disaggregated, so the node must hold four training and four engine gpus."""
        mode = MODES[_MODE]

        scenario_rollout_deterministic.run_ci(_MODE)

        for launch in harness.launches:
            namespace = parse_fault_tolerance_args(launch.request.train_args).namespace
            assert namespace.actor_num_gpus_per_node == mode.train_gpus_per_node == 4
            assert namespace.rollout_num_gpus == 4
            assert namespace.rollout_num_gpus_per_engine == 1
            assert launch.request.num_gpus_per_node == 8

    def test_the_launch_is_deterministic_end_to_end(self, harness: ScenarioHarness) -> None:
        """Without the deterministic knobs two identical runs diverge and every comparison fails spuriously."""
        scenario_rollout_deterministic.run_ci(_MODE)

        for launch in harness.launches:
            assert "--debug-deterministic-collective" in launch.argv
            assert "--deterministic-mode" in launch.argv
            assert launch.value_of("--sglang-router-policy") == "round_robin"
            assert launch.value_of("--num-rollout") == "8"
            assert DETERMINISTIC_ENV_VARS.items() <= json.loads(launch.value_of("--train-env-vars")).items()

    def test_a_mode_that_is_fault_tolerant_on_training_is_refused_before_launching(
        self, harness: ScenarioHarness
    ) -> None:
        """This soak injects into rollout cells only, so a trainer-ft mode would soak nothing it enabled."""
        with pytest.raises(AssertionError, match="injects into rollout cells only"):
            scenario_rollout_deterministic.run_ci("kill_train_rollout__dp2_cp2")

        assert harness.launches == []
        assert harness.soaks == []


class TestWhatTheComparisonIsJudgedBy:
    def test_only_the_target_side_is_graded_on_its_own_events_and_forms(self, harness: ScenarioHarness) -> None:
        """Grading the baseline would fail a correct run for having no injections to recover from."""
        scenario_rollout_deterministic.run_ci(_MODE)

        _, target = harness.soaks
        events = target["event_log"].events
        assert [name for name in harness.checker_names if name != "compare"] == list(_CHECKERS)
        ((_, min_kwargs),) = harness.calls_of("assert_min_injections")
        assert (min_kwargs["kind"], min_kwargs["context"]) == (
            ROLLOUT_CELL_TYPE,
            "rollout_deterministic rollout cells",
        )
        ((recovered_args, recovered_kwargs),) = harness.calls_of("assert_injections_recovered")
        assert recovered_args == (events,)
        assert recovered_kwargs["forms"] is target["forms"]
        ((_, windows_kwargs),) = harness.calls_of("assert_faults_span_progress_windows")
        assert windows_kwargs["dump_dir"] == str(target["dump_dir"])

    def test_the_sides_are_compared_once_both_have_run_without_expected_reconfigures(
        self, harness: ScenarioHarness
    ) -> None:
        """A rollout-only soak must not reconfigure the trainer, so any reconfigure is a comparison failure."""
        scenario_rollout_deterministic.run_ci(_MODE)

        root = _dump_root(harness)
        assert harness.checker_names[-1] == "compare"
        ((_, compare_kwargs),) = harness.calls_of("compare")
        assert compare_kwargs == {
            "baseline_dir": f"{root}/baseline",
            "target_dir": f"{root}/target",
            "min_trained_rollouts": 2,
            "expected_target_reconfigures": [],
        }

    def test_a_failed_launch_is_raised_and_never_compared(self, harness: ScenarioHarness) -> None:
        """A crashed side leaves partial dumps, and comparing them would report a numeric diff, not the crash."""
        harness.launch_error = RuntimeError("the job exited 1")

        with pytest.raises(RuntimeError, match="the job exited 1"):
            scenario_rollout_deterministic.run_ci(_MODE)

        assert harness.calls_of("compare") == []


class TestQuiescenceOnEachBackend:
    def test_a_kubernetes_soak_waits_the_full_quiescence_and_names_no_ray_job(
        self, harness: ScenarioHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pod restarts settle slower than ray actors, and a single quiet poll would inject into a healing run."""
        monkeypatch.setenv("MILES_SCRIPT_CLUSTER_BACKEND", ClusterBackend.KUBERNETES.value)
        monkeypatch.setenv("MILES_SCRIPT_NAMESPACE", "miles-e2e")

        scenario_rollout_deterministic.run_ci(_MODE)

        for soak in harness.soaks:
            assert soak["runner_config"].quiescent_polls_required == QUIESCENT_POLLS_REQUIRED
            assert soak["config"].ray_submission_id is None
        assert {launch.config.cluster_backend for launch in harness.launches} == {ClusterBackend.KUBERNETES}


def _dump_root(harness: ScenarioHarness) -> Path:
    return harness.dumps_root / SCENARIO_RUN_ID / f"rollout_deterministic_{_MODE}"
