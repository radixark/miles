import os
from pathlib import Path

import pytest
from tests.e2e.ft.conftest_ft import scenario_realistic_gsm8k
from tests.fast.e2e.scenario_harness import SCENARIO_RUN_ID, ScenarioHarness, parse_fault_tolerance_args
from tests.utils.soak.core.config import SoakTailConfig, SoakTargetConfig
from tests.utils.soak.core.events import LaunchOutcome
from tests.utils.soak.ft import fault_triggers
from tests.utils.soak.ft.actions.factory import create_cell_fault_forms
from tests.utils.soak.ft.types import ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE, FaultTrigger
from tests.utils.soak.recipes import gsm8k


@pytest.fixture
def harness(scenario_harness: ScenarioHarness, monkeypatch: pytest.MonkeyPatch) -> ScenarioHarness:
    monkeypatch.setattr(gsm8k, "create_backend_for_run", lambda config: config.create_backend())
    monkeypatch.setattr(gsm8k, "prepare_gsm8k", scenario_harness.record_backend_prepare)
    monkeypatch.setattr(fault_triggers, "assert_hook_evidence", scenario_harness.recorder("assert_hook_evidence"))
    monkeypatch.setattr(scenario_realistic_gsm8k, "assert_healing", scenario_harness.recorder("assert_healing"))
    return scenario_harness


class TestTheSoakTheGsm8kRunSchedules:
    def test_both_kinds_are_soaked_with_the_recipe_layout_and_their_own_cadence(
        self, harness: ScenarioHarness
    ) -> None:
        """The counts must match the recipe's two CP2 trainer cells and four engines, or end state is misjudged."""
        _run(seed=5, num_rollout=40)

        (soak,) = harness.soaks
        runner_config = soak["runner_config"]
        assert runner_config.seed == 5
        assert runner_config.target_configs == {
            ACTOR_CELL_TYPE: SoakTargetConfig(expected_count=2, mean_interval_seconds=11.0),
            ROLLOUT_CELL_TYPE: SoakTargetConfig(expected_count=4, mean_interval_seconds=13.0),
        }
        assert runner_config.tail == SoakTailConfig.create(num_rollout=40)

    def test_the_forms_are_those_of_the_requested_triggers(self, harness: ScenarioHarness) -> None:
        """A timer-only run given hook forms would arm hooks the launch never gave a long enough timeout."""
        _run(seed=5, num_rollout=40, requested_triggers=[FaultTrigger.TIMER])

        (soak,) = harness.soaks
        (launch,) = harness.launches
        expected = create_cell_fault_forms(soak["config"], triggers=frozenset({FaultTrigger.TIMER}))
        assert {kind: [form.name for form in forms] for kind, forms in soak["forms"].items()} == {
            kind: [form.name for form in expected[kind]] for kind in (ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE)
        }
        assert "--update-weights-timeout" not in launch.argv


class TestOneGsm8kRunIdentity:
    def test_prepare_the_launch_and_the_soak_share_one_config(self, harness: ScenarioHarness) -> None:
        """The recipe builds the soak config once, and a second one would name another ray job for teardown."""
        _run(seed=5, num_rollout=40)

        (soak,) = harness.soaks
        (launch,) = harness.launches
        (prepared,) = harness.prepared
        assert launch.config is soak["config"]
        assert prepared["config"] == soak["config"]
        assert soak["config"].ray_submission_id.startswith("miles-soak-")

    def test_the_run_event_log_is_the_soak_event_log_and_records_the_launch(self, harness: ScenarioHarness) -> None:
        """The runner stops on the launch outcome, so it must land in the log the runner reads."""
        _run(seed=5, num_rollout=40)

        (soak,) = harness.soaks
        dump_dir = harness.dumps_root / SCENARIO_RUN_ID / "realistic_gsm8k"
        assert soak["dump_dir"] == dump_dir
        assert soak["event_log"].path == soak["evidence_dir"] / "events.jsonl"
        assert soak["evidence_dir"].parent == dump_dir.with_name("realistic_gsm8k-soak")
        (event,) = soak["event_log"].events
        assert event.outcome is LaunchOutcome.FINISHED

    @pytest.mark.parametrize(
        ("fully_async", "requested_triggers", "name"),
        [
            (False, [FaultTrigger.TIMER], "realistic_gsm8k_timer"),
            (True, None, "realistic_gsm8k_fully_async"),
        ],
    )
    def test_each_variant_writes_under_a_name_of_its_own(
        self,
        harness: ScenarioHarness,
        fully_async: bool,
        requested_triggers: list[FaultTrigger] | None,
        name: str,
    ) -> None:
        """Variants sharing a dump directory would refuse to start after the first, or grade its evidence."""
        _run(seed=5, num_rollout=40, fully_async=fully_async, requested_triggers=requested_triggers)

        (soak,) = harness.soaks
        assert soak["dump_dir"] == harness.dumps_root / SCENARIO_RUN_ID / name

    def test_the_launch_runs_without_the_proxies_of_the_launching_shell(
        self, harness: ScenarioHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A proxied local API request would never reach the run the soak is driving."""
        monkeypatch.setenv("http_proxy", "http://proxy.example:3128")

        _run(seed=5, num_rollout=40)

        assert "http_proxy" not in os.environ


class TestTheLaunchedGsm8kArguments:
    def test_the_arguments_pass_the_fault_tolerance_parser_gate(self, harness: ScenarioHarness) -> None:
        """Rollout fault tolerance needs a disaggregated p2p update, or the real validator refuses the launch."""
        _run(seed=5, num_rollout=40)

        (launch,) = harness.launches
        parsed = parse_fault_tolerance_args(launch.request.train_args)
        assert parsed.ft_components == ["train", "rollout"]
        assert parsed.partial_target_weight_update
        assert parsed.mini_ft_controller_enable
        assert not parsed.namespace.colocate
        assert parsed.namespace.actor_num_gpus_per_node == 4
        assert parsed.namespace.rollout_num_gpus == 4
        assert launch.request.num_gpus_per_node == 8
        assert launch.request.megatron_model_type == gsm8k.MODEL_TYPE

    def test_the_run_and_threshold_reach_the_launched_arguments(self, harness: ScenarioHarness) -> None:
        """Grading at another threshold or length would pass or fail runs the scenario never meant to judge."""
        _run(seed=5, num_rollout=40, metric_threshold=0.42)

        (launch,) = harness.launches
        (soak,) = harness.soaks
        assert launch.value_of("--num-rollout") == "40"
        assert launch.value_of("--ci-metric-checker-threshold") == "0.42"
        assert launch.value_of("--update-weights-timeout") == "600"
        assert Path(launch.value_of("--save-debug-event-data")) == soak["dump_dir"] / "events"
        assert launch.request.train_script.endswith("/train.py")

    def test_a_fully_async_run_launches_the_async_driver(self, harness: ScenarioHarness) -> None:
        """The sync driver ignores --fully-async, so the soak would grade a mode that never ran."""
        _run(seed=5, num_rollout=40, fully_async=True)

        (launch,) = harness.launches
        assert launch.request.train_script.endswith("/train_async.py")
        assert "--fully-async" in launch.argv


class TestWhatTheGsm8kSoakIsJudgedBy:
    def test_the_checkers_read_this_soak_and_both_components(self, harness: ScenarioHarness) -> None:
        """Checking fewer components or another log would pass a soak on evidence it never produced."""
        _run(seed=5, num_rollout=40)

        (soak,) = harness.soaks
        events = soak["event_log"].events
        assert harness.checker_names == ["assert_hook_evidence", "assert_healing"]
        ((hook_args, hook_kwargs),) = harness.calls_of("assert_hook_evidence")
        assert hook_args == (frozenset({FaultTrigger.TIMER, FaultTrigger.HOOK}),)
        assert hook_kwargs == {
            "ft_components": ("train", "rollout"),
            "config": soak["config"],
            "events": events,
            "dump_dir": str(soak["dump_dir"]),
        }
        ((healing_args, healing_kwargs),) = harness.calls_of("assert_healing")
        assert healing_args == (("train", "rollout"),)
        assert healing_kwargs["events"] == events
        assert healing_kwargs["forms"] is soak["forms"]
        assert healing_kwargs["context"] == "realistic_gsm8k"

    def test_a_failed_launch_is_raised_recorded_and_never_graded(self, harness: ScenarioHarness) -> None:
        """A run that crashed must fail the soak instead of being graded on the injections it saw."""
        harness.launch_error = RuntimeError("the job exited 1")

        with pytest.raises(RuntimeError, match="the job exited 1"):
            _run(seed=5, num_rollout=40)

        (soak,) = harness.soaks
        (event,) = soak["event_log"].events
        assert event.outcome is LaunchOutcome.FAILED
        assert harness.checks == []


def _run(
    *,
    seed: int,
    num_rollout: int,
    metric_threshold: float = gsm8k.DEFAULT_METRIC_THRESHOLD,
    fully_async: bool = False,
    requested_triggers: list[FaultTrigger] | None = None,
) -> None:
    scenario_realistic_gsm8k.run_ci(
        seed=seed,
        num_rollout=num_rollout,
        trainer_crash_interval_seconds=11.0,
        rollout_crash_interval_seconds=13.0,
        metric_threshold=metric_threshold,
        fully_async=fully_async,
        requested_triggers=requested_triggers,
    )
