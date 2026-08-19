import shlex
from pathlib import Path

import pytest

from tests.e2e.deploy.conftest_deploy.hot_restart import scenario_hot_restart_realistic_gsm8k as scenario
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.fault_form import HotRestartFaultForm
from tests.e2e.ft.conftest_ft import scenario_realistic_gsm8k
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import ACTOR_CELL_TYPE
from tests.e2e.ft.conftest_ft.fault_injection.state import InjectionEvent

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.common import ArgvManipulator


def _run(dump_dir: str) -> scenario_realistic_gsm8k.Gsm8kRun:
    return scenario_realistic_gsm8k.Gsm8kRun(
        base_url="http://orchestrator:18080",
        config=ExecuteTrainConfig(run_id="demo", namespace="rl"),
        dump_dir=dump_dir,
        train_args="",
        launch=lambda config: None,
    )


class TestTheRecipeIsTheOneFtConverges:
    def test_the_run_is_the_realistic_gsm8k_run_and_not_a_copy_of_it(self):
        """A second recipe would drift from the one whose reward bounds this test inherits."""
        assert scenario.run_realistic_gsm8k is scenario_realistic_gsm8k.run_realistic_gsm8k

    def test_the_bounds_the_run_is_graded_against_are_the_ones_ft_declares(self):
        """The reward improvement is asserted by the run itself, off this threshold."""
        assert scenario.DEFAULT_METRIC_THRESHOLD is scenario_realistic_gsm8k.DEFAULT_METRIC_THRESHOLD
        assert scenario.DEFAULT_NUM_ROLLOUT is scenario_realistic_gsm8k.DEFAULT_NUM_ROLLOUT

    def test_this_scenario_spells_no_training_arguments_of_its_own_beyond_its_checkpoints(self):
        """Hot restart keeps the FT recipe except for checkpointing and its incompatible tensor checker."""
        declared = [one for one in shlex.split(scenario._build_train_args("/dumps")) if one.startswith("--")]

        assert sorted(declared) == ["--ci-disable-weight-update-checker", "--load", "--save", "--save-interval"]


class TestTheInjectionPlan:
    def test_the_only_fault_the_plan_may_draw_is_a_hot_restart(self):
        """A pod kill mixed in would make the trainer boot uuid this test pins change for a second reason."""
        forms = scenario.create_hot_restart_forms(_run("/dumps"))

        assert list(forms) == [ACTOR_CELL_TYPE]
        assert [type(one) for one in forms[ACTOR_CELL_TYPE]] == [HotRestartFaultForm]

    def test_the_plan_relaunches_the_release_the_run_was_installed_under(self):
        """A relaunch of another release would leave the trainers of this run behind."""
        run = _run("/dumps")
        [form] = scenario.create_hot_restart_forms(run)[ACTOR_CELL_TYPE]

        assert form._launch is run.launch
        assert form._config is run.config

    def test_the_form_reads_the_progress_of_the_run_it_restarts(self):
        """Eligibility is read off this run's checkpoints and events, not off a neighbouring dump directory."""
        run = _run("/dumps/gsm8k")
        [form] = scenario.create_hot_restart_forms(run)[ACTOR_CELL_TYPE]

        assert form._checkpoint_dir == scenario.compute_checkpoint_dir(run.dump_dir)
        assert form._events_dir == run.events_dir


class TestTheSpecTheseConstantsAreDocumentedIn:
    def test_the_spec_names_the_save_interval_the_run_is_installed_with(self):
        """The spec is what a reader reaches for, and a stale number there is worse than none."""
        assert f"--save-interval {scenario.SAVE_INTERVAL}" in _readme()

    def test_the_spec_names_the_interval_the_draws_are_spaced_by(self):
        """Both halves of this soak's cost are quoted there, and both drift the same way."""
        assert f"interval {int(scenario.DEFAULT_HOT_RESTART_INTERVAL_SECONDS)}s" in _readme()


def _readme() -> str:
    return (Path(__file__).resolve().parents[4] / "e2e" / "deploy" / "README.md").read_text()


class TestCheckpointArgs:
    def test_the_run_saves_and_resumes_from_one_directory_of_its_own(self):
        """A take-over restores the latest checkpoint, which the run has to be both writing and reading."""
        argv = shlex.split(scenario.build_checkpoint_args("/dumps/gsm8k"))
        checkpoint_dir = str(scenario.compute_checkpoint_dir("/dumps/gsm8k"))

        assert ArgvManipulator.get(argv, "--save") == [checkpoint_dir]
        assert ArgvManipulator.get(argv, "--load") == [checkpoint_dir]

    def test_the_run_saves_often_enough_for_a_take_over_to_find_a_checkpoint(self):
        """A run saving once would leave the whole soak ineligible, and no restart would ever fire."""
        argv = shlex.split(scenario.build_checkpoint_args("/dumps/gsm8k"))

        assert ArgvManipulator.get(argv, "--save-interval") == [str(scenario.SAVE_INTERVAL)]
        assert scenario.SAVE_INTERVAL < scenario.DEFAULT_NUM_ROLLOUT


class TestEveryDrawHasToLand:
    def test_a_soak_where_no_attempt_failed_passes(self):
        """Every draw of this form fires, so the log should hold successes only."""
        scenario.assert_no_take_over_attempt_failed(
            [_injection(succeeded=True), _injection(succeeded=True), _crash(succeeded=False)]
        )

    def test_a_soak_where_a_take_over_attempt_failed_is_a_failure(self):
        """Without the eligibility gate a failed attempt can only mean a relaunch that did not land."""
        with pytest.raises(AssertionError, match="take-over attempt\\(s\\) failed"):
            scenario.assert_no_take_over_attempt_failed([_injection(succeeded=True), _injection(succeeded=False)])

    def test_a_run_nothing_restarted_is_a_failure_and_not_a_pass(self):
        """Every assertion past this one is vacuous on a run whose script was never replaced."""
        assert scenario.MIN_HOT_RESTARTS >= 1


def _injection(*, succeeded: bool) -> InjectionEvent:
    return InjectionEvent(
        cell_name="actor-0", form_name=scenario.HOT_RESTART_FORM_NAME, succeeded=succeeded, harmed=False
    )


def _crash(*, succeeded: bool) -> InjectionEvent:
    return InjectionEvent(cell_name="actor-1", form_name="crash_pod", succeeded=succeeded, harmed=True)


class TestWhatEachTakeOverCost:
    def test_a_take_over_resuming_from_the_last_save_is_within_the_bound(self):
        """Whatever the draw's timing, a checkpointed take-over owes at most one save interval."""
        scenario.assert_take_over_loss_within_save_interval(
            [
                _record(index=0, saved=19, finished=19 + scenario.SAVE_INTERVAL),
                _record(index=1, saved=29, finished=29 + scenario.MAX_REDONE_STEPS_PER_TAKE_OVER),
            ]
        )

    def test_a_take_over_before_the_first_save_is_charged_for_every_step(self):
        """Starting over from the reference weights costs the whole run so far, which the bound still covers."""
        scenario.assert_take_over_loss_within_save_interval(
            [_record(index=0, saved=None, finished=scenario.SAVE_INTERVAL - 2)]
        )

    def test_a_take_over_that_threw_away_more_than_a_save_interval_fails(self):
        """That means the run resumed from something older than its latest checkpoint."""
        with pytest.raises(AssertionError, match="threw away"):
            scenario.assert_take_over_loss_within_save_interval(
                [_record(index=0, saved=9, finished=9 + scenario.MAX_REDONE_STEPS_PER_TAKE_OVER + 1)]
            )

    def test_the_bound_follows_the_save_interval_the_run_is_installed_with(self):
        """A bound spelled independently would drift the moment the interval changed."""
        assert scenario.MAX_REDONE_STEPS_PER_TAKE_OVER == scenario.SAVE_INTERVAL + 1


def _record(*, index: int, saved: int | None, finished: int) -> HotRestartRecord:
    return HotRestartRecord(index=index, saved_iteration_at_trigger=saved, frozen_rollout_id=finished)
