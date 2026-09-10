import random
import shlex
import shutil
import uuid
from collections.abc import Iterable
from pathlib import Path

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart import scenario_hot_restart_realistic_gsm8k as scenario
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.soak_form import SoakActionFormHotRestart
from tests.utils.soak.core import SoakActionScheduler
from tests.utils.soak.recipes import gsm8k as scenario_realistic_gsm8k
from tests.utils.soak.state import (
    InjectionEvent,
    SoakActionAppliedEvent,
    SoakActionRequest,
    SoakDeploymentTarget,
    SoakObservation,
    SoakScheduleEvent,
)

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, EventLogger
from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
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

    def test_this_scenario_only_adds_arguments_its_take_over_requires(self):
        """Hot restart keeps the FT recipe except for checkpointing, tracking identity, and its tensor checker."""
        declared = [
            one
            for one in shlex.split(scenario._build_train_args("/dumps", wandb_run_id="run-one"))
            if one.startswith("--")
        ]

        assert sorted(declared) == [
            "--ci-disable-weight-update-checker",
            "--load",
            "--save",
            "--save-interval",
            "--wandb-run-id",
        ]

    def test_the_run_id_is_the_identity_of_the_deployment_this_test_restarts(self):
        """Every process receives the outer config ID instead of an identity derived from another subsystem."""
        argv = shlex.split(scenario._build_train_args("/dumps/another-id", wandb_run_id="run-one"))

        assert ArgvManipulator.get(argv, "--wandb-run-id") == ["run-one"]

    def test_the_shared_recipe_can_keep_its_api_without_enabling_training_ft(self):
        """Hot restart uses the cell API for injection without combining with automatic FT recovery."""
        argv = shlex.split(
            scenario_realistic_gsm8k.get_gsm8k_train_args(
                config=ExecuteTrainConfig(run_id="260101-000000-000", namespace="miles-e2e"),
                seed=scenario.DEFAULT_SEED,
                num_rollout=scenario.DEFAULT_NUM_ROLLOUT,
                metric_threshold=scenario.DEFAULT_METRIC_THRESHOLD,
                fully_async=False,
                test_name=scenario.TEST_NAME,
                enable_fault_tolerance=False,
            )
        )

        assert ArgvManipulator.get(argv, "--api-server-port")
        assert "--use-fault-tolerance" not in argv
        assert "--ft-components" not in argv
        assert "--mini-ft-controller-enable" not in argv


class TestIntermediateCheckpoints:
    @pytest.mark.parametrize("saved", [None, 230, 233, 234])
    def test_the_next_takeover_waits_for_a_checkpoint_newer_than_the_previous_applied_snapshot(
        self, saved: int | None
    ) -> None:
        """A repeated checkpoint cannot satisfy the save-between-takeovers scenario."""
        run = _run("/dumps")
        (form,) = scenario.create_hot_restart_forms(run, max_allowed_rollout_id=249)["deployment"]
        target = _deployment(finished=231)
        request = SoakActionRequest(target=target, form_name=scenario.HOT_RESTART_FORM_NAME, harms_cell=False)
        run.event_log.note_action_requested(request)
        after = target.model_copy(update={"saved_iteration": 233, "finished_rollout_id": 233})
        run.event_log.note_action_applied(
            SoakActionAppliedEvent(request_id=request.request_id, evidence={"after": after.model_dump(mode="json")})
        )
        run.event_log.note_observation(
            SoakObservation(
                cells=[], deployments=[after.model_copy(update={"saved_iteration": saved, "finished_rollout_id": 235})]
            )
        )
        assert form.is_within_injection_window() is (saved == 234)

    @pytest.mark.parametrize("second_saved", [None, 233, 234])
    def test_the_final_verdict_requires_two_applied_takeovers_separated_by_a_new_checkpoint(
        self, second_saved: int | None
    ) -> None:
        """One restart or reuse of the preceding checkpoint cannot earn consecutive-restart coverage."""
        run = _run("/dumps")
        first = _deployment(finished=231)
        after = first.model_copy(update={"saved_iteration": 233})
        for index, saved in enumerate([230] if second_saved is None else [230, second_saved]):
            request = SoakActionRequest(
                target=first.model_copy(update={"saved_iteration": saved}),
                form_name=scenario.HOT_RESTART_FORM_NAME,
                harms_cell=False,
            )
            run.event_log.note_action_requested(request)
            run.event_log.note_action_applied(
                SoakActionAppliedEvent(
                    request_id=request.request_id,
                    evidence={
                        "after": after.model_copy(update={"saved_iteration": 233 + index}).model_dump(mode="json")
                    },
                )
            )
        if second_saved == 234:
            scenario._assert_checkpoints_advanced_between_takeovers(run.event_log.events)
        else:
            with pytest.raises(AssertionError):
                scenario._assert_checkpoints_advanced_between_takeovers(run.event_log.events)


class TestTheInjectionPlan:
    def test_deployment_remains_available_before_the_closing_window(self):
        """A draw before the final fifteen rollouts still reaches the ordinary scheduler."""
        run = _run("/dumps")
        [form] = scenario.create_hot_restart_forms(run, max_allowed_rollout_id=234)["deployment"]
        run.event_log.note_observation(SoakObservation(cells=[], deployments=[_deployment(finished=233)]))

        assert form.is_within_injection_window()

    def test_deployment_becomes_ineligible_at_the_closing_window(self):
        """Completing rollout 234 leaves all of 235-249 free of new take-overs."""
        run = _run("/dumps")
        [form] = scenario.create_hot_restart_forms(run, max_allowed_rollout_id=234)["deployment"]
        run.event_log.note_observation(SoakObservation(cells=[], deployments=[_deployment(finished=234)]))

        assert not form.is_within_injection_window()

    def test_one_real_deployment_is_sufficient_without_borrowing_ft_cells(self):
        """Hot restart targets the actual deployment and needs no imaginary spare cell."""
        forms = scenario.create_hot_restart_forms(_run("/dumps"), max_allowed_rollout_id=234)
        scheduler = SoakActionScheduler(rng=random.Random(0), mean_intervals={"deployment": 1}, forms=forms)
        target = _deployment(finished=233)
        request = scheduler.choose(
            events=[SoakScheduleEvent(due_of_type={"deployment": 0}), SoakObservation(cells=[], deployments=[target])],
            now=1,
        )
        assert request is not None and request.target == target
        assert not request.harms_cell

    def test_the_only_fault_the_plan_may_draw_is_a_hot_restart(self):
        """A pod kill mixed in would make the trainer boot uuid this test pins change for a second reason."""
        forms = scenario.create_hot_restart_forms(_run("/dumps"), max_allowed_rollout_id=234)

        assert list(forms) == [scenario._HOT_RESTART_TARGET_TYPE]
        assert [type(one) for one in forms[scenario._HOT_RESTART_TARGET_TYPE]] == [SoakActionFormHotRestart]

    def test_the_plan_relaunches_the_release_the_run_was_installed_under(self):
        """A relaunch of another release would leave the trainers of this run behind."""
        run = _run("/dumps")
        [form] = scenario.create_hot_restart_forms(run, max_allowed_rollout_id=234)[scenario._HOT_RESTART_TARGET_TYPE]

        assert form._launch_spec.train_args == run.train_args
        assert form._launch_spec.config is run.config

    def test_the_form_reads_the_progress_of_the_run_it_restarts(self):
        """Eligibility is read off this run's checkpoints and events, not off a neighbouring dump directory."""
        run = _run("/dumps/gsm8k")
        [form] = scenario.create_hot_restart_forms(run, max_allowed_rollout_id=234)[scenario._HOT_RESTART_TARGET_TYPE]
        observer = scenario._create_observer(run)

        assert form._event_log is run.event_log
        assert observer.checkpoint_dir == scenario.compute_checkpoint_dir(run.dump_dir)
        assert observer.events_dir == run.events_dir


def _deployment(*, finished: int) -> SoakDeploymentTarget:
    return SoakDeploymentTarget(
        namespace="rl",
        release="demo",
        workload_stamps={},
        workload_uids={},
        saved_iteration=230,
        finished_rollout_id=finished,
    )


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
        cell_name="hot-restart-virtual-cell-0",
        form_name=scenario.HOT_RESTART_FORM_NAME,
        succeeded=succeeded,
        harmed=False,
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


def _records(count: int) -> list[HotRestartRecord]:
    return [_record(index=index, saved=0, finished=0) for index in range(count)]


def _write_finished_steps(events_dir: Path, rollout_ids: Iterable[int]) -> None:
    for rollout_id in rollout_ids:
        logger = EventLogger(
            log_dir=events_dir, file_name=f"step-{rollout_id}.jsonl", source=SimpleProcessIdentity(component="main")
        )
        logger.log(MetricEvent, {"rollout_id": rollout_id, "metrics": {"train/grad_norm": 1.0}}, print_log=False)


def _replace_log(dump_dir: Path, *, rolled_aside_at: str, kept: Iterable[int]) -> None:
    events_dir = dump_dir / EVENTS_DIRNAME
    replaced = dump_dir / f".trash_{rolled_aside_at}_{uuid.uuid4().hex[:8]}"
    shutil.move(str(events_dir), str(replaced))
    events_dir.mkdir(parents=True)
    for rollout_id in kept:
        shutil.copy(replaced / f"step-{rollout_id}.jsonl", events_dir / f"step-{rollout_id}.jsonl")


class TestAssertEveryTakeOverResumedWithinASaveInterval:
    def test_a_take_over_that_redid_only_what_its_checkpoint_missed_passes(self, tmp_path):
        """What a take-over cost is the log it replaced minus the prefix the run that followed kept."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(4))
        _replace_log(tmp_path, rolled_aside_at="20260902_120000", kept=range(2))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(2, 6))

        scenario.assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(1))

    def test_a_take_over_that_resumed_further_back_than_a_save_interval_fails(self, tmp_path):
        """Measuring the resume point off the trigger's tracker would pass a run that reloaded an older save."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(6))
        _replace_log(tmp_path, rolled_aside_at="20260902_120000", kept=[])
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(6))

        with pytest.raises(AssertionError, match="redid 6 step"):
            scenario.assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(1))

    def test_a_take_over_that_left_no_rolled_back_log_fails(self, tmp_path):
        """A take-over whose log is missing hides the very steps this assertion counts."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(4))

        with pytest.raises(AssertionError, match="rolls the log it replaced aside"):
            scenario.assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(1))

    def test_a_take_over_that_carried_a_hole_over_fails(self, tmp_path):
        """A run resumes from one checkpoint, so what survives a take-over is a prefix and never a hole."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(4))
        _replace_log(tmp_path, rolled_aside_at="20260902_120000", kept=[0, 2])
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(3, 6))

        with pytest.raises(AssertionError, match="carried the steps"):
            scenario.assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(1))

    def test_a_take_over_fired_during_the_catch_up_is_read_off_the_log_it_rolled_aside(self, tmp_path):
        """A take-over during the catch-up leaves a log reaching a lower step than the log before it."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(10))
        _replace_log(tmp_path, rolled_aside_at="20260902_120000", kept=range(8))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, [8])
        _replace_log(tmp_path, rolled_aside_at="20260902_120100", kept=range(7))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(7, 12))

        scenario.assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(2))

    def test_a_take_over_fired_during_the_catch_up_is_measured_against_its_own_log(self, tmp_path):
        """Ordering the rolled-aside logs by how far they trained blames the wrong take-over for the redone steps."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(10))
        _replace_log(tmp_path, rolled_aside_at="20260902_120000", kept=range(8))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, [8])
        _replace_log(tmp_path, rolled_aside_at="20260902_120100", kept=range(2))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(2, 12))

        with pytest.raises(AssertionError, match="take-over 1 replaced a log that had reached step 8"):
            scenario.assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(2))

    def test_two_take_overs_that_rolled_their_logs_aside_in_the_same_second_fail(self, tmp_path):
        """Which log a take-over replaced is read off when it was rolled aside, so a tie leaves it unpinned."""
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(4))
        _replace_log(tmp_path, rolled_aside_at="20260902_120000", kept=range(2))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(2, 5))
        _replace_log(tmp_path, rolled_aside_at="20260902_120000", kept=range(3))
        _write_finished_steps(tmp_path / EVENTS_DIRNAME, range(3, 6))

        with pytest.raises(AssertionError, match="same second"):
            scenario.assert_take_overs_resumed_within_save_interval(str(tmp_path), records=_records(2))
