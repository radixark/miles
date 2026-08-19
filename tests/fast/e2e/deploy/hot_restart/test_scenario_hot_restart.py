import dataclasses
import json
import shlex
from pathlib import Path
from typing import Any

import pytest
import tests.e2e.deploy
from tests.e2e.deploy.conftest_deploy.hot_restart import scenario_hot_restart_deterministic as scenario
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import ScheduledFreeze
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartEvidence, HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.freeze_plan import compute_freeze_plan_path
from tests.e2e.deploy.conftest_deploy.hot_restart.scenario_hot_restart_deterministic import (
    HotRestartMode,
    compute_checkpoint_dir,
    read_installed_args,
)
from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE

from miles.utils.external_utils.command_utils.common import ArgvManipulator
from miles.utils.misc import should_run_periodic_action
from miles.utils.test_utils.ft_test_actions import CI_FT_TEST_ACTIONS_PATH_FLAG

ENTRY_DIR: Path = Path(tests.e2e.deploy.__file__).parent


def _saved_rollout_ids(restart_mode: HotRestartMode) -> list[int]:
    return [
        rollout_id
        for rollout_id in range(scenario.NUM_ROLLOUTS)
        if should_run_periodic_action(
            rollout_id, restart_mode.save_interval, num_rollout_per_epoch=None, num_rollout=scenario.NUM_ROLLOUTS
        )
    ]


class TestModes:
    def test_every_mode_of_the_table_names_its_own_test(self):
        """The test name reaches dump directories and wandb runs, so two modes sharing one would collide."""
        assert len({one.test_name for one in scenario.MODES}) == len(scenario.MODES)

    def test_the_checkpointed_mode_is_the_one_the_table_offers(self):
        """A mode nothing runs is a mode nobody notices going stale."""
        assert scenario.CHECKPOINTED in scenario.MODES

    def test_the_entry_of_the_checkpointed_mode_runs_that_mode(self):
        """Nothing else ties the file a red ci job names back to the mode it drove."""
        source = (ENTRY_DIR / f"test_{scenario.CHECKPOINTED.test_name}.py").read_text()

        assert "import CHECKPOINTED, run_ci" in source
        assert "run_ci(CHECKPOINTED)" in source


class TestTiming:
    def test_the_scenario_pins_a_freeze_for_every_restart_it_drives(self):
        """A schedule shorter than the run would leave a take-over free to land wherever it liked."""
        assert len(scenario.CHECKPOINTED.schedule) == scenario.CHECKPOINTED.num_restarts

    def test_the_save_each_take_over_resumes_from_is_one_this_run_really_writes(self):
        """The pinned save is what the take-over rolls back to, so the run has to write it."""
        assert [one.saved_iteration for one in scenario.CHECKPOINTED.schedule] == [1, 3]
        assert {one.saved_iteration for one in scenario.CHECKPOINTED.schedule} <= set(
            _saved_rollout_ids(scenario.CHECKPOINTED)
        )

    def test_the_cadence_the_pinned_saves_are_checked_against_is_the_one_the_run_really_saves_at(self):
        """The literal pinned saves are validated against this rule, and a rule of its own would drift."""
        assert sorted(
            scenario.compute_saved_rollout_ids(save_interval=scenario.CHECKPOINTED.save_interval)
        ) == _saved_rollout_ids(scenario.CHECKPOINTED)

    def test_every_freeze_lands_on_a_step_the_run_did_not_save(self):
        """This is why the run saves every other step: the take-over throws away work no checkpoint holds."""
        assert not set(scenario.CHECKPOINTED.frozen_rollout_ids) & set(_saved_rollout_ids(scenario.CHECKPOINTED))

    def test_the_windows_two_take_overs_redo_cannot_overlap(self):
        """Two restarts sharing a redone step would make that step cost three attempts, not two."""
        assert all(
            later.saved_iteration >= earlier.frozen_rollout_id
            for earlier, later in zip(scenario.CHECKPOINTED.schedule, scenario.CHECKPOINTED.schedule[1:], strict=False)
        )

    def test_the_run_still_trains_a_step_the_last_take_over_did_not_redo(self):
        """A run whose every remaining step is a redone one proves nothing about training on past a take-over."""
        assert max(scenario.CHECKPOINTED.frozen_rollout_ids) < scenario.NUM_ROLLOUTS - 1

    def test_the_gradient_floor_sits_within_the_steps_the_run_trains(self):
        """The surviving log holds every step, and a floor above them could never be met."""
        assert scenario.MIN_TRAINED_ROLLOUTS <= scenario.NUM_ROLLOUTS

    def test_a_mode_freezing_where_its_own_cadence_saves_is_refused(self):
        """Such a take-over resumes exactly where the script it replaced stood, so it redoes nothing."""
        eager = dataclasses.replace(scenario.CHECKPOINTED, save_interval=1)

        with pytest.raises(AssertionError, match="redo nothing"):
            scenario.assert_freeze_schedule_leaves_redo_window(eager)

    def test_a_mode_whose_run_would_only_ever_save_its_last_step_is_refused(self):
        """The comparison watches the run save after it was taken over, and such a run performs none."""
        rare = dataclasses.replace(scenario.CHECKPOINTED, save_interval=scenario.NUM_ROLLOUTS)

        with pytest.raises(AssertionError, match="only ever saves the last step"):
            scenario.assert_freeze_schedule_leaves_redo_window(rare)

    def test_a_mode_freezing_twice_after_the_same_step_is_refused(self):
        """The driver reads the sentinel's step to tell this freeze from the one before it."""
        repeated = dataclasses.replace(
            scenario.CHECKPOINTED,
            schedule=(
                ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),
                ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),
            ),
        )

        with pytest.raises(AssertionError, match="freezes the run twice after the same step"):
            scenario.assert_freeze_schedule_leaves_redo_window(repeated)

    def test_a_mode_pinning_a_save_its_own_cadence_never_writes_is_refused(self):
        """The pinned save is a literal now, and a wrong one would resume the take-over somewhere else."""
        wrong = dataclasses.replace(
            scenario.CHECKPOINTED, schedule=(ScheduledFreeze(frozen_rollout_id=2, saved_iteration=0),)
        )

        with pytest.raises(AssertionError, match="never reasoned about"):
            scenario.assert_freeze_schedule_leaves_redo_window(wrong)


class TestWeightVersionExclusion:
    def test_the_comparison_drops_the_weight_version_statistics_and_nothing_else(self, tmp_path, monkeypatch):
        """A surviving engine keeps a monotonic publication counter, but every other metric still compares exactly."""
        calls: list[dict[str, Any]] = []
        dump_dir = _dump_dir_with_evidence(tmp_path)
        monkeypatch.setattr(scenario, "assert_take_overs_replaced_only_script", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(scenario, "compare_deterministic_sides", lambda **kwargs: calls.append(kwargs))
        restart_mode = dataclasses.replace(scenario.CHECKPOINTED, assert_redone=lambda **_kwargs: None)

        scenario._compare(restart_mode, dump_dir, scenario._MODE)

        assert calls == [
            dict(
                baseline_dir=f"{dump_dir}/{BASELINE_SIDE}",
                target_dir=f"{dump_dir}/{TARGET_SIDE}",
                expected_engine_count=scenario._MODE.rollout_num_engines,
                min_trained_rollouts=scenario.MIN_TRAINED_ROLLOUTS,
                exclude_keys=[
                    "rollout/weight_version/mean",
                    "rollout/weight_version/median",
                    "rollout/weight_version/max",
                    "rollout/weight_version/min",
                ],
            )
        ]


def _dump_dir_with_evidence(tmp_path: Path) -> str:
    dump_dir = tmp_path / "hot_restart_checkpointed"
    evidence = HotRestartEvidence(
        records=(
            HotRestartRecord(index=0, saved_iteration_at_trigger=1, frozen_rollout_id=2),
            HotRestartRecord(index=1, saved_iteration_at_trigger=3, frozen_rollout_id=4),
        ),
        snapshots=(),
        release="demo",
    )
    evidence.write(dump_dir=str(dump_dir / TARGET_SIDE))
    return str(dump_dir)


class TestBuildArgs:
    def test_the_relaunch_of_a_run_repeats_the_arguments_it_was_installed_with(self, monkeypatch):
        """Each call draws a new run id, and a hot restart whose wandb group followed it would change the pods."""
        monkeypatch.setenv("WANDB_API_KEY", "key")

        first = scenario._build_args(scenario.CHECKPOINTED, scenario._MODE, "/dumps/target")
        second = scenario._build_args(scenario.CHECKPOINTED, scenario._MODE, "/dumps/target", True)

        argv = shlex.split(first)

        assert ArgvManipulator.get(argv, scenario.WANDB_GROUP_FLAG) == ArgvManipulator.get(
            argv, scenario.WANDB_RUN_ID_FLAG
        )
        assert first == second

    def test_a_relaunch_repeats_the_string_the_run_was_installed_with(self):
        """Rebuilding the arguments would drop whatever the pipeline was asked for, such as the dumper."""
        args = scenario._build_args(scenario.CHECKPOINTED, scenario._MODE, "/dumps/target/plain", False)

        assert read_installed_args("/dumps/target/plain") == args
        assert "--dumper-dir" not in args

    def test_relaunching_a_run_this_process_never_installed_fails(self):
        """A relaunch of arguments nobody installed would render a pod template of its own."""
        with pytest.raises(AssertionError, match="nothing installed a run"):
            read_installed_args("/dumps/target/never-installed")

    def test_the_weight_decay_of_the_common_arguments_is_replaced_and_not_repeated(self):
        """A repeated flag leaves it to the parser which value wins, and this run needs the one it asked for."""
        argv = shlex.split(scenario._build_args(scenario.CHECKPOINTED, scenario._MODE, "/dumps/target"))

        assert ArgvManipulator.get(argv, "--weight-decay") == ["0"]

    def test_each_side_of_the_comparison_checkpoints_into_its_own_directory(self):
        """A shared checkpoint directory would let the target resume from what the baseline wrote."""
        args = scenario._build_args(scenario.CHECKPOINTED, scenario._MODE, "/dumps/target")

        assert str(compute_checkpoint_dir("/dumps/target")) in args
        assert str(compute_checkpoint_dir("/dumps/baseline")) not in args

    def test_the_run_is_installed_with_the_save_interval_its_mode_is_reasoned_from(self):
        """The pinned triggers are read off this cadence, so a run saving at another pace lands elsewhere."""
        argv = shlex.split(scenario._build_args(scenario.CHECKPOINTED, scenario._MODE, "/dumps/target/interval"))

        assert ArgvManipulator.get(argv, "--save-interval") == [str(scenario.CHECKPOINTED.save_interval)]

    def test_two_modes_installing_one_dump_directory_would_still_be_told_apart(self):
        """The wandb group is derived per test name, so one soak's history cannot swallow another's."""
        first = scenario._compute_wandb_group(test_name="hot_restart_checkpointed", dump_dir="/dumps/target")
        second = scenario._compute_wandb_group(test_name="hot_restart_no_checkpoint", dump_dir="/dumps/target")

        assert first != second


class TestTheFreezeTheRunIsInstalledWith:
    # TODO ad hoc hack: revert after the args refactor
    def test_the_target_side_is_armed_to_sleep_at_the_first_pinned_step(self, tmp_path):
        """Nothing else makes the run stand still, and a take-over of a moving run lands wherever it likes."""
        dump_dir = f"{tmp_path}/target"
        args = scenario._build_frozen_args(scenario.CHECKPOINTED, scenario._MODE, dump_dir, False)

        plan_path = compute_freeze_plan_path(dump_dir)
        assert ArgvManipulator.get(shlex.split(args), CI_FT_TEST_ACTIONS_PATH_FLAG) == [str(plan_path)]
        assert json.loads(plan_path.read_text()) == [
            {"at_rollout": scenario.CHECKPOINTED.frozen_rollout_ids[0], "action": "sleep_forever_at_end"}
        ]

    # TODO ad hoc hack: revert after the args refactor
    def test_the_baseline_side_is_never_frozen(self):
        """The baseline is the run nobody touched, and one asleep at step 2 would never finish."""
        args = scenario._build_args(scenario.CHECKPOINTED, scenario._MODE, "/dumps/baseline/plain", False)

        assert not ArgvManipulator.is_defined(shlex.split(args), CI_FT_TEST_ACTIONS_PATH_FLAG)

    def test_the_relaunch_repeats_the_frozen_arguments_the_run_is_up_with(self, tmp_path):
        """A relaunch whose argv differs from the installed one is refused as more than a hot restart."""
        dump_dir = f"{tmp_path}/target"
        args = scenario._build_frozen_args(scenario.CHECKPOINTED, scenario._MODE, dump_dir, False)

        assert read_installed_args(dump_dir) == args


class TestTheModesThisScenarioRefuses:
    def test_a_mode_whose_last_take_over_leaves_no_step_to_train_is_refused(self):
        """Every remaining step being a redone one proves nothing about training on past a take-over."""
        too_late = dataclasses.replace(
            scenario.CHECKPOINTED,
            schedule=(ScheduledFreeze(frozen_rollout_id=scenario.NUM_ROLLOUTS - 1, saved_iteration=3),),
        )

        with pytest.raises(AssertionError, match="leaving no step past the last take-over"):
            scenario.assert_freeze_schedule_leaves_redo_window(too_late)

    def test_a_colocated_mode_is_refused(self):
        """A take-over keeps the trainers and the engines up, and a colocated mode shares their gpus."""
        colocated = dataclasses.replace(
            scenario._MODE, colocate=True, rollout_num_engines=2, rollout_gpus_per_engine=1
        )

        with pytest.raises(AssertionError, match="colocates them"):
            scenario._build_script_args(
                scenario.CHECKPOINTED, mode=colocated, dump_dir="/dumps/target", enable_dumper=False
            )

    def test_a_mode_with_no_engines_is_refused(self):
        """The take-over replaces the rollout executor, which needs engines to drive when it returns."""
        engineless = dataclasses.replace(scenario._MODE, rollout_num_engines=0, rollout_gpus_per_engine=0)

        with pytest.raises(AssertionError, match="no engines for it to drive"):
            scenario._build_script_args(
                scenario.CHECKPOINTED, mode=engineless, dump_dir="/dumps/target", enable_dumper=False
            )


class TestTheOneEventPerRolloutPremise:
    def test_the_run_is_installed_to_take_exactly_one_optimizer_step_per_rollout(self):
        """Every attempt count in the redo verdict is read off one grad_norm event per rollout."""
        argv = shlex.split(scenario._build_args(scenario.CHECKPOINTED, scenario._MODE, "/dumps/target/batch", False))
        product = int(ArgvManipulator.get(argv, scenario.ROLLOUT_BATCH_SIZE_FLAG)[0]) * int(
            ArgvManipulator.get(argv, scenario.SAMPLES_PER_PROMPT_FLAG)[0]
        )

        assert int(ArgvManipulator.get(argv, scenario.GLOBAL_BATCH_SIZE_FLAG)[0]) == product

    def test_a_run_taking_several_optimizer_steps_per_rollout_is_refused(self):
        """Such a run logs one event per step, so every attempt count would be a multiple of the truth."""
        with pytest.raises(AssertionError, match="one train/grad_norm event per rollout"):
            scenario._assert_one_train_event_per_step(
                "--global-batch-size 128 --rollout-batch-size 32 --n-samples-per-prompt 8 "
            )


class TestTheVerdictEachModeIsMeasuredAgainst:
    def test_the_checkpointed_mode_is_paired_with_the_checkpointed_verdict(self):
        """A mode pointing at the other verdict would measure the run against the wrong redo."""
        assert scenario.CHECKPOINTED.assert_redone is scenario.assert_only_post_checkpoint_steps_redone

    def test_the_comparison_hands_the_verdict_the_target_side_dumps(self):
        """Handing it the base directory would read the two sides' logs as one run's."""
        recorded: dict[str, object] = {}

        def record(**kwargs) -> None:
            recorded.update(kwargs)

        mode = dataclasses.replace(scenario.CHECKPOINTED, assert_redone=record)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(scenario.HotRestartEvidence, "load", classmethod(lambda _cls, *, dump_dir: _evidence()))
            patch.setattr(scenario, "assert_take_overs_replaced_only_script", lambda *a, **k: None)
            patch.setattr(scenario, "compare_deterministic_sides", lambda **_kwargs: None)
            scenario._compare(mode, "/dumps/hot_restart_checkpointed", scenario._MODE)

        assert str(recorded["dump_dir"]).endswith("/target")
        assert str(recorded["checkpoint_dir"]).endswith("/target/checkpoints")
        assert recorded["num_rollouts"] == scenario.NUM_ROLLOUTS
        assert recorded["schedule"] == mode.schedule


def _evidence() -> HotRestartEvidence:
    return HotRestartEvidence(records=(), snapshots=(), release="miles-run-demo-all")


class TestTheSaveShapeTheTakeOverNeeds:
    def test_the_run_is_installed_to_save_synchronously(self):
        """Every take-over is pinned to the checkpoint the frozen run is already holding."""
        args = scenario._build_args(scenario.CHECKPOINTED, scenario._MODE, "/dumps/target/sync", False)

        assert not ArgvManipulator.is_defined(shlex.split(args), scenario.ASYNC_SAVE_FLAG)

    def test_a_run_saving_asynchronously_is_refused(self):
        """Such a checkpoint can land after the step that triggered it, so the pin means nothing."""
        with pytest.raises(AssertionError, match="lets a checkpoint land after"):
            scenario._assert_run_saves_before_step_report("--save /ckpt --async-save ")
