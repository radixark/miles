import dataclasses
import json
import shlex
from pathlib import Path

import pytest
import tests.e2e.deploy
from tests.e2e.deploy.conftest_deploy.hot_restart import scenario_hot_restart_deterministic as scenario
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.freeze_plan import compute_freeze_plan_path
from tests.e2e.deploy.conftest_deploy.hot_restart.scenario_hot_restart_deterministic import (
    HotRestartMode,
    compute_checkpoint_dir,
    read_installed_args,
)

from miles.utils.external_utils.command_utils.common import ArgvManipulator
from miles.utils.misc import should_run_periodic_action
from miles.utils.test_utils.ft_test_actions import CI_FT_TEST_ACTIONS_PATH_FLAG

ENTRY_DIR: Path = Path(tests.e2e.deploy.__file__).parent


def _first_saved_rollout_id(restart_mode: HotRestartMode) -> int:
    return next(
        rollout_id
        for rollout_id in range(scenario.NUM_ROLLOUTS)
        if should_run_periodic_action(
            rollout_id, restart_mode.save_interval, num_rollout_per_epoch=None, num_rollout=scenario.NUM_ROLLOUTS
        )
    )


class TestTheMode:
    def test_the_mode_is_offered_by_the_table_the_scenario_runs(self):
        """A mode outside the table is a mode no entry can ask for."""
        assert scenario.NO_CHECKPOINT in scenario.MODES

    def test_the_entry_of_the_no_checkpoint_mode_runs_that_mode(self):
        """Both entries run one scenario, so the mode is the only thing telling them apart."""
        source = (ENTRY_DIR / f"test_{scenario.NO_CHECKPOINT.test_name}.py").read_text()

        assert "import NO_CHECKPOINT, run_ci" in source
        assert "run_ci(NO_CHECKPOINT)" in source

    def test_the_run_is_frozen_before_it_has_written_anything(self):
        """A pinned save here would make this the take-over the checkpointed mode already covers."""
        assert [one.saved_iteration for one in scenario.NO_CHECKPOINT.schedule] == [None]

    def test_the_redo_of_this_mode_is_measured_against_a_run_that_started_over(self):
        """A run resuming from a checkpoint and one starting over leave different logs behind."""
        assert (
            scenario.NO_CHECKPOINT.assert_redone
            is scenario.assert_a_run_that_had_saved_nothing_was_redone_from_scratch
        )


class TestTiming:
    def test_the_run_finishes_a_step_before_the_freeze_it_is_taken_over_at(self):
        """A take-over before the run finished anything wastes nothing and proves nothing."""
        assert scenario.NO_CHECKPOINT.frozen_rollout_ids == (1,)

    def test_the_run_has_not_saved_by_the_step_it_freezes_at(self):
        """This is the whole point of the mode: a take-over of a run holding no checkpoint at all."""
        assert _first_saved_rollout_id(scenario.NO_CHECKPOINT) > scenario.NO_CHECKPOINT.frozen_rollout_ids[0]

    def test_the_run_still_saves_after_the_restart_it_is_given(self):
        """A run whose only save is its last step would never exercise saving after a take-over."""
        assert _first_saved_rollout_id(scenario.NO_CHECKPOINT) < scenario.NUM_ROLLOUTS - 1

    def test_the_run_is_taken_over_exactly_once(self):
        """The second take-over resumes from a checkpoint, which the checkpointed mode already covers."""
        assert scenario.NO_CHECKPOINT.num_restarts == 1

    def test_the_gradient_floor_sits_above_the_window_the_take_over_can_redo(self):
        """A floor the redone steps alone could fill would pass a run that trained nothing past them."""
        assert _first_saved_rollout_id(scenario.NO_CHECKPOINT) < scenario.MIN_TRAINED_ROLLOUTS <= scenario.NUM_ROLLOUTS

    def test_starting_over_accounts_for_every_republished_update(self):
        """With no checkpoint, versions for every surviving event include both updates done before takeover."""
        records = (HotRestartRecord(index=0, saved_iteration_at_trigger=None, frozen_rollout_id=1),)

        expected = scenario._compute_expected_weight_version_deltas(records=records, num_rollouts=6)

        assert all(by_occurrence == [2, 2, 2, 2, 2, 2] for by_occurrence in expected.values())


class TestBuildArgs:
    def test_the_relaunch_of_a_run_repeats_the_arguments_it_was_installed_with(self, monkeypatch):
        """Each call draws a new run id, and a hot restart whose wandb group followed it would change the pods."""
        monkeypatch.setenv("WANDB_API_KEY", "key")

        first = scenario._build_args(scenario.NO_CHECKPOINT, scenario._MODE, "/dumps/no-checkpoint/target")
        second = scenario._build_args(scenario.NO_CHECKPOINT, scenario._MODE, "/dumps/no-checkpoint/target", True)

        assert "--wandb-group" in first
        assert first == second
        assert read_installed_args("/dumps/no-checkpoint/target") == first

    def test_a_relaunch_repeats_the_string_the_run_was_installed_with(self):
        """Rebuilding the arguments would drop whatever the pipeline was asked for, such as the dumper."""
        args = scenario._build_args(scenario.NO_CHECKPOINT, scenario._MODE, "/dumps/no-checkpoint/plain", False)

        assert read_installed_args("/dumps/no-checkpoint/plain") == args
        assert "--dumper-dir" not in args

    def test_the_run_is_installed_with_the_save_interval_the_timing_is_reasoned_from(self):
        """The gate reads the checkpoint directory, so a run saving at another pace opens it at another step."""
        args = scenario._build_args(scenario.NO_CHECKPOINT, scenario._MODE, "/dumps/no-checkpoint/interval")

        assert ArgvManipulator.get(shlex.split(args), "--save-interval") == [str(scenario.NO_CHECKPOINT.save_interval)]

    def test_each_side_of_the_comparison_checkpoints_into_its_own_directory(self):
        """A shared checkpoint directory would hand the target a checkpoint the baseline wrote."""
        args = scenario._build_args(scenario.NO_CHECKPOINT, scenario._MODE, "/dumps/nc/target")

        assert str(compute_checkpoint_dir("/dumps/nc/target")) in args
        assert str(compute_checkpoint_dir("/dumps/nc/baseline")) not in args

    def test_the_weight_decay_of_the_common_arguments_is_replaced_and_not_repeated(self):
        """A repeated flag leaves it to the parser which value wins, and this run needs the one it asked for."""
        argv = shlex.split(scenario._build_args(scenario.NO_CHECKPOINT, scenario._MODE, "/dumps/no-checkpoint/decay"))

        assert ArgvManipulator.get(argv, "--weight-decay") == ["0"]

    def test_a_run_that_would_only_ever_save_its_last_step_is_refused(self):
        """The mode watches the save that follows the take-over, and such a run performs none."""
        rare = dataclasses.replace(scenario.NO_CHECKPOINT, save_interval=scenario.NUM_ROLLOUTS)

        with pytest.raises(AssertionError, match="only ever saves the last step"):
            scenario._build_args(rare, scenario._MODE, "/dumps/no-checkpoint/rare", False)

    def test_a_run_that_saves_the_step_it_freezes_at_is_refused(self):
        """Such a take-over resumes exactly where the script it replaced stood, so it redoes nothing."""
        eager = dataclasses.replace(scenario.NO_CHECKPOINT, save_interval=1)

        with pytest.raises(AssertionError, match="redo nothing"):
            scenario._build_args(eager, scenario._MODE, "/dumps/no-checkpoint/eager", False)

    # TODO ad hoc hack: revert after the args refactor
    def test_the_target_side_is_armed_to_sleep_before_the_run_has_saved_anything(self, tmp_path):
        """Nothing else holds the run in the one window where it has trained but written nothing."""
        dump_dir = f"{tmp_path}/target"
        args = scenario._build_frozen_args(scenario.NO_CHECKPOINT, scenario._MODE, dump_dir, False)

        plan_path = compute_freeze_plan_path(dump_dir)
        assert ArgvManipulator.get(shlex.split(args), CI_FT_TEST_ACTIONS_PATH_FLAG) == [str(plan_path)]
        assert json.loads(plan_path.read_text()) == [
            {"at_rollout": scenario.NO_CHECKPOINT.frozen_rollout_ids[0], "action": "sleep_forever_at_end"}
        ]
