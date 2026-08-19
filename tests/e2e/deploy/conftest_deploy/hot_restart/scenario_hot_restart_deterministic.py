import hashlib
import shlex
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import typer
from examples.infra_features.hot_restart.run_qwen3_0_6b_hot_restart import ScriptArgs, build_train_args
from examples.infra_features.split_deployment.address_book import DEFAULT_TRAINER_ID
from tests.e2e.deploy.conftest_deploy.common.example_args import (
    assert_example_parallelism_matches,
    build_deterministic_test_args,
    build_script_args,
    with_replaced_value,
    without_weight_decay,
)
from tests.e2e.deploy.conftest_deploy.common.utils import compare_deterministic_sides, run_on_cluster
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_redone_from_checkpoint import (
    assert_only_post_checkpoint_steps_redone,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_redone_from_scratch import (
    assert_unsaved_run_redone_from_scratch,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_workloads import assert_take_overs_replaced_only_script
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import (
    HotRestartDriver,
    ScheduledFreeze,
    compute_checkpoint_dir,
    compute_release_of_config,
    driving_hot_restarts,
    relaunch_with_hot_restart,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import TRAIN_STEP_METRIC_KEY, HotRestartEvidence
from tests.e2e.deploy.conftest_deploy.hot_restart.freeze_plan import (
    arm_first_freeze,
    compute_freeze_plan_path,
    write_freeze_plan,
)
from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE, create_comparison_app_and_run_ci
from tests.e2e.ft.conftest_ft.execution import DATA_DIR, MODEL_DIR
from tests.e2e.ft.conftest_ft.modes import DENSE_MODEL_HF_REPO, DENSE_MODEL_NAME, DENSE_MODEL_TYPE, FTTestMode

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.common import ArgvManipulator, get_mooncake_object_store_args

# ========================== constants and mode table ==========================


NUM_ROLLOUTS: int = 6
MIN_TRAINED_ROLLOUTS: int = 4
SAVE_FLAG: str = "--save"
LOAD_FLAG: str = "--load"
WANDB_GROUP_FLAG: str = "--wandb-group"


GLOBAL_BATCH_SIZE_FLAG: str = "--global-batch-size"
ROLLOUT_BATCH_SIZE_FLAG: str = "--rollout-batch-size"
SAMPLES_PER_PROMPT_FLAG: str = "--n-samples-per-prompt"
ASYNC_SAVE_FLAG: str = "--async-save"

_MODE: FTTestMode = FTTestMode(
    model_name=DENSE_MODEL_NAME,
    model_hf_repo=DENSE_MODEL_HF_REPO,
    megatron_model_type=DENSE_MODEL_TYPE,
    num_cells=2,
    train_gpus_per_node=4,
    rollout_num_engines=2,
    rollout_gpus_per_engine=1,
    parallel_args="--context-parallel-size 2",
)

_TRAIN_ARGS_OF_DUMP_DIR: dict[str, str] = {}


AssertRedoneFn = Callable[..., object]


@dataclass(frozen=True)
class HotRestartMode:
    name: str
    save_interval: int
    schedule: tuple[ScheduledFreeze, ...]
    assert_redone: AssertRedoneFn

    @property
    def test_name(self) -> str:
        return f"hot_restart_{self.name}"

    @property
    def num_restarts(self) -> int:
        return len(self.schedule)

    @property
    def frozen_rollout_ids(self) -> tuple[int, ...]:
        return tuple(one.frozen_rollout_id for one in self.schedule)


CHECKPOINTED: HotRestartMode = HotRestartMode(
    name="checkpointed",
    save_interval=2,
    schedule=(
        ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),
        ScheduledFreeze(frozen_rollout_id=4, saved_iteration=3),
    ),
    assert_redone=assert_only_post_checkpoint_steps_redone,
)
NO_CHECKPOINT: HotRestartMode = HotRestartMode(
    name="no_checkpoint",
    save_interval=4,
    schedule=(ScheduledFreeze(frozen_rollout_id=1, saved_iteration=None),),
    assert_redone=assert_unsaved_run_redone_from_scratch,
)
MODES: tuple[HotRestartMode, ...] = (CHECKPOINTED, NO_CHECKPOINT)


# ================================ entry points ================================


def create_app_and_run_ci(restart_mode: HotRestartMode) -> tuple[typer.Typer, Callable[[], None]]:
    app, run_ci = create_comparison_app_and_run_ci(
        test_name=restart_mode.test_name,
        build_baseline_args=partial(_build_args, restart_mode),
        build_target_args=partial(_build_frozen_args, restart_mode),
        compare_fn=partial(_compare, restart_mode),
        target_side_context=partial(_driving_take_overs_of, restart_mode),
        resolve_mode_fn=lambda _name: _MODE,
    )
    return app, run_on_cluster(run_ci)


def run_ci(restart_mode: HotRestartMode) -> None:
    create_app_and_run_ci(restart_mode)[1]()


def read_installed_args(dump_dir: str) -> str:
    assert (args := _TRAIN_ARGS_OF_DUMP_DIR.get(dump_dir)) is not None, (
        f"nothing installed a run under {dump_dir} in this process; a relaunch that rebuilds its arguments instead "
        f"of repeating the installed ones renders a pod template of its own"
    )
    return args


# ========================== train argument building ===========================


def _build_args(restart_mode: HotRestartMode, mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    checkpoint_dir = str(compute_checkpoint_dir(dump_dir))
    script_args = _build_script_args(restart_mode, mode=mode, dump_dir=dump_dir, enable_dumper=enable_dumper)

    args = without_weight_decay(build_train_args(script_args))
    for flag in (SAVE_FLAG, LOAD_FLAG):
        args = with_replaced_value(args, flag=flag, value=checkpoint_dir)
    if ArgvManipulator.is_defined(shlex.split(args), WANDB_GROUP_FLAG):
        wandb_run_id = _compute_wandb_group(test_name=restart_mode.test_name, dump_dir=dump_dir)
        for flag in (WANDB_GROUP_FLAG, WANDB_RUN_ID_FLAG):
            args = with_replaced_value(args, flag=flag, value=wandb_run_id)
    args += get_mooncake_object_store_args()

    assert_example_parallelism_matches(mode, train_args=args)
    _assert_one_train_event_per_step(args)
    _assert_run_saves_before_step_report(args)
    _TRAIN_ARGS_OF_DUMP_DIR[dump_dir] = args
    return args


def _assert_one_train_event_per_step(train_args: str) -> None:
    argv = shlex.split(train_args)
    [global_batch_size] = ArgvManipulator.get(argv, GLOBAL_BATCH_SIZE_FLAG)
    [rollout_batch_size] = ArgvManipulator.get(argv, ROLLOUT_BATCH_SIZE_FLAG)
    [samples_per_prompt] = ArgvManipulator.get(argv, SAMPLES_PER_PROMPT_FLAG)

    assert int(global_batch_size) == int(rollout_batch_size) * int(samples_per_prompt), (
        f"the redo accounting of this scenario counts one {TRAIN_STEP_METRIC_KEY} event per rollout, and a run "
        f"whose global batch {global_batch_size} is not {rollout_batch_size} x {samples_per_prompt} takes several "
        f"optimizer steps per rollout and logs one event for each, so every attempt count would be a multiple of "
        f"what the schedule reasons about"
    )


def _assert_run_saves_before_step_report(train_args: str) -> None:
    assert not ArgvManipulator.is_defined(shlex.split(train_args), ASYNC_SAVE_FLAG), (
        f"{ASYNC_SAVE_FLAG} lets a checkpoint land after the step that triggered it, and every take-over here is "
        f"pinned to the checkpoint the frozen run is already holding"
    )


def _build_frozen_args(
    restart_mode: HotRestartMode, mode: FTTestMode, dump_dir: str, enable_dumper: bool = True
) -> str:
    # TODO ad hoc hack: revert after the args refactor
    args = arm_first_freeze(
        _build_args(restart_mode, mode, dump_dir, enable_dumper),
        side_dump_dir=dump_dir,
        frozen_rollout_id=restart_mode.frozen_rollout_ids[0],
    )
    _TRAIN_ARGS_OF_DUMP_DIR[dump_dir] = args
    return args


def _build_script_args(
    restart_mode: HotRestartMode, *, mode: FTTestMode, dump_dir: str, enable_dumper: bool
) -> ScriptArgs:
    assert mode.has_real_rollout, (
        f"{restart_mode.test_name} replaces the rollout executor of a live run, and mode {mode.model_name} has no "
        f"engines for it to drive"
    )
    assert not mode.colocate, (
        f"{restart_mode.test_name} keeps a run's trainers and engines up while their script is replaced, and mode "
        f"{mode.model_name} colocates them on shared gpus"
    )

    assert_freeze_schedule_leaves_redo_window(restart_mode)

    return build_script_args(
        command_utils.default_config(),
        script_args_class=ScriptArgs,
        model_name=mode.model_name,
        megatron_model_type=mode.megatron_model_type,
        num_rollout=NUM_ROLLOUTS,
        save_interval=restart_mode.save_interval,
        actor_num_gpus=mode.train_gpus_per_node,
        num_engines=mode.rollout_num_engines,
        gpus_per_engine=mode.rollout_gpus_per_engine,
        model_dir=MODEL_DIR,
        data_dir=DATA_DIR,
        extra_args=build_deterministic_test_args(mode=mode, dump_dir=dump_dir, enable_dumper=enable_dumper),
    )


def _compute_wandb_group(*, test_name: str, dump_dir: str) -> str:
    return f"{test_name}_{hashlib.sha256(dump_dir.encode()).hexdigest()[:12]}"


# ============================= take-over driving ==============================


@contextmanager
def _driving_take_overs_of(
    restart_mode: HotRestartMode, mode: FTTestMode, dump_dir: str, config: command_utils.ExecuteTrainConfig
) -> Iterator[None]:
    release = compute_release_of_config(config)
    # TODO ad hoc hack: revert after the args refactor
    plan_path = compute_freeze_plan_path(dump_dir)

    def relaunch(frozen_rollout_id: int | None) -> None:
        write_freeze_plan(plan_path, frozen_rollout_id=frozen_rollout_id)
        relaunch_with_hot_restart(
            train_args=read_installed_args(dump_dir), mode=mode, config=config, installed_release=release
        )

    driver = HotRestartDriver(
        relaunch=relaunch,
        checkpoint_dir=compute_checkpoint_dir(dump_dir),
        events_dir=Path(dump_dir) / EVENTS_DIRNAME,
        release=release,
        namespace=config.namespace,
        trainer_id=DEFAULT_TRAINER_ID,
        freeze_plan_path=plan_path,
        schedule=restart_mode.schedule,
    )

    with driving_hot_restarts(driver, dump_dir=dump_dir):
        yield

    driver.assert_all_restarts_happened()


# ========================= comparison and assertions ==========================


def assert_freeze_schedule_leaves_redo_window(restart_mode: HotRestartMode) -> None:
    assert restart_mode.save_interval < NUM_ROLLOUTS, (
        f"{restart_mode.test_name} watches the run save after a take-over, and a run of {NUM_ROLLOUTS} step(s) "
        f"saving every {restart_mode.save_interval} only ever saves the last step it trains"
    )
    assert max(restart_mode.frozen_rollout_ids) < NUM_ROLLOUTS - 1, (
        f"{restart_mode.test_name} freezes the run after step {max(restart_mode.frozen_rollout_ids)} of "
        f"{NUM_ROLLOUTS}, leaving no step past the last take-over that is not a redone one"
    )
    frozen = restart_mode.frozen_rollout_ids
    assert len(set(frozen)) == len(frozen), (
        f"{restart_mode.test_name} freezes the run twice after the same step ({list(frozen)}), and the driver "
        f"tells one freeze from the one before it by the step the sentinel names: the sentinel a previous freeze "
        f"left would be read as this one, and the take-over would go before the run had parked"
    )

    saved = compute_saved_rollout_ids(save_interval=restart_mode.save_interval)
    for scheduled in restart_mode.schedule:
        assert scheduled.frozen_rollout_id not in saved, (
            f"{restart_mode.test_name} freezes the run after step {scheduled.frozen_rollout_id}, which a run "
            f"saving every {restart_mode.save_interval} step(s) checkpoints, so the take-over would redo nothing"
        )
        pinned = max((one for one in saved if one < scheduled.frozen_rollout_id), default=None)
        assert scheduled.saved_iteration == pinned, (
            f"{restart_mode.test_name} pins the take-over after step {scheduled.frozen_rollout_id} to iteration "
            f"{scheduled.saved_iteration}, and a run saving every {restart_mode.save_interval} step(s) holds "
            f"{pinned} by then: the take-over resumes from somewhere this scenario never reasoned about"
        )


def compute_saved_rollout_ids(*, save_interval: int) -> frozenset[int]:
    return frozenset(one for one in range(NUM_ROLLOUTS) if (one + 1) % save_interval == 0 or one == NUM_ROLLOUTS - 1)


def _compare(restart_mode: HotRestartMode, dump_dir: str, mode: FTTestMode) -> None:
    baseline_dir: str = f"{dump_dir}/{BASELINE_SIDE}"
    target_dir: str = f"{dump_dir}/{TARGET_SIDE}"

    evidence = HotRestartEvidence.load(dump_dir=target_dir)
    assert_take_overs_replaced_only_script(
        evidence, num_restarts=len(evidence.records), minimum_restarts=restart_mode.num_restarts
    )
    restart_mode.assert_redone(
        dump_dir=target_dir,
        checkpoint_dir=str(compute_checkpoint_dir(target_dir)),
        records=evidence.records,
        num_rollouts=NUM_ROLLOUTS,
        schedule=restart_mode.schedule,
    )

    compare_deterministic_sides(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        expected_engine_count=mode.rollout_num_engines,
        min_trained_rollouts=MIN_TRAINED_ROLLOUTS,
        exclude_keys=list(_WEIGHT_VERSION_METRIC_KEYS),
    )

    print(f"Hot restart {restart_mode.name} comparison test PASSED")


# ================================= app wiring =================================


app: typer.Typer = typer.Typer()
for one in MODES:
    app.add_typer(create_app_and_run_ci(one)[0], name=one.name)

if __name__ == "__main__":
    app()
