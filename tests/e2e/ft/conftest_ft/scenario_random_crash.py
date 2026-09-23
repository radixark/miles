# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.


import asyncio
from pathlib import Path

import typer
from tests.e2e.ft.conftest_ft.cli_options import (
    FullyAsyncOption,
    ModeOption,
    NumStepsOption,
    RolloutCrashIntervalSecondsOption,
    SeedOption,
    TrainerCrashIntervalSecondsOption,
)
from tests.e2e.ft.conftest_ft.execution import (
    get_common_train_args,
    get_ft_args,
    materialize_cyclic_debug_rollout_data,
    prepare,
    run_training,
)
from tests.e2e.ft.conftest_ft.modes import FTTestMode, resolve_mode
from tests.utils.ft.launch import get_fully_async_args, get_train_script
from tests.utils.soak.core.checkers.engine_checksums import assert_engine_checksums_cover_published_updates
from tests.utils.soak.core.config import SoakRunnerConfig, SoakTailConfig, SoakTargetConfig
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.runner import SoakRunner
from tests.utils.soak.core.utils import (
    API_SERVER_ARGS,
    assert_fresh_dump_dir,
    create_soak_config,
    evidence_directory,
    note_launch_outcome,
    resolve_dump_dir,
)
from tests.utils.soak.core.views import read_training_events
from tests.utils.soak.ft.actions.factory import compute_mean_interval_seconds_of_kind
from tests.utils.soak.ft.checkers.healing import assert_healing
from tests.utils.soak.ft.entrypoint import run_cell_soak
from tests.utils.soak.ft.types import ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE

from miles.utils.external_utils import command_utils

app: typer.Typer = typer.Typer()

TEST_NAME: str = "random_crash"

DEFAULT_SEED: int = 42
DEFAULT_NUM_STEPS: int = 60
DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS: float = 120.0
DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS: float = 240.0


@app.command(name="run")
def run_ci(
    mode: ModeOption,
    seed: SeedOption = DEFAULT_SEED,
    num_steps: NumStepsOption = DEFAULT_NUM_STEPS,
    trainer_crash_interval_seconds: TrainerCrashIntervalSecondsOption = DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS,
    rollout_crash_interval_seconds: RolloutCrashIntervalSecondsOption = DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS,
    fully_async: FullyAsyncOption = False,
) -> None:
    """Random failure soak test, for whichever components the mode enables ft on.

    Runs an async session that injects faults at random intervals via the
    api server HTTP API. The mini FT controller auto-recovers; the test passes
    if training completes without hanging.

    Doubles as the per-mode CI entry point: a CI file calls ``run_ci(mode)`` (defaults);
    manual runs use the ``run`` CLI subcommand with optional --seed/--num-steps/etc.
    """
    ft_mode: FTTestMode = resolve_mode(mode)
    if fully_async:
        assert_mode_supports_fully_async(ft_mode, mode=mode)

    config = create_soak_config(command_utils.default_config())
    test_name: str = f"{TEST_NAME}_fully_async" if fully_async else TEST_NAME
    dump_dir: str = resolve_dump_dir(f"{test_name}_{mode}", run_id=config.run_id)
    print(f"Dump directory: {dump_dir}")
    mean_interval_seconds_of_cell_type: dict[str, float] = compute_mean_interval_seconds_of_kind(
        ft_mode.ft_components,
        trainer_crash_interval_seconds=trainer_crash_interval_seconds,
        rollout_crash_interval_seconds=rollout_crash_interval_seconds,
    )
    print(f"Seed: {seed}, Steps: {num_steps}, Mean injection intervals: {mean_interval_seconds_of_cell_type}")
    print(f"FT components: {ft_mode.ft_components}, cluster backend: {config.cluster_backend.value}")
    print(f"Train script: {get_train_script(fully_async=fully_async)}")

    prepare(ft_mode, config=config)
    train_args = _build_train_args(
        ft_mode, config=config, dump_dir=dump_dir, num_steps=num_steps, fully_async=fully_async
    )

    injector = _run_soak(
        ft_mode,
        config=config,
        dump_dir=dump_dir,
        seed=seed,
        num_steps=num_steps,
        mean_interval_seconds_of_cell_type=mean_interval_seconds_of_cell_type,
        train_args=train_args,
        fully_async=fully_async,
    )

    training_events = read_training_events(injector.event_log.events, dump_dir=dump_dir)
    if ft_mode.has_real_rollout:
        assert_engine_checksums_cover_published_updates(training_events)
    assert_healing(
        ft_mode.ft_components,
        events=injector.event_log.events,
        forms=injector.forms,
        context=f"{test_name} {mode}",
    )

    print(f"Random failure soak test PASSED ({test_name}, mode={mode}, seed={seed}, steps={num_steps})")


def _build_train_args(
    ft_mode: FTTestMode, *, config: command_utils.ExecuteTrainConfig, dump_dir: str, num_steps: int, fully_async: bool
) -> str:
    debug_rollout_data_dir = None if ft_mode.has_real_rollout else materialize_cyclic_debug_rollout_data(num_steps)
    train_args = (
        get_common_train_args(
            ft_mode, dump_dir=dump_dir, num_steps=num_steps, debug_rollout_data_dir=debug_rollout_data_dir
        )
        + get_ft_args(ft_mode, api_server_args=API_SERVER_ARGS)
        + get_fully_async_args(fully_async=fully_async)
        + "--mini-ft-controller-enable "
    )
    assert_fresh_dump_dir(Path(dump_dir))
    return train_args


def _run_soak(
    ft_mode: FTTestMode,
    *,
    config: command_utils.ExecuteTrainConfig,
    dump_dir: str,
    seed: int,
    num_steps: int,
    mean_interval_seconds_of_cell_type: dict[str, float],
    train_args: str,
    fully_async: bool,
) -> SoakRunner:
    expected_counts: dict[str, int] = {
        ACTOR_CELL_TYPE: ft_mode.num_cells,
        ROLLOUT_CELL_TYPE: ft_mode.rollout_num_engines,
    }
    evidence_dir = evidence_directory(Path(dump_dir))
    event_log = EventLog(evidence_dir / "events.jsonl")
    return asyncio.run(
        run_cell_soak(
            config=config,
            dump_dir=Path(dump_dir),
            sut_run=note_launch_outcome(
                event_log=event_log,
                request_id=None,
                launching=asyncio.to_thread(
                    run_training,
                    train_args=train_args,
                    mode=ft_mode,
                    config=config,
                    train_script=get_train_script(fully_async=fully_async),
                ),
            ),
            runner_config=SoakRunnerConfig(
                seed=seed,
                target_configs={
                    kind: SoakTargetConfig(expected_count=expected_counts[kind], mean_interval_seconds=interval)
                    for kind, interval in mean_interval_seconds_of_cell_type.items()
                },
                tail=SoakTailConfig.create(num_rollout=num_steps),
            ),
            event_log=event_log,
            evidence_dir=evidence_dir,
        )
    )


def assert_mode_supports_fully_async(ft_mode: FTTestMode, *, mode: str) -> None:
    assert ft_mode.has_real_rollout, (
        f"Mode {mode!r} has no rollout engines, so a fully-async soak would train off pre-recorded debug rollout "
        f"data and would prove nothing about generating while training"
    )


if __name__ == "__main__":
    app()
