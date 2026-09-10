# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.


from functools import partial
from pathlib import Path

import typer
from tests.e2e.ft.conftest_ft.app import resolve_dump_dir
from tests.e2e.ft.conftest_ft.cli_options import (
    AllowDuringRecoveryOption,
    FullyAsyncOption,
    MaxConcurrentActionsOption,
    MinSurvivorsOption,
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
from tests.utils.soak.checks.ft import assert_healing
from tests.utils.soak.checks.tail import assert_tail_complete
from tests.utils.soak.config import create_policy, create_tail_policy
from tests.utils.soak.entrypoint import API_SERVER_PORT, spawn_fault_injector
from tests.utils.soak.fault_forms import compute_mean_interval_seconds_of_cell_type, create_cell_fault_forms
from tests.utils.soak.teardown import teardown_run
from tests.utils.soak.utils import (
    create_soak_config,
    evidence_directory,
    get_api_server_args,
    get_fully_async_args,
    get_train_script,
)

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
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
    allow_during_recovery: AllowDuringRecoveryOption = True,
    min_survivors: MinSurvivorsOption = 1,
    max_concurrent_actions: MaxConcurrentActionsOption = 1,
) -> None:
    """Random failure soak test, for whichever components the mode enables ft on.

    Starts a background thread that injects faults at random intervals via the
    api server HTTP API. The mini FT controller auto-recovers; the test passes
    if training completes without hanging.

    Doubles as the per-mode CI entry point: a CI file calls ``run_ci(mode)`` (defaults);
    manual runs use the ``run`` CLI subcommand with optional --seed/--num-steps/etc.
    """
    ft_mode: FTTestMode = resolve_mode(mode)
    tail_policy = create_tail_policy(num_rollout=num_steps)
    if fully_async:
        assert_mode_supports_fully_async(ft_mode, mode=mode)

    config = create_soak_config(command_utils.default_config())
    test_name: str = f"{TEST_NAME}_fully_async" if fully_async else TEST_NAME
    dump_dir: str = resolve_dump_dir(f"{test_name}_{mode}", run_id=config.run_id)
    print(f"Dump directory: {dump_dir}")
    mean_interval_seconds_of_cell_type: dict[str, float] = compute_mean_interval_seconds_of_cell_type(
        ft_mode.ft_components,
        trainer_crash_interval_seconds=trainer_crash_interval_seconds,
        rollout_crash_interval_seconds=rollout_crash_interval_seconds,
    )
    print(f"Seed: {seed}, Steps: {num_steps}, Mean injection intervals: {mean_interval_seconds_of_cell_type}")
    print(f"FT components: {ft_mode.ft_components}, cluster backend: {config.cluster_backend.value}")
    print(f"Train script: {get_train_script(fully_async=fully_async)}")

    prepare(ft_mode, config=config)

    debug_rollout_data_dir = None if ft_mode.has_real_rollout else materialize_cyclic_debug_rollout_data(num_steps)
    train_args = (
        get_common_train_args(
            ft_mode, dump_dir=dump_dir, num_steps=num_steps, debug_rollout_data_dir=debug_rollout_data_dir
        )
        + get_ft_args(ft_mode)
        + get_fully_async_args(fully_async=fully_async)
        + get_api_server_args(config)
        + "--mini-ft-controller-enable "
    )

    base_url = f"http://{config.create_backend().api_server_host(config)}:{API_SERVER_PORT}"
    evidence_dir = evidence_directory(Path(dump_dir))
    injector = spawn_fault_injector(
        tail_policy=tail_policy,
        policy=create_policy(
            expected_cells={
                kind: count
                for kind, count in {"actor": ft_mode.num_cells, "rollout": ft_mode.rollout_num_engines}.items()
                if kind in mean_interval_seconds_of_cell_type
            },
            allow_during_recovery=allow_during_recovery,
            min_survivors=min_survivors,
            max_concurrent_actions=max_concurrent_actions,
        ),
        evidence_path=evidence_dir / "events.jsonl",
        sources={"training_events": Path(dump_dir) / EVENTS_DIRNAME},
        config=config,
        base_url=base_url,
        seed=seed,
        mean_interval_seconds_of_cell_type=mean_interval_seconds_of_cell_type,
        cell_fault_forms=create_cell_fault_forms(base_url=base_url, config=config),
    )

    try:
        run_training(
            injector=injector,
            train_args=train_args,
            mode=ft_mode,
            dump_dir=dump_dir,
            extra_env_vars={},
            config=config,
            train_script=get_train_script(fully_async=fully_async),
        )
    finally:
        injector.stop_and_join(
            teardown=partial(teardown_run, config=config, event_log=injector.event_log, evidence_dir=evidence_dir)
        )

    assert_tail_complete(injector.event_log.events)
    assert_healing(
        ft_mode.ft_components,
        injector=injector,
        event_dir=Path(dump_dir) / EVENTS_DIRNAME,
        context=f"{test_name} {mode}",
    )

    print(f"Random failure soak test PASSED ({test_name}, mode={mode}, seed={seed}, steps={num_steps})")


def assert_mode_supports_fully_async(ft_mode: FTTestMode, *, mode: str) -> None:
    assert ft_mode.has_real_rollout, (
        f"Mode {mode!r} has no rollout engines, so a fully-async soak would train off pre-recorded debug rollout "
        f"data and would prove nothing about generating while training"
    )
    assert (
        not ft_mode.colocate
    ), f"Mode {mode!r} is colocated, which train_async.py rejects: a fully-async run needs engines of its own"


if __name__ == "__main__":
    app()
