# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.


from collections import Counter
from pathlib import Path

import typer
from tests.e2e.ft.conftest_ft.app import resolve_dump_dir
from tests.e2e.ft.conftest_ft.cli_options import (
    FullyAsyncOption,
    ModeOption,
    NumStepsOption,
    RolloutCrashIntervalSecondsOption,
    SeedOption,
    TrainerCrashIntervalSecondsOption,
)
from tests.e2e.ft.conftest_ft.execution import (
    get_api_server_args,
    get_common_train_args,
    get_ft_args,
    get_fully_async_args,
    get_train_script,
    materialize_cyclic_debug_rollout_data,
    prepare,
    run_training,
)
from tests.e2e.ft.conftest_ft.fault_injection.entrypoint import (
    API_SERVER_PORT,
    FaultInjectorHandle,
    spawn_fault_injector,
)
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import (
    ACTOR_CELL_TYPE,
    ROLLOUT_CELL_TYPE,
    compute_mean_interval_seconds_of_cell_type,
    create_cell_fault_forms,
)
from tests.e2e.ft.conftest_ft.fault_injection.views import (
    compute_cells_not_seen_serving_after_last_injection,
    compute_forms_drawn_but_never_successful,
    compute_injected_cell_names,
    compute_num_injections,
    compute_states_of_cell_name,
)
from tests.e2e.ft.conftest_ft.modes import FTTestMode, resolve_mode

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.external_utils import command_utils
from miles.utils.test_utils.comparisons.metrics import read_rollout_completion_times
from miles.utils.test_utils.reconfigure_assertions import (
    assert_min_soak_injections,
    assert_soak_reconfigure_events,
    load_reconfigure_events,
)
from miles.utils.workers.naming import parse_cell_id

app: typer.Typer = typer.Typer()

TEST_NAME: str = "random_crash"

DEFAULT_SEED: int = 42
DEFAULT_NUM_STEPS: int = 100
DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS: float = 120.0
DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS: float = 240.0
TERMINAL_FAULT_FREE_STEPS: int = 6


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

    Starts a background thread that injects faults at random intervals via the
    api server HTTP API. The mini FT controller auto-recovers; the test passes
    if training completes without hanging.

    Doubles as the per-mode CI entry point: a CI file calls ``run_ci(mode)`` (defaults);
    manual runs use the ``run`` CLI subcommand with optional --seed/--num-steps/etc.
    """
    ft_mode: FTTestMode = resolve_mode(mode)
    if fully_async:
        assert_mode_supports_fully_async(ft_mode, mode=mode)

    config = command_utils.default_config()
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
    injector = spawn_fault_injector(
        base_url=base_url,
        seed=seed,
        mean_interval_seconds_of_cell_type=mean_interval_seconds_of_cell_type,
        cell_fault_forms=create_cell_fault_forms(base_url=base_url, config=config),
        injection_enabled=lambda: _fault_injection_enabled(dump_dir, num_steps=num_steps),
    )

    try:
        run_training(
            train_args=train_args,
            mode=ft_mode,
            dump_dir=dump_dir,
            extra_env_vars={},
            config=config,
            train_script=get_train_script(fully_async=fully_async),
        )
    finally:
        injector.stop_and_join()

    assert_healing(
        ft_mode.ft_components,
        injector=injector,
        event_dir=Path(dump_dir) / EVENTS_DIRNAME,
        context=f"{test_name} {mode}",
    )

    print(f"Random failure soak test PASSED ({test_name}, mode={mode}, seed={seed}, steps={num_steps})")


def _fault_injection_enabled(dump_dir: str, *, num_steps: int) -> bool:
    completed_rollout_ids: set[int] = {rollout_id for rollout_id, _ in read_rollout_completion_times(dump_dir)}
    next_rollout_id: int = max(completed_rollout_ids, default=-1) + 1
    return next_rollout_id < num_steps - TERMINAL_FAULT_FREE_STEPS


def assert_mode_supports_fully_async(ft_mode: FTTestMode, *, mode: str) -> None:
    assert ft_mode.has_real_rollout, (
        f"Mode {mode!r} has no rollout engines, so a fully-async soak would train off pre-recorded debug rollout "
        f"data and would prove nothing about generating while training"
    )
    assert (
        not ft_mode.colocate
    ), f"Mode {mode!r} is colocated, which train_async.py rejects: a fully-async run needs engines of its own"


def assert_healing(
    ft_components: tuple[str, ...], *, injector: FaultInjectorHandle, event_dir: Path, context: str
) -> None:
    events = injector.event_log.events

    _assert_every_drawn_fault_form_worked(injector)

    if "train" in ft_components:
        assert_soak_reconfigure_events(
            event_dir, num_successful_injections=compute_num_injections(events, cell_type=ACTOR_CELL_TYPE)
        )
        assert_every_trainer_injection_healed(injector, event_dir=event_dir)

    if "rollout" in ft_components:
        assert_min_soak_injections(
            compute_num_injections(events, cell_type=ROLLOUT_CELL_TYPE), context=f"{context} rollout cells"
        )
        assert_every_rollout_cell_served_after_its_last_injection(injector)


def _assert_every_drawn_fault_form_worked(injector: FaultInjectorHandle) -> None:
    never_worked = compute_forms_drawn_but_never_successful(injector.event_log.events)
    assert not never_worked, f"Fault forms drawn but never once successful: {never_worked}"


def assert_every_trainer_injection_healed(injector: FaultInjectorHandle, *, event_dir: Path) -> None:
    injected: Counter[int] = Counter(
        parse_cell_id(name).cell_index
        for name in compute_injected_cell_names(injector.event_log.events, cell_type=ACTOR_CELL_TYPE)
    )
    healed: Counter[int] = Counter(
        cell_index for event in load_reconfigure_events(event_dir) for cell_index in event.healed_cell_indices
    )
    debt: Counter[int] = injected - healed

    assert not debt, (
        f"Trainer recovery witness failed: cell index -> accepted injection(s) never healed {dict(debt)} when "
        f"training ended (injected {dict(injected)}, healed {dict(healed)} across the events in {event_dir})"
    )

    print(
        f"Trainer recovery witness assertion passed: every one of {sum(injected.values())} accepted injection(s) "
        f"is paired with a healing of the same cell ({dict(healed)})"
    )


def assert_every_rollout_cell_served_after_its_last_injection(injector: FaultInjectorHandle) -> None:
    events = injector.event_log.events
    num_injections: int = compute_num_injections(events, cell_type=ROLLOUT_CELL_TYPE)
    offenders: dict[str, list[str]] = compute_cells_not_seen_serving_after_last_injection(
        events, cell_type=ROLLOUT_CELL_TYPE
    )
    observed: dict[str, list[str]] = {
        name: [state.value for state in states] for name, states in compute_states_of_cell_name(events).items()
    }

    assert not offenders, (
        f"Rollout recovery witness failed: {sorted(offenders)} were never observed Serving on a reading fresh "
        f"enough to outlast the stale-status window after their last accepted injection, so the run may have "
        f"ended with a permanently missing replica ({num_injections} accepted injection(s); observed states: "
        f"{observed})"
    )

    print(
        f"Rollout recovery witness assertion passed: every injected cell was observed Serving on a fresh "
        f"reading after its last of {num_injections} accepted injection(s)"
    )


if __name__ == "__main__":
    app()
