# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.


import asyncio
from collections import Counter
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
from tests.e2e.ft.conftest_ft.fault_injection.entrypoint import FaultInjectorHandle
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import CELL_TYPE_OF_FT_COMPONENT
from tests.e2e.ft.conftest_ft.fault_injection.views import (
    compute_cells_not_serving_after_injection,
    compute_forms_drawn_without_success,
    compute_injected_cell_names,
    compute_num_injections,
    compute_states_of_cell_name,
    compute_successful_form_names,
)
from tests.e2e.ft.conftest_ft.modes import FTTestMode, resolve_mode
from tests.utils.ft.launch import get_fully_async_args, get_train_script
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
from tests.utils.soak.ft.actions.factory import compute_mean_interval_seconds_of_kind
from tests.utils.soak.ft.checkers import healing
from tests.utils.soak.ft.checkers.reconfigure import (
    assert_min_soak_injections,
    assert_soak_reconfigure_events,
    load_reconfigure_events,
)
from tests.utils.soak.ft.entrypoint import run_cell_soak
from tests.utils.soak.ft.types import ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE

from miles.utils.external_utils import command_utils
from miles.utils.workers.naming import parse_cell_id

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

    healing.assert_healing(
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


def assert_healing(
    ft_components: tuple[str, ...], *, injector: FaultInjectorHandle, event_dir: Path, context: str
) -> None:
    events = injector.event_log.events

    _assert_drawn_fault_forms_worked(injector)

    if "train" in ft_components:
        assert_soak_reconfigure_events(
            event_dir, num_successful_injections=compute_num_injections(events, cell_type=ACTOR_CELL_TYPE)
        )
        assert_trainer_injections_healed(injector, event_dir=event_dir)

    if "rollout" in ft_components:
        assert_min_soak_injections(
            compute_num_injections(events, cell_type=ROLLOUT_CELL_TYPE), context=f"{context} rollout cells"
        )
        assert_rollout_cells_served_after_injection(injector)

    _assert_enabled_fault_forms_worked(injector, ft_components=ft_components)


def _assert_drawn_fault_forms_worked(injector: FaultInjectorHandle) -> None:
    never_worked = compute_forms_drawn_without_success(injector.event_log.events)
    assert not never_worked, f"Fault forms drawn but never once successful: {never_worked}"


def _assert_enabled_fault_forms_worked(injector: FaultInjectorHandle, *, ft_components: tuple[str, ...]) -> None:
    events = injector.event_log.events
    never_worked: list[tuple[str, str]] = []
    for component in ft_components:
        cell_type = CELL_TYPE_OF_FT_COMPONENT[component]
        if (forms := injector.cell_fault_forms.get(cell_type)) is None:
            continue
        worked = compute_successful_form_names(events, cell_type=cell_type)
        never_worked += [(cell_type, form.name) for form in forms if form.name not in worked]

    assert not never_worked, f"fault forms this soak enabled but never injected successfully: {sorted(never_worked)}"


def assert_trainer_injections_healed(injector: FaultInjectorHandle, *, event_dir: Path) -> None:
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


def assert_rollout_cells_served_after_injection(injector: FaultInjectorHandle) -> None:
    events = injector.event_log.events
    num_injections: int = compute_num_injections(events, cell_type=ROLLOUT_CELL_TYPE)
    offenders: dict[str, list[str]] = compute_cells_not_serving_after_injection(events, cell_type=ROLLOUT_CELL_TYPE)
    observed: dict[str, list[str]] = {
        name: [state.value for state in states] for name, states in compute_states_of_cell_name(events).items()
    }

    assert not offenders, (
        f"Rollout recovery witness failed: {sorted(offenders)} were never observed healthy and Serving on a "
        f"reading fresh enough to outlast the stale-status window after their last accepted injection, so the "
        f"run may have ended with a permanently missing replica ({num_injections} accepted injection(s); "
        f"observed states: {observed})"
    )

    print(
        f"Rollout recovery witness assertion passed: every injected cell was observed healthy and Serving on a "
        f"fresh reading after its last of {num_injections} accepted injection(s)"
    )


if __name__ == "__main__":
    app()
