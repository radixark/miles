# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.


from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from tests.e2e.ft.conftest_ft.app import resolve_dump_dir
from tests.e2e.ft.conftest_ft.execution import (
    get_common_train_args,
    get_ft_args,
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
    create_cell_fault_forms,
)
from tests.e2e.ft.conftest_ft.fault_injection.views import (
    compute_cells_with_unfinished_recovery,
    compute_forms_drawn_without_success,
    compute_injected_cell_names,
    compute_num_completed_recoveries,
    compute_num_injections,
    compute_states_of_cell_name,
)
from tests.e2e.ft.conftest_ft.modes import FTTestMode, resolve_mode

from miles.utils.external_utils import command_utils
from miles.utils.test_utils.reconfigure_assertions import (
    assert_min_soak_injections,
    assert_soak_reconfigure_events,
    load_reconfigure_events,
)
from miles.utils.workers.naming import parse_cell_id

app: typer.Typer = typer.Typer()

TEST_NAME: str = "random_crash"

DEFAULT_CRASH_INTERVAL_SECONDS: float = 120.0


@app.command(name="run")
def run_ci(
    mode: Annotated[str, typer.Option(help="Test mode variant")],
    seed: Annotated[int, typer.Option(help="Random seed for fault injection")] = 42,
    num_steps: Annotated[int, typer.Option(help="Number of train() calls")] = 30,
    crash_interval_seconds: Annotated[
        float, typer.Option(help="Mean seconds between injections, shared out across the mode's ft components")
    ] = DEFAULT_CRASH_INTERVAL_SECONDS,
) -> None:
    """Random failure soak test, for whichever components the mode enables ft on.

    Starts a background thread that injects faults at random intervals via the
    api server HTTP API. The mini FT controller auto-recovers; the test passes
    if training completes without hanging.

    Doubles as the per-mode CI entry point: a CI file calls ``run_ci(mode)`` (defaults);
    manual runs use the ``run`` CLI subcommand with optional --seed/--num-steps/etc.
    """
    ft_mode: FTTestMode = resolve_mode(mode)
    config = command_utils.default_config()
    dump_dir: str = resolve_dump_dir(f"{TEST_NAME}_{mode}")
    print(f"Dump directory: {dump_dir}")
    mean_interval: float = crash_interval_seconds / len(ft_mode.ft_components)
    print(f"Seed: {seed}, Steps: {num_steps}, Mean injection interval: {mean_interval:.1f}s")
    print(f"FT components: {ft_mode.ft_components}, cluster backend: {config.cluster_backend.value}")

    prepare(ft_mode, config=config)

    debug_rollout_data_dir = None if ft_mode.has_real_rollout else materialize_cyclic_debug_rollout_data(num_steps)
    train_args = (
        get_common_train_args(
            ft_mode, dump_dir=dump_dir, num_steps=num_steps, debug_rollout_data_dir=debug_rollout_data_dir
        )
        + get_ft_args(ft_mode)
        + f"--api-server-port {API_SERVER_PORT} "
        + "--mini-ft-controller-enable "
    )

    base_url = f"http://{config.create_backend().api_server_host()}:{API_SERVER_PORT}"
    injector = spawn_fault_injector(
        base_url=base_url,
        seed=seed,
        mean_interval_seconds=mean_interval,
        cell_type=compute_injected_cell_type(ft_mode),
        cell_fault_forms=create_cell_fault_forms(base_url=base_url, config=config),
    )

    try:
        run_training(train_args=train_args, mode=ft_mode, dump_dir=dump_dir, config=config)
    finally:
        injector.stop_and_join()

    assert_healing(ft_mode, injector=injector, dump_dir=dump_dir)

    print(f"Random failure soak test PASSED (mode={mode}, seed={seed}, steps={num_steps})")


def compute_injected_cell_type(ft_mode: FTTestMode) -> str | None:
    match tuple(sorted(ft_mode.ft_components)):
        case ("train",):
            return ACTOR_CELL_TYPE
        case ("rollout",):
            return ROLLOUT_CELL_TYPE
        case _:
            return None


def assert_healing(ft_mode: FTTestMode, *, injector: FaultInjectorHandle, dump_dir: str) -> None:
    events = injector.event_log.events
    event_dir = Path(dump_dir) / "events"

    _assert_drawn_fault_forms_worked(injector)

    if "train" in ft_mode.ft_components:
        assert_soak_reconfigure_events(
            event_dir, num_successful_injections=compute_num_injections(events, cell_type=ACTOR_CELL_TYPE)
        )
        assert_trainer_injections_healed(injector, event_dir=event_dir)

    if "rollout" in ft_mode.ft_components:
        assert_min_soak_injections(
            compute_num_injections(events, cell_type=ROLLOUT_CELL_TYPE), context=f"{TEST_NAME} rollout cells"
        )
        assert_every_rollout_injection_recovered(injector)


def _assert_drawn_fault_forms_worked(injector: FaultInjectorHandle) -> None:
    never_worked = compute_forms_drawn_without_success(injector.event_log.events)
    assert not never_worked, f"Fault forms drawn but never once successful: {never_worked}"


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


def assert_every_rollout_injection_recovered(injector: FaultInjectorHandle) -> None:
    events = injector.event_log.events
    num_injections: int = compute_num_injections(events, cell_type="rollout")
    num_recoveries: int = compute_num_completed_recoveries(events, cell_type="rollout")
    unfinished: dict[str, int] = compute_cells_with_unfinished_recovery(events, cell_type="rollout")
    observed: dict[str, list[str]] = {
        name: [state.value for state in states] for name, states in compute_states_of_cell_name(events).items()
    }

    assert not unfinished, (
        f"Rollout recovery witness failed: {unfinished} still had an accepted injection with no completed "
        f"relaunch-and-serve cycle when training ended ({num_recoveries}/{num_injections} injection(s) "
        f"recovered; observed states: {observed})"
    )
    assert num_recoveries >= num_injections, (
        f"Rollout recovery witness failed: only {num_recoveries} completed recovery(ies) for "
        f"{num_injections} accepted injection(s) (observed states: {observed})"
    )

    print(
        f"Rollout recovery witness assertion passed: {num_recoveries} completed relaunch-and-serve cycle(s) "
        f"for {num_injections} accepted injection(s)"
    )


if __name__ == "__main__":
    app()
