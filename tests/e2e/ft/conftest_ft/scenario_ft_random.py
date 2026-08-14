# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.


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
from tests.e2e.ft.conftest_ft.fault_injection import (
    API_SERVER_PORT,
    MEAN_INTERVAL_SECONDS,
    FaultInjectorHandle,
    spawn_fault_injector,
)
from tests.e2e.ft.conftest_ft.modes import FTTestMode, resolve_mode

from miles.utils.test_utils.reconfigure_assertions import assert_min_soak_injections, assert_soak_reconfigure_events

app: typer.Typer = typer.Typer()

TEST_NAME: str = "ft_random"


@app.command(name="run")
def run_ci(
    mode: Annotated[str, typer.Option(help="Test mode variant")],
    seed: Annotated[int, typer.Option(help="Random seed for fault injection")] = 42,
    num_steps: Annotated[int, typer.Option(help="Number of train() calls")] = 30,
    crash_probability: Annotated[float, typer.Option(help="Per-step crash probability per cell")] = 0.5,
) -> None:
    """Random failure soak test, for whichever components the mode enables ft on.

    Starts a background thread that injects faults at random intervals via the
    api server HTTP API. The mini FT controller auto-recovers; the test passes
    if training completes without hanging.

    Doubles as the per-mode CI entry point: a CI file calls ``run_ci(mode)`` (defaults);
    manual runs use the ``run`` CLI subcommand with optional --seed/--num-steps/etc.
    """
    ft_mode: FTTestMode = resolve_mode(mode)
    dump_dir: str = resolve_dump_dir(f"{TEST_NAME}_{mode}")
    print(f"Dump directory: {dump_dir}")
    mean_interval: float = MEAN_INTERVAL_SECONDS / max(crash_probability, 0.01)
    print(f"Seed: {seed}, Steps: {num_steps}, Mean injection interval: {mean_interval:.1f}s")
    print(f"FT components: {ft_mode.ft_components}")

    prepare(ft_mode)

    debug_rollout_data_dir = None if ft_mode.has_real_rollout else materialize_cyclic_debug_rollout_data(num_steps)
    train_args = (
        get_common_train_args(
            ft_mode, dump_dir=dump_dir, num_steps=num_steps, debug_rollout_data_dir=debug_rollout_data_dir
        )
        + get_ft_args(ft_mode)
        + f"--api-server-port {API_SERVER_PORT} "
        + "--mini-ft-controller-enable "
    )

    injector = spawn_fault_injector(
        seed=seed, mean_interval_seconds=mean_interval, cell_type=compute_injected_cell_type(ft_mode)
    )

    try:
        run_training(train_args=train_args, mode=ft_mode, dump_dir=dump_dir)
    finally:
        injector.stop_and_join(timeout_seconds=5)

    assert_healing(ft_mode, injector=injector, dump_dir=dump_dir)

    print(f"Random failure soak test PASSED (mode={mode}, seed={seed}, steps={num_steps})")


def compute_injected_cell_type(ft_mode: FTTestMode) -> str | None:
    match tuple(sorted(ft_mode.ft_components)):
        case ("train",):
            return "actor"
        case ("rollout",):
            return "rollout"
        case _:
            return None


def assert_healing(ft_mode: FTTestMode, *, injector: FaultInjectorHandle, dump_dir: str) -> None:
    assert_min_soak_injections(injector.num_successful_injections, context=f"{TEST_NAME} {ft_mode.ft_components}")

    if "train" in ft_mode.ft_components:
        assert_soak_reconfigure_events(
            Path(dump_dir) / "events",
            num_successful_injections=injector.num_successful_injections,
        )

    if "rollout" in ft_mode.ft_components:
        assert_rollout_healed_through_pending(injector)


def assert_rollout_healed_through_pending(injector: FaultInjectorHandle) -> None:
    history = injector.phase_history
    outcomes = history.injection_outcomes(cell_type="rollout")
    recovered = [outcome for outcome in outcomes if outcome.recovered]
    unexplained = [outcome for outcome in outcomes if not outcome.recovered and not outcome.still_down]

    assert not unexplained, (
        f"Rollout healing witness failed: {len(unexplained)} injection(s) were followed by no "
        f"Running -> Pending -> Running of the injected cell, yet the cell ended up Running "
        f"({unexplained}; observed phases: {history.phases_of_cell_name})"
    )
    assert len(recovered) >= MIN_SOAK_HEALINGS, (
        f"Rollout healing witness failed: only {len(recovered)} of {len(outcomes)} rollout injection(s) "
        f"were paired with a Running -> Pending -> Running of the injected cell, need >= {MIN_SOAK_HEALINGS} "
        f"(observed phases: {history.phases_of_cell_name})"
    )

    print(
        f"Rollout healing witness assertion passed: {len(recovered)} of {len(outcomes)} injection(s) "
        f"were each paired with a Running -> Pending -> Running of the injected cell"
    )


if __name__ == "__main__":
    app()
