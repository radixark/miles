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
from tests.e2e.ft.conftest_ft.fault_injection import CONTROL_SERVER_PORT, MEAN_INTERVAL_SECONDS, spawn_fault_injector
from tests.e2e.ft.conftest_ft.modes import FTTestMode, resolve_mode

from miles.utils.test_utils.reconfigure_assertions import assert_soak_reconfigure_events

app: typer.Typer = typer.Typer()

TEST_NAME: str = "trainer_ft_random"


@app.command(name="run")
def run_ci(
    mode: Annotated[str, typer.Option(help="Test mode variant")],
    seed: Annotated[int, typer.Option(help="Random seed for fault injection")] = 42,
    num_steps: Annotated[int, typer.Option(help="Number of train() calls")] = 30,
    crash_probability: Annotated[float, typer.Option(help="Per-step crash probability per cell")] = 0.1,
) -> None:
    """Random failure soak test.

    Starts a background thread that injects faults at random intervals via the
    control server HTTP API. The mini FT controller auto-recovers; the test passes
    if training completes without hanging.

    Doubles as the per-mode CI entry point: a CI file calls ``run_ci(mode)`` (defaults);
    manual runs use the ``run`` CLI subcommand with optional --seed/--num-steps/etc.
    """
    ft_mode: FTTestMode = resolve_mode(mode)
    dump_dir: str = resolve_dump_dir(f"{TEST_NAME}_{mode}")
    print(f"Dump directory: {dump_dir}")
    mean_interval: float = MEAN_INTERVAL_SECONDS / max(crash_probability, 0.01)
    print(f"Seed: {seed}, Steps: {num_steps}, Mean injection interval: {mean_interval:.1f}s")

    prepare(ft_mode)

    # The recorded debug rollouts are fewer than the soak's step count; symlink them cyclically
    # into a temp dir so each rollout_id has a file, keeping the production load path unchanged.
    cyclic_data_dir = materialize_cyclic_debug_rollout_data(num_steps)
    train_args = (
        get_common_train_args(ft_mode, dump_dir=dump_dir, num_steps=num_steps, debug_rollout_data_dir=cyclic_data_dir)
        + get_ft_args(ft_mode)
        + f"--control-server-port {CONTROL_SERVER_PORT} "
        + "--mini-ft-controller-enable "
    )

    injector = spawn_fault_injector(seed=seed, mean_interval_seconds=mean_interval)

    try:
        run_training(train_args=train_args, mode=ft_mode, dump_dir=dump_dir)
    finally:
        injector.stop_and_join(timeout_seconds=5)

    assert_soak_reconfigure_events(
        Path(dump_dir) / "events",
        num_successful_injections=injector.num_successful_injections,
    )

    print(f"Random failure soak test PASSED (seed={seed}, steps={num_steps})")


if __name__ == "__main__":
    app()
