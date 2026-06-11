# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import logging
import random
import threading
import time
from typing import Annotated

import requests
import typer

from tests.e2e.ft.conftest_ft.app import resolve_dump_dir
from tests.e2e.ft.conftest_ft.execution import (
    get_common_train_args,
    get_ft_args,
    materialize_cyclic_debug_rollout_data,
    prepare,
    run_training,
)
from tests.e2e.ft.conftest_ft.modes import FTTestMode, resolve_mode

from miles.utils.test_utils.fault_injector import FailureMode

logger = logging.getLogger(__name__)

app: typer.Typer = typer.Typer()

TEST_NAME: str = "trainer_ft_random"

_CONTROL_SERVER_PORT: int = 18080
_MEAN_INTERVAL_SECONDS: float = 60.0
# Hard floor between consecutive injections so the FT controller has time to
# spawn the replacement actor and let it rejoin before the next crash. Without
# this, the exponential delay can produce several injections within a few
# seconds, causing the all-cells-dead cascade.
_MIN_GAP_BETWEEN_INJECTIONS_SECONDS: float = 30.0
_FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL, FailureMode.EXIT, FailureMode.SEGFAULT]


def _run_fault_injection_loop(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds: float,
    stop_event: threading.Event,
) -> None:
    rng = random.Random(seed)
    last_injection_at: float = 0.0

    while not stop_event.is_set():
        delay = rng.expovariate(1.0 / mean_interval_seconds)
        if stop_event.wait(timeout=delay):
            break

        elapsed = time.monotonic() - last_injection_at
        if elapsed < _MIN_GAP_BETWEEN_INJECTIONS_SECONDS:
            logger.info(
                "Skipping injection: only %.1fs since last, need %.1fs",
                elapsed,
                _MIN_GAP_BETWEEN_INJECTIONS_SECONDS,
            )
            continue

        try:
            resp = requests.get(f"{base_url}/api/v1/cells", timeout=5)
            resp.raise_for_status()
            cells = resp.json()["items"]
        except Exception:
            logger.info("Failed to list cells from control server", exc_info=True)
            continue

        # A cell is "alive" iff its Healthy condition is TRUE. Note: phase=="Running"
        # is also true for StateAllocatedErrored (cell crashed mid-step but not yet
        # cleaned up), so phase alone is too permissive.
        def _is_alive(cell: dict) -> bool:
            return any(cond["type"] == "Healthy" and cond["status"] == "True" for cond in cell["status"]["conditions"])

        alive = [c for c in cells if _is_alive(c)]
        # Skip injection only when killing one more would leave us with no
        # redundancy left (≤1 alive). Otherwise inject — even if some peers
        # are still mid-recovery, we tolerate further reductions because dp
        # still has spare cells.
        if len(alive) <= 1:
            logger.info(
                "Skipping injection: %d/%d cells alive (need >1 to keep redundancy)",
                len(alive),
                len(cells),
            )
            continue

        target = rng.choice(alive)
        cell_name = target["metadata"]["name"]
        mode = rng.choice(_FAILURE_MODES)

        try:
            resp = requests.post(
                f"{base_url}/api/v1/cells/{cell_name}/inject-fault",
                json={"mode": mode.value, "sub_index": 0},
                timeout=5,
            )
            resp.raise_for_status()
            last_injection_at = time.monotonic()
        except Exception:
            logger.info("Failed to inject fault into %s", cell_name, exc_info=True)


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
    mean_interval: float = _MEAN_INTERVAL_SECONDS / max(crash_probability, 0.01)
    print(f"Seed: {seed}, Steps: {num_steps}, Mean injection interval: {mean_interval:.1f}s")

    prepare(ft_mode)

    # The recorded debug rollouts are fewer than the soak's step count; symlink them cyclically
    # into a temp dir so each rollout_id has a file, keeping the production load path unchanged.
    cyclic_data_dir = materialize_cyclic_debug_rollout_data(num_steps)
    train_args = (
        get_common_train_args(ft_mode, dump_dir=dump_dir, num_steps=num_steps, debug_rollout_data_dir=cyclic_data_dir)
        + get_ft_args(ft_mode)
        + f"--control-server-port {_CONTROL_SERVER_PORT} "
        + "--mini-ft-controller-enable "
    )

    base_url = f"http://localhost:{_CONTROL_SERVER_PORT}"
    stop_event = threading.Event()
    injector_thread = threading.Thread(
        target=_run_fault_injection_loop,
        kwargs={"base_url": base_url, "seed": seed, "mean_interval_seconds": mean_interval, "stop_event": stop_event},
        daemon=True,
        name="ft-random-fault-injector",
    )
    injector_thread.start()

    try:
        run_training(train_args=train_args, mode=ft_mode)
    finally:
        stop_event.set()
        injector_thread.join(timeout=5)

    print(f"Random failure soak test PASSED (seed={seed}, steps={num_steps})")


if __name__ == "__main__":
    app()
