# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import typer
from tests.e2e.ft.conftest_ft.app import resolve_dump_dir
from tests.e2e.ft.conftest_ft.cli_options import (
    FullyAsyncOption,
    MetricThresholdOption,
    NumRolloutOption,
    RolloutCrashIntervalSecondsOption,
    SeedOption,
    TrainerCrashIntervalSecondsOption,
)
from tests.e2e.ft.conftest_ft.execution import (
    DATA_DIR,
    MODEL_DIR,
    get_api_server_args,
    get_fully_async_args,
    get_train_script,
)
from tests.e2e.ft.conftest_ft.scenario_random_crash import assert_healing
from tests.fast.cluster_backends import create_backend_for_run
from tests.utils.soak.entrypoint import API_SERVER_PORT, FaultInjectorHandle, spawn_fault_injector
from tests.utils.soak.fault_forms import (
    CellFaultForms,
    compute_mean_interval_seconds_of_cell_type,
    create_cell_fault_forms,
)

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend

app: typer.Typer = typer.Typer()

TEST_NAME: str = "realistic_gsm8k"

DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS: float = 600.0
DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS: float = 1200.0

@app.command(name="run")
def run_ci(
    seed: SeedOption = DEFAULT_SEED,
    num_rollout: NumRolloutOption = DEFAULT_NUM_ROLLOUT,
    trainer_crash_interval_seconds: TrainerCrashIntervalSecondsOption = DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS,
    rollout_crash_interval_seconds: RolloutCrashIntervalSecondsOption = DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS,
    metric_threshold: MetricThresholdOption = DEFAULT_METRIC_THRESHOLD,
    fully_async: FullyAsyncOption = False,
) -> None:
    test_name: str = f"{TEST_NAME}_fully_async" if fully_async else TEST_NAME
    outcome = run_realistic_gsm8k(
        config=command_utils.default_config(),
        test_name=test_name,
        seed=seed,
        num_rollout=num_rollout,
        metric_threshold=metric_threshold,
        fully_async=fully_async,
        mean_interval_seconds_of_cell_type=compute_mean_interval_seconds_of_cell_type(
            FT_COMPONENTS,
            trainer_crash_interval_seconds=trainer_crash_interval_seconds,
            rollout_crash_interval_seconds=rollout_crash_interval_seconds,
        ),
        create_forms=lambda run: create_cell_fault_forms(base_url=run.base_url, config=run.config),
        build_extra_train_args=lambda _dump_dir: "",
    )

    assert_healing(FT_COMPONENTS, injector=outcome.injector, event_dir=outcome.run.events_dir, context=test_name)

    print(f"Random failure gsm8k accuracy test PASSED ({test_name}, seed={seed}, rollouts={num_rollout})")


if __name__ == "__main__":
    app()
