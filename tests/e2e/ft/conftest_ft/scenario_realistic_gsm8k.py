# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.


import typer
from tests.e2e.ft.conftest_ft.cli_options import (
    FullyAsyncOption,
    MetricThresholdOption,
    NumRolloutOption,
    RolloutCrashIntervalSecondsOption,
    SeedOption,
    TrainerCrashIntervalSecondsOption,
)
from tests.utils.soak.checks.ft import assert_healing
from tests.utils.soak.checks.weights import assert_published_weight_checksums
from tests.utils.soak.fault_forms import compute_mean_interval_seconds_of_cell_type, create_cell_fault_forms
from tests.utils.soak.recipes.gsm8k import (
    DEFAULT_METRIC_THRESHOLD,
    DEFAULT_NUM_ROLLOUT,
    DEFAULT_SEED,
    FT_COMPONENTS,
    run_realistic_gsm8k,
)
from tests.utils.soak.state import event_source

from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.external_utils import command_utils

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
    assert_published_weight_checksums(
        read_events(
            event_source(outcome.injector.event_log.events, name="training_events", fallback=outcome.run.events_dir)
        )
    )

    print(f"Random failure gsm8k accuracy test PASSED ({test_name}, seed={seed}, rollouts={num_rollout})")


if __name__ == "__main__":
    app()
