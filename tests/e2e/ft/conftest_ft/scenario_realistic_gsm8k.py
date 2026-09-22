# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.


import asyncio

import typer
from tests.e2e.ft.conftest_ft.cli_options import (
    FullyAsyncOption,
    MetricThresholdOption,
    NumRolloutOption,
    RolloutCrashIntervalSecondsOption,
    SeedOption,
    TrainerCrashIntervalSecondsOption,
)
from tests.utils.soak.ft.actions.factory import compute_mean_interval_seconds_of_kind, create_cell_fault_forms
from tests.utils.soak.ft.checkers.healing import assert_healing
from tests.utils.soak.ft.types import ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE
from tests.utils.soak.recipes.gsm8k import (
    CONTEXT_PARALLEL_SIZE,
    DEFAULT_METRIC_THRESHOLD,
    DEFAULT_NUM_ROLLOUT,
    DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS,
    DEFAULT_SEED,
    DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS,
    ROLLOUT_GPUS,
    ROLLOUT_GPUS_PER_ENGINE,
    TRAIN_GPUS,
    run_realistic_gsm8k,
)

from miles.utils.external_utils import command_utils

app: typer.Typer = typer.Typer()

TEST_NAME: str = "realistic_gsm8k"
FT_COMPONENTS: tuple[str, ...] = ("train", "rollout")


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

    outcome = asyncio.run(
        run_realistic_gsm8k(
            config=command_utils.default_config(),
            test_name=test_name,
            seed=seed,
            num_rollout=num_rollout,
            metric_threshold=metric_threshold,
            fully_async=fully_async,
            mean_interval_seconds_of_kind=compute_mean_interval_seconds_of_kind(
                FT_COMPONENTS,
                trainer_crash_interval_seconds=trainer_crash_interval_seconds,
                rollout_crash_interval_seconds=rollout_crash_interval_seconds,
            ),
            expected_counts={
                ACTOR_CELL_TYPE: TRAIN_GPUS // CONTEXT_PARALLEL_SIZE,
                ROLLOUT_CELL_TYPE: ROLLOUT_GPUS // ROLLOUT_GPUS_PER_ENGINE,
            },
            create_forms=lambda run: create_cell_fault_forms(base_url=run.base_url, config=run.launch_spec.config),
            build_extra_train_args=lambda _dump_dir: "",
        )
    )

    assert_healing(
        FT_COMPONENTS,
        events=outcome.injector.event_log.events,
        forms=outcome.forms,
        context=test_name,
    )

    print(f"Random failure gsm8k accuracy test PASSED ({test_name}, seed={seed}, rollouts={num_rollout})")


if __name__ == "__main__":
    app()
