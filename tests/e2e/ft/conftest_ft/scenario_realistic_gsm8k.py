# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.


import asyncio
from pathlib import Path

import typer
from tests.e2e.ft.conftest_ft.cli_options import (
    FullyAsyncOption,
    MetricThresholdOption,
    NumRolloutOption,
    RolloutCrashIntervalSecondsOption,
    SeedOption,
    TrainerCrashIntervalSecondsOption,
)
from tests.utils.soak.core.config import SoakRunnerConfig, SoakTailConfig, SoakTargetConfig
from tests.utils.soak.core.utils import compute_base_url
from tests.utils.soak.ft.actions.factory import compute_mean_interval_seconds_of_kind, create_cell_fault_forms
from tests.utils.soak.ft.checkers.healing import assert_healing
from tests.utils.soak.ft.entrypoint import run_cell_soak
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
    execute_gsm8k_session,
    prepare_gsm8k_run,
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

    run = prepare_gsm8k_run(
        config=command_utils.default_config(),
        test_name=test_name,
        seed=seed,
        num_rollout=num_rollout,
        build_extra_train_args=lambda _dump_dir: "",
        metric_threshold=metric_threshold,
        fully_async=fully_async,
    )
    injector = asyncio.run(
        run_cell_soak(
            config=run.launch_spec.config,
            dump_dir=Path(run.dump_dir),
            sut_run=execute_gsm8k_session(run),
            runner_config=_build_runner_config(
                seed=seed,
                num_rollout=num_rollout,
                trainer_crash_interval_seconds=trainer_crash_interval_seconds,
                rollout_crash_interval_seconds=rollout_crash_interval_seconds,
            ),
            event_log=run.event_log,
            evidence_dir=run.evidence_dir,
            cell_fault_forms=create_cell_fault_forms(
                base_url=compute_base_url(run.launch_spec.config), config=run.launch_spec.config
            ),
        )
    )

    assert_healing(
        FT_COMPONENTS,
        events=injector.event_log.events,
        forms=injector.forms,
        context=test_name,
    )

    print(f"Random failure gsm8k accuracy test PASSED ({test_name}, seed={seed}, rollouts={num_rollout})")


def _build_runner_config(
    *,
    seed: int,
    num_rollout: int,
    trainer_crash_interval_seconds: float,
    rollout_crash_interval_seconds: float,
) -> SoakRunnerConfig:
    expected_counts: dict[str, int] = {
        ACTOR_CELL_TYPE: TRAIN_GPUS // CONTEXT_PARALLEL_SIZE,
        ROLLOUT_CELL_TYPE: ROLLOUT_GPUS // ROLLOUT_GPUS_PER_ENGINE,
    }
    mean_intervals = compute_mean_interval_seconds_of_kind(
        FT_COMPONENTS,
        trainer_crash_interval_seconds=trainer_crash_interval_seconds,
        rollout_crash_interval_seconds=rollout_crash_interval_seconds,
    )

    return SoakRunnerConfig(
        seed=seed,
        target_configs={
            kind: SoakTargetConfig(expected_count=expected_counts[kind], mean_interval_seconds=interval)
            for kind, interval in mean_intervals.items()
        },
        tail=SoakTailConfig.create(num_rollout=num_rollout),
    )


if __name__ == "__main__":
    app()
