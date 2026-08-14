# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.


import typer
from tests.e2e.ft.conftest_ft import scenario_realistic_gsm8k
from tests.e2e.ft.conftest_ft.cli_options import (
    MetricThresholdOption,
    NumRolloutOption,
    RolloutCrashIntervalSecondsOption,
    SeedOption,
    TrainerCrashIntervalSecondsOption,
)

app: typer.Typer = typer.Typer()


@app.command(name="run")
def run_ci(
    seed: SeedOption = scenario_realistic_gsm8k.DEFAULT_SEED,
    num_rollout: NumRolloutOption = scenario_realistic_gsm8k.DEFAULT_NUM_ROLLOUT,
    trainer_crash_interval_seconds: TrainerCrashIntervalSecondsOption = scenario_realistic_gsm8k.DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS,
    rollout_crash_interval_seconds: RolloutCrashIntervalSecondsOption = scenario_realistic_gsm8k.DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS,
    metric_threshold: MetricThresholdOption = scenario_realistic_gsm8k.DEFAULT_METRIC_THRESHOLD,
) -> None:
    scenario_realistic_gsm8k.run_ci(
        seed=seed,
        num_rollout=num_rollout,
        trainer_crash_interval_seconds=trainer_crash_interval_seconds,
        rollout_crash_interval_seconds=rollout_crash_interval_seconds,
        metric_threshold=metric_threshold,
        fully_async=True,
    )


if __name__ == "__main__":
    app()
