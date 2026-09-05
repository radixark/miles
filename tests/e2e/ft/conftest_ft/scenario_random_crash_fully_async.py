# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.


import typer
from tests.e2e.ft.conftest_ft import scenario_random_crash
from tests.e2e.ft.conftest_ft.cli_options import (
    ModeOption,
    NumStepsOption,
    RolloutCrashIntervalSecondsOption,
    SeedOption,
    TrainerCrashIntervalSecondsOption,
)

app: typer.Typer = typer.Typer()


@app.command(name="run")
def run_ci(
    mode: ModeOption,
    seed: SeedOption = scenario_random_crash.DEFAULT_SEED,
    num_steps: NumStepsOption = scenario_random_crash.DEFAULT_NUM_STEPS,
    trainer_crash_interval_seconds: TrainerCrashIntervalSecondsOption = scenario_random_crash.DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS,
    rollout_crash_interval_seconds: RolloutCrashIntervalSecondsOption = scenario_random_crash.DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS,
) -> None:
    scenario_random_crash.run_ci(
        mode,
        seed=seed,
        num_steps=num_steps,
        trainer_crash_interval_seconds=trainer_crash_interval_seconds,
        rollout_crash_interval_seconds=rollout_crash_interval_seconds,
        fully_async=True,
    )


if __name__ == "__main__":
    app()
