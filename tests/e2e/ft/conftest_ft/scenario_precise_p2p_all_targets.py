import typer
from tests.e2e.ft.conftest_ft.cli_options import FullyAsyncOption, ModeOption, NumStepsOption, SeedOption
from tests.e2e.ft.conftest_ft.scenario_random_crash import DEFAULT_NUM_STEPS, DEFAULT_SEED
from tests.e2e.ft.conftest_ft.scenario_random_crash import run_ci as run_random_crash

app: typer.Typer = typer.Typer()


@app.command(name="run")
def run_ci(
    mode: ModeOption,
    seed: SeedOption = DEFAULT_SEED,
    num_steps: NumStepsOption = DEFAULT_NUM_STEPS,
    fully_async: FullyAsyncOption = False,
) -> None:
    run_random_crash(
        mode=mode,
        seed=seed,
        num_steps=num_steps,
        fully_async=fully_async,
        precise_p2p=True,
        all_p2p_targets=True,
        min_survivors=2,
        allow_during_recovery=False,
    )


if __name__ == "__main__":
    app()
