# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.


from tests.e2e.ft.conftest_ft.cli_options import PreciseHook
from tests.e2e.ft.conftest_ft.scenario_random_crash import create_precise_app

app, run_ci = create_precise_app(PreciseHook.ALL_GATHER, mix=False)

if __name__ == "__main__":
    app()
