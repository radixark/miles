# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# Thin CI entry: registers the test and runs it via bare `python3 <file>`
# (the CUDA CI runner's execution model). Scenario logic lives in
# tests/e2e/ft/conftest_ft/scenario_realistic_gsm8k.py.

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft.scenario_realistic_gsm8k import run_ci

register_cuda_ci(
    est_time=9000,
    suite="stage-c-8-gpu-h200",
    labels=["ft-long", "long"],
    disabled="FT soak tests pending CI infra support",
)

if __name__ == "__main__":
    run_ci()
