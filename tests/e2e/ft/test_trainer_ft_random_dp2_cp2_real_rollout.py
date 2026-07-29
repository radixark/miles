# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# Thin per-mode CI entry: registers the test and runs ONE mode via bare `python3 <file>`
# (the CUDA CI runner's execution model). Scenario logic lives in
# tests/e2e/ft/conftest_ft/scenario_ft_random.py.

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft.scenario_ft_random import run_ci

register_cuda_ci(
    est_time=2400, suite="stage-c-8-gpu-h200", labels=["ft-long"], disabled="FT soak tests pending CI infra support"
)

_MODE: str = "dp2_cp2_real_rollout"

if __name__ == "__main__":
    run_ci(_MODE)
