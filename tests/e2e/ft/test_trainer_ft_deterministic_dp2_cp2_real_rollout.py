# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# Thin per-mode CI entry: registers the test and runs ONE mode via bare `python3 <file>`
# (the CUDA CI runner's execution model). Scenario logic lives in
# tests/e2e/ft/conftest_ft/scenario_deterministic.py.

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft.scenario_deterministic import run_ci

register_cuda_ci(
    est_time=1500,
    suite="stage-c-8-gpu-h100",
    labels=["ft-short"],
    disabled="real_rollout deterministic-inference sglang hang on current image",
)

_MODE: str = "dp2_cp2_real_rollout"

if __name__ == "__main__":
    run_ci(_MODE)
