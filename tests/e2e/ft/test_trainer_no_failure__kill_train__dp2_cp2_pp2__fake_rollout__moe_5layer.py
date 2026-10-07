# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# Thin per-mode CI entry: registers the test and runs ONE mode via bare `python3 <file>`
# (the CUDA CI runner's execution model). Scenario logic lives in
# tests/e2e/ft/conftest_ft/scenario_trainer_no_failure.py.

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.e2e.ft.conftest_ft.scenario_trainer_no_failure import run_ci

register_cuda_ci(
    est_time=800,
    suite="stage-c-8-gpu-h100",
    labels=["ft-short"],
    hardware=["hopper", "blackwell"],
)
register_rocm_ci(est_time=900, suite="nightly-stage-c-8-gpu-mi350", labels=["ft-short"])

_MODE: str = "kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer"

if __name__ == "__main__":
    run_ci(_MODE)
