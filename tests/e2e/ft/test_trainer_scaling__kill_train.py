# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# Thin CI entry: registers the test and runs the scenario via bare `python3 <file>`
# (the CUDA CI runner's execution model). Scenario logic lives in
# tests/e2e/ft/conftest_ft/scenario_trainer_scaling.py.

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft.scenario_trainer_scaling import run_ci

register_cuda_ci(
    est_time=3600,
    suite="stage-c-8-gpu-h200",
    labels=["ft-long"],
    hardware=["hopper", "blackwell"],
    disabled="needs a Kubernetes cluster backend: the trainer pool is resized as a LeaderWorkerSet",
)

if __name__ == "__main__":
    run_ci()
