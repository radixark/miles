# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# Thin per-mode CI entry: registers the test and runs ONE mode via bare `python3 <file>`
# (the CUDA CI runner's execution model). Scenario logic lives in
# tests/e2e/ft/conftest_ft/scenario_random_crash.py.

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft.scenario_random_crash import run_ci

register_cuda_ci(
    est_time=7000,
    suite="stage-c-8-gpu-h200",
    labels=["ft-long"],
    disabled=(
        "FT soak tests pending CI infra support: every ft-long entry is disabled for the same reason, so "
        "run-ci-ft-long currently schedules nothing. Unblock condition: an ft-long capable "
        "stage-c-8-gpu-h200 lane, then drop this argument -- nothing in the test itself is known broken. "
        "Until then tests/fast/e2e/ft/test_rollout_gated_recovery.py is the fast-layer stand-in for this entry "
        "(suspend -> gated relaunch -> recovery, no GPU). See tests/e2e/ft/README.md."
    ),
)

_MODE: str = "kill_rollout__dp4__colocate"

if __name__ == "__main__":
    run_ci(_MODE)
