# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# Thin per-mode CI entry: registers the test and runs ONE mode via bare `python3 <file>`
# (the CUDA CI runner's execution model). Scenario logic lives in
# tests/e2e/ft/conftest_ft/scenario_ft_random.py.

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft.scenario_ft_random import run_ci

register_cuda_ci(
    est_time=2400,
    suite="stage-c-8-gpu-h200",
    labels=["ft-long"],
    disabled=(
        "FT soak tests pending CI infra support: every ft-long entry is disabled for the same reason, and the "
        "specific infra gap is not recorded anywhere in the repo. Unblock condition: an ft-long capable "
        "stage-c-8-gpu-h200 lane, then drop this argument -- nothing in the test itself is known broken. "
        "Until then tests/fast/e2e/ft/test_rollout_ft_gated_recovery.py is the fast-layer stand-in for this entry "
        "(suspend -> gated relaunch -> recovery, no GPU). See tests/e2e/ft/README.md."
    ),
)

_MODE: str = "colocate_dp2_cp2_rollout_ft"

if __name__ == "__main__":
    run_ci(_MODE)
