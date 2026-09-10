from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft.scenario_precise_p2p_all_targets import run_ci

register_cuda_ci(est_time=4800, suite="stage-c-8-gpu-h200", labels=["ft-long"])

_MODE: str = "kill_rollout__dp2_tp2"

if __name__ == "__main__":
    run_ci(_MODE)
