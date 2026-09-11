from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft.scenario_random_crash import run_ci

register_cuda_ci(est_time=4800, suite="stage-c-8-gpu-h200", labels=["ft-long", "weight-update"])

if __name__ == "__main__":
    run_ci(mode="kill_rollout__dp2_tp2", precise_p2p=True, mix_wall_clock=True)
