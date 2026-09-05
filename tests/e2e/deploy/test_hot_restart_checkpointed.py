# NOTE: You MUST read tests/e2e/deploy/README.md as source-of-truth and documentations
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.deploy.conftest_deploy.hot_restart.scenario_hot_restart_deterministic import CHECKPOINTED, run_ci

register_cuda_ci(
    est_time=5400,
    suite="stage-c-8-gpu-h200",
    labels=["deploy"],
    disabled="needs a Kubernetes cluster backend",
)

if __name__ == "__main__":
    run_ci(CHECKPOINTED)
