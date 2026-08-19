# NOTE: You MUST read tests/e2e/deploy/README.md as source-of-truth and documentations
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.deploy.conftest_deploy.split.scenario_split_multi_policy import run_ci

register_cuda_ci(
    est_time=2400,
    suite="stage-c-4-gpu-h200",
    labels=["deploy", "multi-policy", "fully-async"],
    disabled="needs a Kubernetes cluster backend",
)

if __name__ == "__main__":
    run_ci()
