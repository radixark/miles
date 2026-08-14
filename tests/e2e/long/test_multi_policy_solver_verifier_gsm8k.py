import os

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.short.test_multi_policy_solver_verifier_gsm8k import (
    SOLVER_MODEL_ID,
    VERIFIER_MODEL_ID,
    TrainRewardBounds,
    execute,
    prepare,
)

register_cuda_ci(est_time=5400, suite="stage-c-8-gpu-h100", labels=["long"])

NUM_ROLLOUT = int(os.environ.get("MILES_TEST_NUM_ROLLOUT", "100"))

# TODO: tighten these weak bounds once the e2e run has been observed.
TRAIN_REWARD_BOUNDS = {
    SOLVER_MODEL_ID: TrainRewardBounds(initial_max=0.6, final_min=0.5),
    VERIFIER_MODEL_ID: TrainRewardBounds(initial_max=0.9, final_min=0.1),
}


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(num_rollout=NUM_ROLLOUT, train_reward_bounds=TRAIN_REWARD_BOUNDS)
