import dataclasses
import os

from examples.multi_policy.run_solver_verifier_gsm8k import SOLVER_MODEL_ID, VERIFIER_MODEL_ID, ScriptArgs, prepare
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.conftest_multi_policy import TrainRewardBounds, execute

from miles.utils.external_utils import command_utils

register_cuda_ci(est_time=5400, suite="stage-c-4-gpu-h200", labels=["long"])

NUM_ROLLOUT = int(os.environ.get("MILES_TEST_NUM_ROLLOUT", "100"))

# TODO: tighten these weak bounds once the e2e run has been observed.
TRAIN_REWARD_BOUNDS = {
    SOLVER_MODEL_ID: TrainRewardBounds(initial_max=0.6, final_min=0.5, min_growth=0.2),
    VERIFIER_MODEL_ID: TrainRewardBounds(initial_max=0.9, final_min=0.1, min_growth=0.2),
}


if __name__ == "__main__":
    args = dataclasses.replace(command_utils.default_config(ScriptArgs), num_rollout=NUM_ROLLOUT)
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(
        args,
        wandb_args=command_utils.get_default_wandb_args(__file__),
        train_reward_bounds=TRAIN_REWARD_BOUNDS,
    )
