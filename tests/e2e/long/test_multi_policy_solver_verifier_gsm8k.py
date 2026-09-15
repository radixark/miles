import dataclasses
import os

from examples.multi_policy.run_solver_verifier_gsm8k import SOLVER_MODEL_ID, VERIFIER_MODEL_ID, ScriptArgs, prepare
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.conftest_multi_policy import EvalScoreBounds, execute

from miles.utils.external_utils import command_utils

register_cuda_ci(est_time=36000, suite="stage-c-4-gpu-h200", labels=["long"])

NUM_ROLLOUT = int(os.environ.get("MILES_TEST_NUM_ROLLOUT", "250"))

# Calibrated against a full 250-rollout run of this recipe: eval/gsm8k/solver
# rose .473 -> .566 (first -> best point) and eval/gsm8k/verifier .569 -> .821;
# thresholds sit at roughly one third of the observed growth.
EVAL_SCORE_BOUNDS = {
    SOLVER_MODEL_ID: EvalScoreBounds(initial_max=0.52, peak_min=0.53, min_growth=0.03),
    VERIFIER_MODEL_ID: EvalScoreBounds(initial_max=0.65, peak_min=0.70, min_growth=0.08),
}


if __name__ == "__main__":
    args = dataclasses.replace(command_utils.default_config(ScriptArgs), num_rollout=NUM_ROLLOUT)
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(
        args,
        wandb_args=command_utils.get_default_wandb_args(__file__),
        eval_score_bounds=EVAL_SCORE_BOUNDS,
    )
