import dataclasses
import os

from examples.multi_policy.run_solver_verifier_gsm8k import ScriptArgs, prepare
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.conftest_multi_policy import execute

from miles.utils.external_utils import command_utils

register_cuda_ci(est_time=900, suite="stage-c-8-gpu-h100", labels=["short", "multi-policy", "fully-async"])

NUM_ROLLOUT = int(os.environ.get("MILES_TEST_NUM_ROLLOUT", "3"))


if __name__ == "__main__":
    args = dataclasses.replace(command_utils.default_config(ScriptArgs), num_rollout=NUM_ROLLOUT)
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args, wandb_args=command_utils.get_default_wandb_args(__file__))
