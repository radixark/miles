"""K3 CP2 + EP8, using the same two-rollout BF16 workload as the CP1 case."""

from dataclasses import replace

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.torchtitan._common import execute
from tests.e2e.torchtitan.test_kimi_k3_4layer_ep import CASE as EP_CASE
from tests.e2e.torchtitan.test_kimi_k3_4layer_ep import prepare

register_cuda_ci(est_time=1800, suite="stage-c-8-gpu-h100", num_gpus=8, labels=["torchtitan"], hardware=["hopper"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")

CASE = replace(EP_CASE, cp_size=2)


if __name__ == "__main__":
    prepare()
    execute(CASE, wandb_file=__file__)
