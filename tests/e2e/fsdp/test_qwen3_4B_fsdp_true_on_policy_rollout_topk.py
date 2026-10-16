"""True-on-policy rollout top-k candidates match the trainer, on the native /generate rollout path.

SGLang returns the candidates as nested ``output_top_logprobs`` rows; the check is in
``_true_on_policy_rollout_topk``.
"""

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.fsdp._true_on_policy_rollout_topk import main

register_cuda_ci(
    est_time=900, suite="stage-c-2-gpu-h200", labels=["fsdp", "sglang", "replay"], hardware=["hopper"], num_gpus=2
)
# --ci-test already asserts both are exactly 0 on every rollout.
register_ci_gate(
    metric_key="rollout/topk_logprob_abs_diff", steps="all", constraint={"abs_floor_up": 0.0, "abs_floor_down": 0.0}
)
register_ci_gate(
    metric_key="rollout/topk_set_mismatch", steps="all", constraint={"abs_floor_up": 0.0, "abs_floor_down": 0.0}
)
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")

if __name__ == "__main__":
    main(__file__, rollout_path_args="--apply-chat-template ")
