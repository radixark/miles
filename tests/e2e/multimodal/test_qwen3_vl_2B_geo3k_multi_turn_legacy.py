"""Qwen3-VL-2B on the geo3k multi-turn recipe, with the Miles-side processor (no token space).

Runs `examples/geo3k_vlm/multi_turn/run_geo3k_vlm_multi_turn.py` as committed, cut to two rollouts of 8 prompts.
`examples/geo3k_vlm/multi_turn/rollout.py` serves this path and token space from one file, so this is the run
of the branch the token-space files never take. Functionality, not accuracy.
"""

from examples.geo3k_vlm.multi_turn.run_geo3k_vlm_multi_turn import execute, prepare
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

register_cuda_ci(
    est_time=600,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "multimodal"],
    hardware=["hopper"],
    num_gpus=4,
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")
register_ci_gate(metric_key="perf/actor_train_mfu")

# The later flag wins, so these override the recipe's 3000 rollouts of 64 x 8 / 512.
EXTRA_ARGS = "--num-rollout 2 --rollout-batch-size 8 --global-batch-size 64 --ci-test "


if __name__ == "__main__":
    prepare()
    execute(extra_args=EXTRA_ARGS)
