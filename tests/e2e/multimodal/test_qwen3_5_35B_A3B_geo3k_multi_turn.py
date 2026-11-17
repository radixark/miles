"""Qwen3.5-35B-A3B on the geo3k multi-turn recipe, in token space.

Runs `examples/geo3k_vlm/multi_turn/run_geo3k_vlm_multi_turn_token_space.py` as committed, cut to two
rollouts of 8 prompts on 4 GPUs: TP2 x EP4 with the Adam state on the host, routing replay over the per-turn
rows, DP vision encoder and the disk item feature cache on two TP2 engines. Functionality, not accuracy.
"""

from examples.geo3k_vlm.multi_turn.run_geo3k_vlm_multi_turn_token_space import ScriptArgs, execute, prepare
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

# 4 GPUs hold the recipe's TP2 x EP4; its 8 only add data-parallel ranks and engines.
register_cuda_ci(
    est_time=1200,
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


def _args() -> ScriptArgs:
    # The later flag wins, so the batch sizes here override the recipe's 32 x 8 / 256.
    return ScriptArgs.from_env(
        num_gpus_per_node=4,
        num_rollout=2,
        extra_args="--rollout-batch-size 8 --global-batch-size 64 --ci-test ",
    )


if __name__ == "__main__":
    args = _args()
    prepare(args)
    execute(args)
