import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.model_scripts.test_glm5_744b_a40b_4layer_r3 import _args, execute, prepare

# test_glm5_744b_a40b_4layer_r3.py with the indexer top-k inline in the SGLang response
# instead of the output store. The two reach a rank's rows through different branches of
# attach_replay_data, and a file that declares metric gates holds one run, so each
# transport has its own file.

register_cuda_ci(
    est_time=1100,
    suite="stage-c-2-gpu-h200",
    labels=["megatron", "model-scripts", "replay"],
    hardware=["hopper", "blackwell"],
    nightly=True,
    num_gpus=2,
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")
register_ci_gate(metric_key="perf/actor_train_mfu")


if __name__ == "__main__":
    args = _args(use_output_store=False)
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
