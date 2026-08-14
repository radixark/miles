import os

from scripts.run_nemotron_3_ultra_550b_a55b import ScriptArgs, _execute_train, _prepare_download
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

# Smoke test for the Nemotron-3-Ultra (nemotron_h: hybrid Mamba2 + Attention + latent-MoE)
# training script. It runs a 4-layer slice on a single 8-GPU H200 node and only verifies that
# the training script is functional, not model accuracy.
#
# The slice keeps source layers 0,1,7,8 renumbered to 0..3, so its block pattern is
# mamba, moe, attention, moe ("ME*E") -- every block type the full 108-layer model has, with
# the 512-expert / top-22 / moe_latent_size=2048 MoE config untouched. 8 GPUs (not 4) because
# SGLang DP-attention needs attn_tp to divide Mamba n_groups=8, and --ci-test turns on the
# Megatron -> SGLang weight equality check, which is the main thing this test guards.


register_cuda_ci(
    est_time=900,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "model-scripts"],
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_org="CharyZeng",
        model_name="NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-4layer",
        mode="debug_minimal",
        num_nodes=1,
        num_gpus_per_node=8,
        num_rollout=2,
        rollout_batch_size=8,
        n_samples_per_prompt=2,
        global_batch_size=16,
        skip_saving=True,
        extra_args=("--ci-test " "--ci-disable-logprobs-checker "),
    )


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.output_dir}")
    _prepare_download(args)


def execute(args: ScriptArgs):
    _execute_train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
