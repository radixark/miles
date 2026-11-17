import os

from scripts.run_minimax_m3 import _RECIPES, ScriptArgs, execute
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

register_cuda_ci(
    est_time=1500,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "model-scripts"],
    hardware=["hopper", "blackwell"],
    num_gpus=4,
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
# deterministic_random scores each sample by a hash bit, so raw_reward has nothing to regress; track it
register_ci_gate(metric_key="rollout/raw_reward", constraint=None)
register_ci_gate(metric_key="perf/actor_train_mfu")

_MODEL_ORG = "CharyZeng"


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="MiniMax-M3-4layer",
        num_nodes=1,
        num_gpus_per_node=4,
        num_rollout=3,
        rollout_max_response_len=512,
        extra_args=(
            "--ci-test "
            "--ci-disable-logprobs-checker "
            # the pruned model scores 0 on math, which zeroes every advantage; a pseudo-random
            # reward keeps the weights moving so each weight sync carries real deltas
            "--rm-type deterministic_random "
            "--offload-train-target cpu "
        ),
    )


def prepare(args: ScriptArgs):
    U = args.create_backend()
    recipe = _RECIPES[args.model_name]
    hf_checkpoint = f"{args.model_dir}/{args.model_name}"
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download {_MODEL_ORG}/{args.model_name} --local-dir {hf_checkpoint}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.convert_checkpoint(
        model_name=recipe.megatron_model_type,
        megatron_model_type=recipe.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=hf_checkpoint,
        megatron_path=args.megatron_path,
    )


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
