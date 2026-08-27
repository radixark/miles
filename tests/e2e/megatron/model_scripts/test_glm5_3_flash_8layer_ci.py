import os

from scripts.run_glm5_3_flash import _MODEL_REGISTRY, ScriptArgs, _train

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

import miles.utils.external_utils.command_utils as U


register_cuda_ci(
    est_time=1800,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "model-scripts"],
    disabled="needs CharyZeng/GLM-5.3-Flash-8layer on HF and a CI image with GLM-5.3-Flash "
    "sglang support (branch sglang-miles-glm53); re-enable once both land.",
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

_MODEL_ORG = "CharyZeng"


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="GLM-5.3-Flash-8layer",
        num_nodes=1,
        num_gpus_per_node=8,
        num_rollout=5,
        rollout_max_response_len=512,
        enable_r3=True,
        skip_saving=True,
        extra_args=(
            "--ci-test " "--ci-disable-kl-checker " "--ci-disable-logprobs-checker " "--offload-train-target cpu "
        ),
    )


def prepare(args: ScriptArgs):
    os.environ["CONVERT_KEEP_PP1"] = "1"
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.ckpt_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download {_MODEL_ORG}/{args.model_name} --local-dir {args.hf_checkpoint}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.convert_checkpoint(
        model_name=_MODEL_REGISTRY[args.model_name],
        megatron_model_type=_MODEL_REGISTRY[args.model_name],
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.ckpt_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
        extra_args="--tensor-model-parallel-size 2 --pipeline-model-parallel-size 1",
    )


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
