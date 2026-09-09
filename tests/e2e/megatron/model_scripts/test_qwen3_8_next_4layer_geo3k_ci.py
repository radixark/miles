import os

from scripts.run_qwen3_8_next import _MODEL_REGISTRY, ScriptArgs, _train

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

import miles.utils.external_utils.command_utils as U


register_cuda_ci(
    est_time=1800,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "model-scripts"],
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

_MODEL_ORG = "CharyZeng"


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="Qwen3.8-Flash-Next-4layer",
        task="geo3k",
        num_nodes=1,
        num_gpus_per_node=8,
        num_rollout=2,
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        global_batch_size=4,
        rollout_max_response_len=512,
        skip_saving=True,
        extra_args=(
            "--ci-test " "--ci-disable-kl-checker " "--ci-disable-logprobs-checker " "--offload-train-target cpu "
        ),
    )


def prepare(args: ScriptArgs):
    os.environ["CONVERT_KEEP_PP1"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.ckpt_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download {_MODEL_ORG}/{args.model_name} --local-dir {args.hf_checkpoint}")
    U.hf_download_dataset("chenhegu/geo3k_imgurl", data_dir=args.data_dir)
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
