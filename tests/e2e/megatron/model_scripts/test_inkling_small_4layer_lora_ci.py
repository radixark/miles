import os

from scripts.run_inkling import _MODEL_REGISTRY, ScriptArgs, _train
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

import miles.utils.external_utils.command_utils as U

# Smoke test for scripts/run_inkling.py --train-mode lora on the 4-layer slice:
# shared-outer grouped-expert LoRA served through SGLang's virtual-experts path, one
# 4-GPU engine, adapter sync verified by checksum. Functionality, not accuracy.


register_cuda_ci(
    est_time=800,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "model-scripts", "lora"],
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

_MODEL_ORG = "CharyZeng"


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="Inkling-Small-4layer",
        train_mode="lora",
        task="dapo_math",
        num_nodes=1,
        num_gpus_per_node=4,
        rollout_num_gpus_per_engine=4,
        num_rollout=2,
        rollout_max_response_len=512,
        sglang_context_length=1024,
        extra_args=(
            "--ci-test "
            "--ci-disable-kl-checker "
            # frozen towers and the engine-derived adapter buffers never match the snapshot
            "--check-weight-update-skip-list visual. audio. ._w1_delta ._a_cat "
            "--ci-disable-logprobs-checker "
            "--check-lora-weight-equal "
        ),
    )


def prepare(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download {_MODEL_ORG}/{args.model_name} --local-dir {args.hf_checkpoint}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=_MODEL_REGISTRY[args.model_name],
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
