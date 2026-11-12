import os
import tempfile
from argparse import Namespace
from pathlib import Path

import torch
from tests.ci.ci_register import register_cuda_ci

from miles.utils.external_utils import command_utils
from miles.utils.types import Sample

register_cuda_ci(
    est_time=600,
    suite="stage-c-2-gpu-h200",
    labels=["short"],
    hardware=["hopper"],
    nightly=True,
    num_gpus=2,
)

MODEL_NAME = "Qwen3-VL-2B-Instruct"
MODEL_TYPE = "qwen3-1.7B"
DATASET_NAME = "chenhegu/geo3k_imgurl"
NUM_GPUS = 2


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset(DATASET_NAME)


def _train_args(extra_args: str) -> str:
    rollout_args = (
        "--prompt-data /root/datasets/geo3k_imgurl/train.parquet "
        "--input-key problem "
        "--label-key answer "
        '--multimodal-keys \'{"image": "images"}\' '
        "--apply-chat-template "
        "--rollout-shuffle "
        "--custom-generate-function-path tests.manual.compact_split_generate.generate_one_to_three "
        "--custom-rm-path tests.e2e.short.test_qwen3_vl_2B_compact_rollout.reward_by_index "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--global-batch-size 16 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.6 "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        "--megatron-to-hf-mode bridge "
    )

    return (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        "--ci-test "
        f"{misc_args} "
        f"{extra_args} "
    )


async def reward_by_index(args: Namespace, samples: Sample | list[Sample], **_kwargs) -> float | list[float]:
    if isinstance(samples, list):
        return [float(sample.index % 2) for sample in samples]
    return float(samples.index % 2)


def _merge_rollouts(split_path: str, merged_path: str) -> None:
    payload = torch.load(split_path, weights_only=False)
    rollouts: dict[int, list[Sample]] = {}
    for sample in map(Sample.from_dict, payload["samples"]):
        rollouts.setdefault(sample.rollout_id, []).append(sample)

    merged = []
    for pieces in rollouts.values():
        if len(pieces) == 1:
            merged.append(pieces[0].to_dict())
            continue
        whole = max(pieces, key=lambda piece: piece.response_length)
        whole.loss_mask = [
            max(bits)
            for bits in zip(
                *(piece.loss_mask + [0] * (whole.response_length - piece.response_length) for piece in pieces),
                strict=True,
            )
        ]
        merged.append(whole.to_dict())
    torch.save({**payload, "samples": merged}, merged_path)


def execute():
    U = command_utils.default_config().create_backend()
    os.environ["MODEL_ARGS_ROTARY_BASE"] = "5000000"
    work_dir = Path(tempfile.mkdtemp(prefix="qwen3_vl_compact_rollout_"))
    split_rollouts = f"{work_dir}/split_{{rollout_id}}.pt"
    merged_rollouts = f"{work_dir}/merged_{{rollout_id}}.pt"
    grad_norms = f"{work_dir}/grad_norm_{{role}}_{{rollout_id}}_{{step_id}}.pt"

    def train(extra_args: str) -> None:
        U.execute_train(train_args=_train_args(extra_args), num_gpus_per_node=NUM_GPUS, megatron_model_type=MODEL_TYPE)

    train(f"--num-rollout 2 --save-debug-rollout-data {split_rollouts} ")
    _merge_rollouts(split_rollouts.format(rollout_id=0), merged_rollouts.format(rollout_id=0))
    replay = "--num-rollout 1 --no-enable-sample-ownership-checker --lr 0 "
    train(f"{replay} --load-debug-rollout-data {merged_rollouts} --ci-save-grad-norm {grad_norms} ")
    train(f"{replay} --load-debug-rollout-data {split_rollouts} --ci-load-grad-norm {grad_norms} ")


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
