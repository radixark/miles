"""Shared driver for the FSDP rollout-routing-replay (R3) e2e cases.

Each case runs a short GRPO job with ``--use-rollout-routing-replay`` and ``--ci-test``, which
turns on ``BaseReplayManager.check_replay_result``: for every token it compares the training
engine's recomputed topk against the indices replayed from the rollout and raises once the
mismatched fraction exceeds ``replay_check_max_mismatch_fraction`` (1e-2). A case that
finishes is therefore evidence that FSDP reproduced the rollout's routing, not merely that
nothing crashed.

``--gradient-checkpointing`` is deliberately on: it is what exercises the backward replay
cursor, since HF's GradientCheckpointingLayer re-runs each layer's forward during backward.
"""

import os
from dataclasses import dataclass

import miles.utils.external_utils.command_utils as U


@dataclass
class CaseConfig:
    model_name: str
    hf_repo: str
    num_gpus: int
    rollout_num_gpus_per_engine: int


def prepare(case: CaseConfig) -> None:
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download {case.hf_repo} --local-dir /root/models/{case.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")


def execute(case: CaseConfig, wandb_file: str) -> None:
    ckpt_args = f"--hf-checkpoint /root/models/{case.model_name} "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    fsdp_args = (
        "--train-backend fsdp "
        "--gradient-checkpointing "
        "--update-weight-buffer-size 536870912 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
        # Scoped to the training process on purpose: expandable_segments disables
        # TorchMemorySaver, which colocated SGLang engines need to release memory.
        '--train-env-vars \'{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}\' '
    )

    replay_args = "--use-rollout-routing-replay "

    sglang_args = (
        f"--rollout-num-gpus-per-engine {case.rollout_num_gpus_per_engine} "
        "--sglang-mem-fraction-static 0.8 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-chunked-prefill-size 4096 "
        "--sglang-attention-backend fa3 "
        "--attn-implementation flash_attention_3 "
    )

    ci_args = "--ci-test "

    misc_args = "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {case.num_gpus} " "--colocate "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{fsdp_args} "
        f"{replay_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{U.get_default_wandb_args(wandb_file)} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=case.num_gpus,
        megatron_model_type=None,
    )


def main(case: CaseConfig, wandb_file: str) -> None:
    prepare(case)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(case, wandb_file=wandb_file)
