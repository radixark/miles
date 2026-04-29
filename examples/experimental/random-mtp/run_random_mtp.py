"""Train a model with randomly initialized MTP (Multi-Token Prediction) head.

Adds an MTP head with random weights to a base model that has no pretrained MTP,
then trains the MTP head jointly during RL. Over the course of training the MTP
accept rate should climb from ~0 to meaningful levels, improving inference speed
for future rollouts.

Supported setups:
    - colocate  (default, single-node)
    - disagg    (non-colocated, uses distributed weight sync)

Example usage:
    # 1-node, 8 GPU, colocated (default)
    python run_random_mtp.py

    # 1-node, 8 GPU, disaggregated
    python run_random_mtp.py --disagg

    # Quick smoke test
    python run_random_mtp.py --mode debug_minimal
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3-4B"
    num_gpus_per_node: int = 8
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    disagg: bool = False
    mtp_num_layers: int = 1
    mtp_loss_scaling_factor: float = 0.2
    extra_args: str = ""


def prepare(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command(f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)

    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type="qwen3-4B",
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    is_debug = args.mode == "debug_minimal"
    ref_load_path = f"{args.model_dir}/{args.model_name}_torch_dist"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name}/ "
        f"--ref-load {ref_load_path} "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 50 "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type dapo "
        "--reward-key score "
        f"--num-rollout {2 if is_debug else 300} "
        f"--rollout-batch-size {2 if is_debug else 16} "
        f"--n-samples-per-prompt {1 if is_debug else 4} "
        f"--rollout-max-response-len {32 if is_debug else 2048} "
        "--rollout-temperature 1 "
        f"--global-batch-size {2 if is_debug else 64} "
    )

    eval_args = ""
    if not is_debug:
        eval_args = (
            "--eval-interval 50 "
            "--skip-eval-before-train "
            "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 1024 "
            "--eval-top-k 1 "
        )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
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

    # Random MTP: adds an MTP head with random init and trains it during RL.
    # The speculative algorithm is auto-configured by --init-random-mtp.
    # Use raw mode so miles' model_provider actually builds the MTP block from CLI args;
    # --init-random-mtp also routes checkpoint loading to bridge.load_hf_weights internally,
    # which naturally skips MTP keys that don't exist in the base HF checkpoint.
    mtp_args = (
        f"--mtp-num-layers {args.mtp_num_layers} "
        "--enable-mtp-training "
        f"--mtp-loss-scaling-factor {args.mtp_loss_scaling_factor} "
        "--init-random-mtp "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 2 "
        f"--sglang-mem-fraction-static {0.4 if is_debug else 0.7} "
        "--sglang-enable-metrics "
        "--sglang-speculative-num-steps 2 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 3 "
    )
    if is_debug:
        sglang_args += "--sglang-disable-cuda-graph "

    if args.disagg:
        misc_args = (
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-softmax-in-fp32 "
            "--attention-backend flash "
            "--update-weight-transfer-mode broadcast "
            f"--actor-num-nodes {args.num_nodes} "
            f"--actor-num-gpus-per-node 4 "
            f"--num-gpus-per-node {args.num_gpus_per_node} "
            f"--rollout-num-gpus {args.num_gpus_per_node} "
            "--update-weight-buffer-size 536870912 "
        )
    else:
        misc_args = (
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-softmax-in-fp32 "
            "--attention-backend flash "
            "--colocate "
            "--update-weight-buffer-size 536870912 "
            f"--actor-num-nodes {args.num_nodes} "
            f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
            f"--num-gpus-per-node {args.num_gpus_per_node} "
        )

    train_args = (
        f"{ckpt_args}"
        f"{rollout_args}"
        f"{eval_args}"
        f"{optimizer_args}"
        f"{grpo_args}"
        f"{mtp_args}"
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args}"
        f"{sglang_args}"
        f"{misc_args}"
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3-4B",
        megatron_path=args.megatron_path,
        extra_env_vars={
            "PYTHONPATH": f"{args.megatron_path}",
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
