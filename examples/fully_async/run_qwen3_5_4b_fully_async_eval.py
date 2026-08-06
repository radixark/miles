"""Fully-async Qwen3.5-4B with async checkpoint eval on one 8-GPU node.

Both eval backends implement the same trainer contract (CheckpointEvalFn): the
trainer exports an HF snapshot per eval point and fires eval without pausing
training; the backend pins the snapshot and evals it.

    # in-job dedicated eval fleet (Ray-managed engine)
    python run_qwen3_5_4b_fully_async_eval.py --eval-backend fleet

    # fn-launched standalone sglang server on the spare GPU
    python run_qwen3_5_4b_fully_async_eval.py --eval-backend external

GPU split is identical in both: 4 actor (TP=2) + 3 rollout engines (TP=1) + 1 eval
GPU — Ray-managed fleet engine (fleet) or fn-owned sglang server (external).
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    eval_backend: Literal["fleet", "external"] = "fleet"
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3.5-4B"
    megatron_model_type: str = "qwen3.5-4B"
    num_gpus_per_node: int = 8
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""


def prepare(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command(f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {args.output_dir}/{args.run_id}/checkpoints "
        f"--save {args.output_dir}/{args.run_id}/checkpoints "
        "--save-interval 20 "
    )

    rollout_args = (
        "--rollout-function-path miles.rollout.fully_async_rollout.FullyAsyncRolloutFn "
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type dapo "
        "--reward-key score "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
        "--balance-data "
        # retract (default) can deadlock flush_cache in fully_async under load
        "--pause-generation-mode in_place "
    )

    # Shared by both backends: snapshot staging on tmpfs + the eval datasets.
    # All eval config lives here in the training args — the backend reads it as-is.
    eval_args = (
        "--eval-interval 5 "
        f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 8 "
        "--eval-max-response-len 16384 "
        "--eval-top-p 1 "
        "--eval-hf-dir /dev/shm/miles_eval_hf "
        "--eval-keep-snapshots 2 "
    )
    # 4 actor + N-5 rollout + 1 eval (a Ray fleet engine, or the external server's GPU).
    rollout_num_gpus = args.num_gpus_per_node - 4 - 1
    eval_env = {}
    if args.eval_backend == "fleet":
        eval_args += "--eval-num-gpus 1 --eval-num-gpus-per-engine 1 "
    else:
        eval_args += "--eval-function-path examples.fully_async.external_eval_fn.ExternalSglangEvalFn "
        # The fn launches its own sglang server on the last GPU, outside the Ray split.
        eval_env = {"MILES_EXTERNAL_EVAL_GPUS": str(args.num_gpus_per_node - 1)}

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--use-tis "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 4 "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {rollout_num_gpus} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{eval_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        train_script="train_async.py",
        megatron_path=args.megatron_path,
        extra_env_vars={
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            "PYTHONPATH": args.megatron_path,
            **eval_env,
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
