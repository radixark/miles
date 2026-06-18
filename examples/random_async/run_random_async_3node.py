import os
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3.5-35B-A3B"
    megatron_model_type: str = "qwen3.5-35B-A3B"
    num_gpus_per_node: int = 8
    actor_num_nodes: int = 2
    rollout_num_gpus: int = 8
    num_rollout: int = 4
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    pause_generation_mode: Literal["in_place", "retract"] = "in_place"
    update_weight_transfer_mode: Literal["broadcast", "p2p"] = "broadcast"
    skip_prepare: bool = False
    extra_args: str = ""


def prepare(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.model_dir}")
    U.exec_command(
        f'test "$(cat {args.model_dir}/{args.model_name}_torch_dist/latest_checkpointed_iteration.txt 2>/dev/null)" = release || '
        f"test -e {args.model_dir}/{args.model_name} || "
        f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"
    )
    U.exec_command(
        f"test -e {args.model_dir}/{args.model_name}-FP8 || "
        f"hf download Qwen/{args.model_name}-FP8 --local-dir {args.model_dir}/{args.model_name}-FP8"
    )
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    if args.pause_generation_mode == "in_place" and args.update_weight_transfer_mode == "p2p":
        raise ValueError(
            "in_place + p2p is not supported: P2P transfer engine conflicts with "
            "active NCCL inference. Use broadcast with in_place, or retract with p2p."
        )

    example_dir = os.path.dirname(os.path.abspath(__file__))
    sglang_config_path = os.path.join(example_dir, "sglang_config_qwen3_5_35b_1p1d.yaml")
    ref_load_path = f"{args.model_dir}/{args.model_name}_torch_dist"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name}-FP8/ "
        f"--ref-load {ref_load_path} "
        f"--load {load_save_path} "
    )

    rollout_args = (
        "--rollout-function-path random_async_rollout.generate_rollout_random_async "
        "--disable-rollout-global-dataset "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 16 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        "--global-batch-size 512 "
        "--balance-data "
        f"--pause-generation-mode {args.pause_generation_mode} "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 8 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8000 "
        "--log-probs-chunk-size 1024 "
        "--moe-token-dispatcher-type flex "
        "--moe-flex-dispatcher-backend deepep "
        "--transformer-impl transformer_engine "
        "--fp8-format e4m3 "
        "--fp8-recipe blockwise "
    )

    runtime_args = (
        "--router-prefill-policy manual "
        "--router-decode-policy manual "
        "--router-assignment-mode min_load "
        '--train-env-vars \'{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}\' '
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
        "--lr 0.0 "
        "--lr-decay-style constant "
        "--weight-decay 0.0 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_extra = ""
    if args.update_weight_transfer_mode == "p2p":
        sglang_extra = "--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine "

    sglang_args = (
        "--rollout-num-gpus-per-engine 4 "
        f"--sglang-config {sglang_config_path} "
        f"--sglang-mem-fraction-static 0.85 {sglang_extra}"
        "--sglang-attention-backend fa3 "
        "--sglang-enable-dp-attention "
        "--sglang-data-parallel-size 4 "
        "--sglang-expert-parallel-size 4 "
        "--sglang-enable-dp-lm-head "
        "--sglang-moe-a2a-backend deepep "
        "--sglang-context-length 80000 "
        "--sglang-enable-metrics "
        "--sglang-server-concurrency 384 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        f"--attention-backend flash --update-weight-transfer-mode {args.update_weight_transfer_mode} "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.rollout_num_gpus} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{runtime_args} "
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
        config=args,
        extra_env_vars={
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "RANDOM_ASYNC_MAX_CONTEXT_TOKENS": "60000",
            "RANDOM_ASYNC_CONCURRENCY_PER_GPU": "64",
            "SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK": "1",
            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "900",
            "PYTHONPATH": f"{args.megatron_path}:{example_dir}",
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
