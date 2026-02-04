"""
This file is in preview, and will be further refined and optimized.
"""

import re
from dataclasses import dataclass
from typing import Literal, Optional
import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "debug_minimal"
    run_id: str = U.create_run_id()
    model_org: str = "deepseek-ai"
    model_name: Literal["DeepSeek-V4-285B", "DeepSeek-V4-285B-5layer"] = "DeepSeek-V4-285B"
    hf_checkpoint: Optional[str] = None
    num_gpus_per_node: int = 8
    enable_eval: bool = True
    extra_args: str = ""
    task: Literal["dapo_aime", "gsm8k"] = "gsm8k"
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    model_local_dir: str = "/root/local_data"
    save_dir: str = "/root/models"
    megatron_path: str = "/host_home/primary_synced/megatron-sunrise"
    enable_r3: bool = True
    enable_rir: bool = False
    enable_pp: bool = False
    optimizer_offload: bool = False
    use_fault_tolerance: bool = True
    debug_train_run_id: str | None = None
    debug_train_rollout_id: str | None = None
    train_partial_deterministic: bool = True
    fp8_training: bool = False
    enable_mis: bool = False

    @property
    def megatron_model_type(self):
        return {
            "DeepSeek-V4-285B": "deepseek-v4-285B",
            "DeepSeek-V4-285B-5layer": "deepseek-v4-285B-5layer",
        }[self.model_name]


@app.command()
@U.dataclass_cli
def prepare_single(args: ScriptArgs):
    """This script only needs to be executed on one node."""
    match args.task:
        case "dapo_aime":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
            U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)

    U.fp8_cast_bf16(
        path_src=args.hf_checkpoint,
        path_dst=f"{args.model_dir}/{args.model_name}-bf16/",
    )


@app.command()
@U.dataclass_cli
def prepare_spmd(args: ScriptArgs):
    extra_args = "--tensor-model-parallel-size 1 " "--expert-tensor-parallel-size 1 "
    if args.num_nodes == 1 and args.model_name == "DeepSeek-V4-285B-5layer":
        extra_args += "--pipeline-model-parallel-size 1 " "--expert-model-parallel-size 1 "
    else:
        extra_args += (
            "--pipeline-model-parallel-size 8 "
            "--expert-model-parallel-size 4 "
            "--decoder-first-pipeline-num-layers 7 "
            "--decoder-last-pipeline-num-layers 6 "
        )

    num_gpus_for_convert = args.num_gpus_per_node
    if args.model_name == "DeepSeek-V4-285B-5layer":
        num_gpus_for_convert = min(num_gpus_for_convert, 5)

    U.convert_checkpoint(
        model_name=args.model_name,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}-bf16",
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=num_gpus_for_convert,
        multinode=True if args.num_nodes > 1 else False,
        extra_args=extra_args,
        dir_dst=f"{args.model_dir}",
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare_cp(args: ScriptArgs):
    _prepare_cp(args)


def _prepare_cp(args: ScriptArgs):
    U.rsync_simple(
        path_src=f"{args.model_dir}/{args.model_name}_torch_dist",
        path_dst=f"{args.model_local_dir}/{args.model_name}_torch_dist",
    )
    U.rsync_simple(
        path_src=f"{args.model_dir}/{args.model_name}",
        path_dst=f"{args.model_local_dir}/{args.model_name}",
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    print("running on {args.num_nodes} nodes")

    load_save_path = f"{args.save_dir}/{args.run_id}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.model_local_dir}/{args.model_name}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 20 "
        "--save-retain-interval 20 "
    )

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
    )

    if args.mode != "debug_minimal":
        rollout_args += (
            "--over-sampling-batch-size 512 "
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    eval_args = ""
    if args.enable_eval:
        eval_args += "--eval-interval 20 " "--eval-top-p 0.7 "

    match args.task:
        case "dapo_aime":
            rollout_args += (
                f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
                "--input-key prompt "
                f"--rollout-max-response-len 3072 "
                """--apply-chat-template-kwargs '{"thinking":true}' """
            )
            eval_args += (
                f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
                "--n-samples-per-eval-prompt 8 "
                "--eval-max-response-len 6144"
            )
        case "gsm8k":
            rollout_args += (
                f"--prompt-data {args.data_dir}/gsm8k/train.parquet "
                "--input-key messages "
                "--rollout-max-response-len 256 "
            )
            eval_args += (
                f"--eval-prompt-data gsm8k {args.data_dir}/gsm8k/test.parquet "
                "--n-samples-per-eval-prompt 1 "
                "--eval-max-response-len 256 "
            )

    if args.num_nodes <= 2:
        if args.enable_pp:
            perf_args = (
                "--tensor-model-parallel-size 2 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 2 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 2 "
                "--expert-tensor-parallel-size 1 "
                "--pipeline-model-parallel-layout 'E,t*2\\|t*3,L' " # TODO: temporarily for pp=2
            )
        else:
            perf_args = (
                f"--tensor-model-parallel-size {args.num_gpus_per_node} "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 1 "
                "--context-parallel-size 1 "
                f"--expert-model-parallel-size {args.num_gpus_per_node} "
                "--expert-tensor-parallel-size 1 "
            )
    elif args.num_nodes <= 4:
        perf_args = (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 4 "
            "--expert-tensor-parallel-size 1 "
        )
    elif args.num_nodes <= 6:
        perf_args = (
            "--tensor-model-parallel-size 8 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 6 "
            "--decoder-first-pipeline-num-layers 8 "
            "--decoder-last-pipeline-num-layers 7 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 8 "
            "--expert-tensor-parallel-size 1 "
        )
    elif args.num_nodes <= 8:
        perf_args = (
            "--tensor-model-parallel-size 8 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 8 "
            "--decoder-first-pipeline-num-layers 8 "
            "--decoder-last-pipeline-num-layers 5 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 8 "
            "--expert-tensor-parallel-size 1 "
        )
    else:
        raise NotImplementedError

    perf_args += (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--max-tokens-per-gpu 2048 "
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
    if args.optimizer_offload:
        optimizer_args += (
            "--optimizer-cpu-offload "
            "--overlap-cpu-optimizer-d2h-h2d "
            "--use-precision-aware-optimizer "
        )

    sglang_world_size = args.num_gpus_per_node
    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "

        f"--sglang-tp-size {sglang_world_size} "
        f"--sglang-dp-size {sglang_world_size} "
        "--sglang-enable-dp-attention "

        "--sglang-disable-radix-cache "
        "--sglang-attention-backend compressed "
        "--sglang-page-size 256 "
        f"--sglang-max-running-requests {16 * sglang_world_size} "
        "--sglang-chunked-prefill-size 8192 "

        "--sglang-server-concurrency 1024 "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "  # TODO improve
    )
    extra_env_vars = {
        "SGLANG_HACK_V4_SET_K_AND_S_BACKEND": "triton",
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        "SGLANG_SKIP_SECOND_APT_CONVERT": "1",
    }

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        f"--update-weight-buffer-size {4 * 1024 ** 3} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        f"--dump-details /root/shared_data/{args.run_id}/dump_details "
        "--disable-weights-backuper "
        "--model-name deepseekv4 "  # for mbridge load
        "--train-memory-margin-bytes 1073741824 "
        "--qkv-format bshd "
        "--moe-router-freeze-gate "
        "--freeze-e-score-correction-bias "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
    )
    
    if args.enable_mis:
        misc_args += (
            "--use-tis "
            "--custom-config-path examples/train_infer_mismatch_helper/mis.yaml "
            "--custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp "
        )

    if args.use_fault_tolerance:
        misc_args += "--use-fault-tolerance "

    if args.debug_train_run_id is not None:
        if args.debug_train_rollout_id is None:
            args.debug_train_rollout_id = 1
        misc_args += f"--load-debug-rollout-data \
            /root/shared_data/{args.debug_train_run_id}/dump_details/rollout_data/{args.debug_train_rollout_id}.pt "
        misc_args += "--debug-train-only "

    if args.enable_r3:
        misc_args += "--use-rollout-routing-replay "
    if args.enable_rir:
        misc_args += "--use-rollout-indexer-replay "
    if args.enable_r3 or args.enable_rir:
        misc_args += "--use-miles-router "

    if args.train_partial_deterministic:
        extra_env_vars |= {
            "MILES_HACK_TRAIN_TORCH_DETERMINISTIC": "1",
            "NCCL_ALGO": "Ring",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    
    if args.fp8_training:
        misc_args += (
            "--transformer-impl transformer_engine "
            "--bf16 "
            "--fp8-format e4m3 "
            "--fp8-recipe blockwise "
        )
        extra_env_vars |= {
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
        }

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={**extra_env_vars},
        megatron_path=args.megatron_path,
    )


if __name__ == "__main__":
    app()
