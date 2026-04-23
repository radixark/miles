"""
This file is in preview, and will be further refined and optimized.
"""

from dataclasses import dataclass
from typing import Literal
import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "debug_minimal"
    run_id: str = U.create_run_id()
    model_org: str = "deepseek-ai"
    model_name: Literal["DeepSeek-V4-285B", "DeepSeek-V4-285B-4layer"] = "DeepSeek-V4-285B"
    hf_checkpoint: str | None = None
    num_gpus_per_node: int = 8
    enable_eval: bool = True
    extra_args: str = ""
    task: Literal["dapo_aime", "gsm8k"] = "gsm8k"
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    model_local_dir: str = "/root/local_data"
    save_dir: str = "/root/models"
    megatron_path: str = "/workspace/Megatron-LM"
    enable_r3: bool = True
    enable_rir: bool = False
    test_pp_single_node: bool = False
    test_cp_single_node: bool = False
    optimizer_offload: bool = False
    use_fault_tolerance: bool = True
    debug_train_run_id: str | None = None
    debug_train_rollout_id: str | None = None
    train_partial_deterministic: bool = True
    fp8_training: bool = False
    enable_mis: bool = False
    gb300: bool = False
    skip_saving: bool = False
    cp_size: int = 1

    @property
    def megatron_model_type(self):
        return {
            "DeepSeek-V4-285B": "deepseek-v4-285B",
            "DeepSeek-V4-285B-4layer": "deepseek-v4-285B-4layer",
        }[self.model_name]

    @property
    def torch_dist_name(self):
        return f"{self.model_name}_torch_dist"

    @property
    def bf16_name(self):
        return f"{self.model_name}-bf16"


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
        path_dst=f"{args.model_dir}/{args.bf16_name}/",
    )


@app.command()
@U.dataclass_cli
def prepare_spmd(args: ScriptArgs):
    extra_args = "--expert-tensor-parallel-size 1 --context-parallel-size 1 "
    if args.num_nodes == 1 and args.model_name == "DeepSeek-V4-285B-4layer":
        extra_args += (
            "--tensor-model-parallel-size 1 " "--pipeline-model-parallel-size 1 " "--expert-model-parallel-size 1 "
        )
    elif args.num_nodes == 7 and args.model_name == "DeepSeek-V4-285B":
        extra_args += (
            "--tensor-model-parallel-size 4 "
            "--pipeline-model-parallel-size 7 "
            "--expert-model-parallel-size 4 "
            "--decoder-first-pipeline-num-layers 4 "
            "--decoder-last-pipeline-num-layers 4 "
            "--make-vocab-size-divisible-by 64 "
        )
    else:
        extra_args += (
            "--tensor-model-parallel-size 1 "
            "--pipeline-model-parallel-size 8 "
            "--expert-model-parallel-size 4 "
            "--decoder-first-pipeline-num-layers 7 "
            "--decoder-last-pipeline-num-layers 6 "
        )

    num_gpus_for_convert = args.num_gpus_per_node
    if args.model_name == "DeepSeek-V4-285B-4layer":
        num_gpus_for_convert = min(num_gpus_for_convert, 4)

    U.convert_checkpoint(
        model_name=args.model_name,
        hf_checkpoint=f"{args.model_dir}/{args.bf16_name}",
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
        path_src=f"{args.model_dir}/{args.torch_dist_name}",
        path_dst=f"{args.model_local_dir}/{args.torch_dist_name}",
    )
    U.rsync_simple(
        path_src=f"{args.model_dir}/{args.model_name}",
        path_dst=f"{args.model_local_dir}/{args.model_name}",
    )


def _get_parallel_config(args: ScriptArgs) -> str:
    """Return parallel config args for tested GPU configurations.

    Only includes configurations that have been verified to work.
    Raises NotImplementedError for untested configurations.
    """
    total_gpus = args.num_nodes * args.num_gpus_per_node

    # Single-node configs (any GPU count)
    if args.num_nodes == 1:
        return (
            f"--tensor-model-parallel-size {args.num_gpus_per_node} "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            f"--expert-model-parallel-size {args.num_gpus_per_node} "
            "--expert-tensor-parallel-size 1 "
        )

    # GB300: 4 GPUs/node
    if args.num_gpus_per_node == 4:
        if total_gpus == 32:  # 8 nodes × 4 GPUs
            if args.cp_size == 2:
                return (
                    "--tensor-model-parallel-size 2 "
                    "--sequence-parallel "
                    "--pipeline-model-parallel-size 8 "
                    "--decoder-first-pipeline-num-layers 4 "
                    "--decoder-last-pipeline-num-layers 3 "
                    "--context-parallel-size 2 "
                    "--expert-model-parallel-size 4 "
                    "--expert-tensor-parallel-size 1 "
                )
            return (
                "--tensor-model-parallel-size 8 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 4 "
                "--decoder-first-pipeline-num-layers 11 "
                "--decoder-last-pipeline-num-layers 10 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 8 "
                "--expert-tensor-parallel-size 1 "
            )
        elif total_gpus == 28:  # 7 nodes × 4 GPUs
            return (
                "--tensor-model-parallel-size 4 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 7 "
                "--decoder-first-pipeline-num-layers 7 "
                "--decoder-last-pipeline-num-layers 6 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 4 "
                "--expert-tensor-parallel-size 1 "
            )

    # H200: 8 GPUs/node
    if args.num_gpus_per_node == 8:
        if total_gpus == 64:  # 8 nodes × 8 GPUs
            return (
                "--tensor-model-parallel-size 8 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 8 "
                "--decoder-first-pipeline-num-layers 4 "
                "--decoder-last-pipeline-num-layers 3 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 8 "
                "--expert-tensor-parallel-size 1 "
            )
        elif total_gpus == 56:  # 7 nodes × 8 GPUs
            return (
                "--tensor-model-parallel-size 8 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 7 "
                "--pipeline-model-parallel-layout 'E,t*4\\|t*7\\|t*7\\|t*7\\|t*7\\|t*7\\|t*4,L' "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 8 "
                "--expert-tensor-parallel-size 1 "
            )

    raise NotImplementedError(
        f"No verified parallel config for {total_gpus} GPUs "
        f"({args.num_nodes} nodes × {args.num_gpus_per_node} GPUs/node). "
        f"Add a tested config to _get_parallel_config()."
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    print(f"running on {args.num_nodes} nodes")

    load_save_path = f"{args.save_dir}/{args.run_id}/checkpoints"
    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} " f"--ref-load {args.model_local_dir}/{args.torch_dist_name} "
    if not args.skip_saving:
        ckpt_args += (
            f"--load {load_save_path} " f"--save {load_save_path} " "--save-interval 20 " "--save-retain-interval 20 "
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
                "--eval-max-response-len 3072 "
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

    perf_args = _get_parallel_config(args)

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
            "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
        )

    sglang_world_size = args.num_gpus_per_node
    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        f"--sglang-tp-size {sglang_world_size} "
        f"--sglang-dp-size {sglang_world_size} "
        "--sglang-enable-dp-attention "
        "--sglang-attention-backend compressed "
        "--sglang-page-size 256 "
        f"--sglang-max-running-requests {16 * sglang_world_size} "
        "--sglang-chunked-prefill-size 8192 "
        "--sglang-server-concurrency 1024 "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "  # TODO improve
    )
    sglang_args += f"--sglang-ep-size {sglang_world_size} "
    extra_env_vars = {
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        "SGLANG_DSV4_FP4_EXPERTS": "0",
    }

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # when use tp=4, 4GB will cause wgate and wkv not in same bucket
        f"--update-weight-buffer-size {8 * 1024 ** 3} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        f"--dump-details /root/shared_data/{args.run_id}/dump_details "
        "--disable-weights-backuper "
        "--model-name deepseekv4 "  # for mbridge load
        "--qkv-format bshd "
        "--moe-router-freeze-gate "
        "--freeze-e-score-correction-bias "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
    )

    if args.gb300:
        misc_args += "--train-memory-margin-bytes 3221225472 "
        misc_args += "--sglang-mem-fraction-static 0.7 "
    else:
        misc_args += "--train-memory-margin-bytes 1073741824 "

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
        misc_args += "--sglang-mem-fraction-static 0.6 "
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
        misc_args += "--transformer-impl transformer_engine " "--bf16 " "--fp8-format e4m3 " "--fp8-recipe blockwise "
        extra_env_vars |= {
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
            "MEGATRON_USE_KV_QAT": "1",
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
