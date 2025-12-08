import re
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    hf_checkpoint: str = "/root/.cache/dsv32-ckpt/DeepSeek-V3.2-5layer"
    torch_dist_checkpoint: str = "/root/.cache/dsv32-ckpt/DeepSeek-V3-0324-5layer_torch_dist"
    num_gpus_per_node: int = 8
    enable_eval: bool = False
    enable_deepep: bool = False
    extra_args: str = ""
    task: Literal["dapo_aime", "gsm8k"] = "dapo_aime"
    mode: Literal["normal", "debug_minimal"] = "debug_minimal"


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    load_save_path = f"/root/shared_data/{args.run_id}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.torch_dist_checkpoint} "
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
        "--rollout-batch-size 128 "
        "--n-samples-per-prompt 8 "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 4 "
        "--balance-data "
    )

    if args.mode != "debug_minimal":
        rollout_args += (
            "--over-sampling-batch-size 256 "
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += "--eval-interval 20 " "--eval-top-p 0.7 "

    match args.task:
        case "dapo_aime":
            rollout_args += (
                "--prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl "
                "--input-key prompt "
                f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 32768} "
            )
            eval_args += (
                "--eval-prompt-data aime /root/aime-2024/aime-2024.jsonl "
                "--n-samples-per-eval-prompt 8 "
                "--eval-max-response-len 32768 "
            )
        case "gsm8k":
            rollout_args += (
                "--prompt-data /root/gsm8k/train.parquet "
                "--input-key messages "
                "--rollout-max-response-len 256 "
            )
            eval_args += (
                "--eval-prompt-data gsm8k /root/gsm8k/test.parquet "
                "--n-samples-per-eval-prompt 1 "
                "--eval-max-response-len 256 "
            )

    if args.num_nodes <= 2:
        perf_args = (
            "--tensor-model-parallel-size 1 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 8 "
            "--expert-model-parallel-size 8 "
            "--expert-tensor-parallel-size 1 "
        )
    elif args.num_nodes <= 4:
        perf_args = (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 8 "
            "--expert-model-parallel-size 8 "
            "--expert-tensor-parallel-size 1 "
        )
    else:
        perf_args = (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 8 "
            "--expert-model-parallel-size 16 "
            "--expert-tensor-parallel-size 1 "
        )
    perf_args += (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
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

    sglang_decode_max_bs = 256
    sglang_world_size = 8 if args.num_nodes <= 4 else 64
    sglang_attn_dp_size = 1 if args.num_nodes <= 4 else 8
    sglang_attn_tp_size = sglang_world_size // sglang_attn_dp_size
    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        "--sglang-mem-fraction-static 0.7 "
        # f"--sglang-tp-size {sglang_world_size} "
        f"--sglang-tp-size 1 "
        f"--sglang-ep-size {sglang_world_size} "
        "--sglang-enable-dp-attention "
        f"--sglang-dp-size {sglang_attn_dp_size} "
        "--sglang-moe-dense-tp-size 1 "
        "--sglang-enable-dp-lm-head "
        "--sglang-server-concurrency 1024 "
        f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
        f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
        f"--sglang-cuda-graph-max-bs {sglang_decode_max_bs} "
    )
    if args.enable_deepep:
        sglang_args += (
            "--sglang-moe-a2a-backend deepep "
            "--sglang-deepep-mode low_latency "
        )
    sglang_extra_env_vars = {}
    if args.enable_deepep:
        sglang_extra_env_vars["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = f"{sglang_decode_max_bs}"

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
        "--use-fault-tolerance "
        f"--dump-details /root/shared_data/{args.run_id}/dump_details "
        "--disable-weights-backuper "
    )

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
        train_script="scripts/train_dsv32.py",
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="deepseek-v32-5layer",
        extra_env_vars={**sglang_extra_env_vars},
    )


if __name__ == "__main__":
    app()

