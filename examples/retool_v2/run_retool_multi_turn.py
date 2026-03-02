import os
from dataclasses import dataclass, field
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

WANDB_PROJECT = "miles-dev-retool-v2"
WANDB_GROUP = "sft-multi-turn-batch-32"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = field(default_factory=U.create_run_id)
    hardware: Literal["H100", "GB200", "GB300"] = "H100"
    num_gpus_per_node: int | None = None
    use_sft_model: bool = True
    save_path: str = "/root/Qwen3-4B_miles/retool_v2_multi_turn"
    prompt_data: str = "/root/dapo-math-17k/dapo-math-17k.jsonl"
    generate_max_turns: int = 16
    rollout_num_gpus_per_engine: int = 4
    extra_args: str = ""

    # resolved in __post_init__, not set by user
    hf_checkpoint: str = field(init=False)
    ref_load: str = field(init=False)

    def __post_init__(self):
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]
        if self.use_sft_model:
            self.hf_checkpoint = "/root/font-info/qwen3-4b-sft"
            self.ref_load = "/root/font-info/qwen3-4b-sft_torch_dist"
        else:
            self.hf_checkpoint = "/root/models/Qwen3-4B"
            self.ref_load = "/root/models/Qwen3-4B_torch_dist"


## TODO: Delete in the future
def _get_wandb_args() -> str:
    WANDB_API_KEY = "a37f4796e6205800c4212556a38e1319b5f144b7"
    return (
        "--use-wandb "
        f"--wandb-project {WANDB_PROJECT} "
        f"--wandb-group {WANDB_GROUP} "
        f"--wandb-key {WANDB_API_KEY} "
    )


def prepare(args: ScriptArgs):
    U.exec_command("mkdir -p /root/dapo-math-17k /root/aime-2024")
    U.exec_command("hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k")
    U.exec_command("hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024")

    if args.use_sft_model:
        U.exec_command("mkdir -p /root/font-info")
        U.exec_command(f"hf download font-info/qwen3-4b-sft-SGLang-RL --local-dir {args.hf_checkpoint}")
        U.convert_checkpoint(
            model_name="qwen3-4b-sft",
            megatron_model_type="qwen3-4B",
            num_gpus_per_node=args.num_gpus_per_node,
            hf_checkpoint=args.hf_checkpoint,
            dir_dst="/root/font-info",
        )
    else:
        U.exec_command("mkdir -p /root/models")
        U.exec_command("hf download Qwen/Qwen3-4B --local-dir /root/models/Qwen3-4B")
        U.convert_checkpoint(
            model_name="Qwen3-4B",
            megatron_model_type="qwen3-4B",
            num_gpus_per_node=args.num_gpus_per_node,
            dir_dst="/root/models",
        )


def execute(args: ScriptArgs):
    megatron_model_type = "qwen3-4B"

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        f"--save {args.save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 1000} "
        f"{'--rotary-base 5000000 ' if args.use_sft_model else ''}"
    )

    custom_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate "
        "--generate-tool-specs-path examples.retool_v2.tool_sandbox.tool_specs "
        "--generate-execute-tool-function-path examples.retool_v2.tool_sandbox.execute_tool "
        "--generate-tool-call-parser qwen25 "
        f"--generate-max-turns {args.generate_max_turns} "
        "--log-multi-turn "
    )

    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--custom-rm-path examples.retool_v2.tool_sandbox.reward_func "
        "--reward-key score "
        "--num-rollout 3000 "
        "--rollout-batch-size 16 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        "--global-batch-size 128 "
        "--balance-data "
    )

    eval_args = ""
    if args.mode != "debug_minimal":
        eval_args = (
            "--eval-interval 20 "
            "--eval-prompt-data aime /root/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
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

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        "--sglang-mem-fraction-static 0.7 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {2 if args.num_gpus_per_node == 8 else 1} "
        "--sequence-parallel "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--log-passrate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{_get_wandb_args()} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{custom_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        extra_env_vars={
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            "PYTHONPATH": "/root/Megatron-LM/:/root/miles",
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
