"""Qwen3-30B-A3B GRPO LoRA training on MI350X/MI355X, colocated.

Trains a rank-32 adapter over attention and the routed experts through
Megatron-Bridge (``--megatron-to-hf-mode bridge``) and republishes it to the
colocated SGLang engines after every rollout.

``--experts-shared-outer-loras`` picks the MoE-expert adapter layout. The two
layouts are not checkpoint-compatible, and each is paired here with the serving
path that carries it, the same pairing the MoE LoRA E2Es use: shared-outer with
SGLang's virtual experts, per-expert with the fused_moe_lora alignment path.
Both layouts train and serve; the hyperparameters below are tuned for the
per-expert default, and shared-outer shortens responses far more aggressively
under them.

Requires ``prepare`` to have downloaded the HF checkpoint and the datasets; the
bridge path reads the HF checkpoint directly, so no torch_dist conversion is
needed.

Args:
    --model-name: HF repository name, also the directory under --model-dir.
    --hardware: node type; "auto" resolves it from the visible GPU.
    --num-rollout: total GRPO rollouts. This is a global endpoint, so keep it at
        the final value when resuming rather than adding to it.
    --lora-adapter-path: resume from a previously saved ``iter_N/adapter``,
        which carries optimizer and scheduler state alongside the weights.
    --experts-shared-outer-loras: train the shared-outer expert layout instead
        of the per-expert one.
    --sglang-attention-backend: off by default; overrides SGLang's own choice.

Example:
    python scripts/amd/run_qwen3_30b_a3b_lora.py --num-rollout 6
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3-30B-A3B"
    megatron_model_type: str = "qwen3-30B-A3B"
    hardware: Literal["auto", "MI350X", "MI355X"] = "auto"
    num_gpus_per_node: int | None = None
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = _TARGET_MODULES
    lora_adapter_path: str = ""
    # Host-RAM mirror of the frozen base, so it survives the colocate offload
    # instead of being re-shipped from the trainer every step.
    lora_base_cpu_backup: bool = True
    # Shares gate_up lora_A and down lora_B across experts. Roughly a third of
    # the per-expert parameter count, and a much larger step per parameter at a
    # given learning rate.
    experts_shared_outer_loras: bool = False

    # rollout
    num_rollout: int = 30
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    over_sampling_batch_size: int = 64
    rollout_max_response_len: int = 8192

    # rollout engine
    sglang_attention_backend: str | None = None
    sglang_lora_backend: str = "triton"
    sglang_mem_fraction_static: float = 0.7
    sglang_max_running_requests: int = 512
    rollout_num_gpus_per_engine: int = 1

    save_interval: int = 10
    enable_eval: bool = True
    extra_args: str = ""

    def __post_init__(self):
        self.hardware = U.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]


def _get_parallel_config(args: ScriptArgs) -> str:
    """Megatron parallel layout, for the topologies this recipe has been run on."""
    match (args.hardware, args.num_nodes):
        case ("MI350X" | "MI355X", 1):
            return (
                "--tensor-model-parallel-size 1 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 1 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 4 "
                "--expert-tensor-parallel-size 1 "
                "--recompute-granularity full "
                "--recompute-method uniform "
                "--recompute-num-layers 1 "
                "--use-dynamic-batch-size "
                "--max-tokens-per-gpu 16384 "
                "--micro-batch-size 1 "
            )
        case _:
            raise NotImplementedError(f"no verified layout for {args.hardware} on {args.num_nodes} node(s)")


def prepare(args: ScriptArgs):
    """Download the HF checkpoint and the training and eval datasets."""
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


def execute(args: ScriptArgs):
    """Run GRPO LoRA training (assumes ``prepare`` already ran)."""
    save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        "--megatron-to-hf-mode bridge "
        f"--save {save_path} "
        f"--save-interval {args.save_interval} "
    )
    if args.lora_adapter_path:
        ckpt_args += f"--lora-adapter-path {args.lora_adapter_path} "

    lora_args = (
        f"--lora-rank {args.lora_rank} "
        f"--lora-alpha {args.lora_alpha} "
        f"--lora-dropout {args.lora_dropout} "
        f'--target-modules "{args.target_modules}" '
    )
    if args.lora_base_cpu_backup:
        lora_args += "--lora-base-cpu-backup "
    if args.experts_shared_outer_loras:
        lora_args += "--experts-shared-outer-loras "
    else:
        lora_args += "--no-sglang-lora-use-virtual-experts "

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--rm-type deepscaler "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--balance-data "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
        f"--over-sampling-batch-size {args.over_sampling_batch_size} "
        "--dynamic-sampling-filter-path "
        "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
    )

    eval_args = ""
    if args.enable_eval:
        eval_args = (
            "--eval-interval 5 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            # AIME-2024 is 30 problems, so a single pass moves in steps of 3.3%;
            # averaging 8 samples keeps the eval readable at this interval.
            "--n-samples-per-eval-prompt 8 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    perf_args = _get_parallel_config(args)

    grpo_args = "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "

    optimizer_args = (
        "--optimizer adam "
        # A rank-32 adapter over a sparse MoE needs a much larger step than a
        # dense full-parameter run to move the policy at all.
        "--lr 2e-4 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-max-running-requests {args.sglang_max_running_requests} "
        f"--sglang-max-lora-rank {args.lora_rank} "
        # Triton is the only SGLang LoRA backend that applies adapters to MoE layers.
        f"--sglang-lora-backend {args.sglang_lora_backend} "
        "--sglang-moe-runner-backend triton "
        "--sglang-decode-log-interval 1000 "
    )
    if args.sglang_attention_backend not in (None, "default"):
        # Escape hatch for the throttled-engine ROCm attention fault; see docs/advanced/lora.md.
        # Off by default: the flags above do not throttle and the default backend runs this recipe.
        sglang_args += f"--sglang-attention-backend {args.sglang_attention_backend} "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--calculate-per-token-loss "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
    )

    train_args = (
        f"{ckpt_args} "
        f"{lora_args} "
        f"{rollout_args} "
        f"{eval_args} "
        f"{perf_args} "
        f"{grpo_args} "
        f"{optimizer_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
