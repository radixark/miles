"""Offline DiffusionGemma text SFT with Miles FSDP2; no inference engine.

Requires the Miles training image, a local HF DiffusionGemma checkpoint, and
JSONL/Parquet rows with a `messages` conversation ending in a labeled assistant
answer. Uses Transformers' differentiable modules and SDPA with explicit masks.
The initial full-parameter recipe needs a multi-GPU CUDA host; its memory/throughput
must be measured for the chosen sequence lengths and hardware.

python scripts/run_diffusiongemma_26b_a4b_fsdp_sft.py \
  --hf-checkpoint /models/diffusiongemma-26B-A4B-it --data-path /data/sft.jsonl \
  --output-dir /runs/diffusion-sft --num-gpus-per-node 8

Args: --hf-checkpoint, --data-path, --output-dir are shared paths on all workers.
--num-nodes / --num-gpus-per-node set topology; --global-batch-size must divide
the rollout batch and be divisible by total GPUs * --micro-batch-size.
--extra-args supplies additional standard Miles flags.
"""

import shlex
from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    hf_checkpoint: str = "/root/models/diffusiongemma-26B-A4B-it"
    data_path: str = "/root/datasets/sft.jsonl"
    num_gpus_per_node: int = 8
    global_batch_size: int = 8
    rollout_batch_size: int = 8
    micro_batch_size: int = 1
    num_epoch: int = 1
    lr: float = 1e-5
    save_interval: int = 10
    extra_args: str = ""


def execute(args: ScriptArgs):
    world_size = args.num_nodes * args.num_gpus_per_node
    if min(world_size, args.global_batch_size, args.rollout_batch_size, args.micro_batch_size) < 1:
        raise ValueError("GPU and batch counts must be positive")
    if args.global_batch_size % (world_size * args.micro_batch_size):
        raise ValueError("global_batch_size must be divisible by total GPUs * micro_batch_size")
    if args.rollout_batch_size % args.global_batch_size:
        raise ValueError("rollout_batch_size must be divisible by global_batch_size")
    U = args.create_backend()
    checkpoint_dir = shlex.quote(f"{args.output_dir}/checkpoints")
    train_args = (
        "--train-backend fsdp "
        f"--hf-checkpoint {shlex.quote(args.hf_checkpoint)} "
        f"--prompt-data {shlex.quote(args.data_path)} --input-key messages "
        "--rollout-function-path miles.rollout.diffusion_gemma_sft.generate_rollout "
        "--rollout-shuffle --n-samples-per-prompt 1 --debug-train-only "
        "--loss-type sft_loss --disable-compute-advantages-and-returns "
        "--qkv-format bshd --attn-implementation sdpa --gradient-checkpointing "
        f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-batch-size {args.rollout_batch_size} --global-batch-size {args.global_batch_size} "
        f"--micro-batch-size {args.micro_batch_size} --num-epoch {args.num_epoch} "
        f"--load {checkpoint_dir} --save {checkpoint_dir} --save-interval {args.save_interval} "
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant --weight-decay 0.01 "
        "--diffusion-noise-epsilon 0.001 --diffusion-self-conditioning-probability 0.5 "
        "--diffusion-encoder-loss-weight 1.0 --diffusion-freeze-router "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} {args.extra_args}"
    )
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=None,
        # Serial dataset consumption keeps the saved offset aligned with the
        # completed optimizer step; async prefetch can save an untrained batch.
        train_script="train.py",
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
