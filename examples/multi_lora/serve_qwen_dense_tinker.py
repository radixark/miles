"""Serve dense Qwen models through Miles' Tinker-compatible multi-LoRA gateway.

The default is a disaggregated three-node B300 deployment: two eight-GPU
trainer nodes and one eight-GPU SGLang node. Set ``MILES_SCRIPT_EXTERNAL_RAY=1``
on the head of an existing three-node Ray cluster before launching it.
"""

from dataclasses import dataclass, field
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass(frozen=True)
class _Recipe:
    megatron_model_type: str
    tp: int
    pp: int
    sglang_mem_fraction_static: float


_RECIPES = {
    "Qwen/Qwen3.5-4B": _Recipe("qwen3.5-4B", tp=2, pp=2, sglang_mem_fraction_static=0.70),
    "Qwen/Qwen3.6-27B": _Recipe("qwen3.6-27B", tp=4, pp=2, sglang_mem_fraction_static=0.50),
    "Qwen/Qwen3.8-27B": _Recipe("qwen3.8-27B", tp=4, pp=2, sglang_mem_fraction_static=0.80),
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    base_model: Literal["Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B"] = "Qwen/Qwen3.5-4B"
    hf_checkpoint: str | None = None
    model_dir: str = "/data/model-cache/models"
    save_dir: str | None = None
    megatron_path: str = "/root/Megatron-LM"

    num_gpus_per_node: int = 8
    actor_num_nodes: int = 2
    actor_num_gpus_per_node: int = 8
    rollout_num_gpus: int = 8

    lora_rank: int = 32
    lora_alpha: int = 32
    n_adapters: int = 4
    target_modules: str = "attn,mlp"

    tinker_port: int = 10613
    rollout_num_gpus_per_engine: int = 1
    context_length: int = 65504
    extra_args: str = ""

    @property
    def recipe(self) -> _Recipe:
        return _RECIPES[self.base_model]

    def __post_init__(self) -> None:
        trainer_gpus = self.actor_num_nodes * self.actor_num_gpus_per_node
        if (self.num_gpus_per_node, trainer_gpus, self.rollout_num_gpus) != (8, 16, 8):
            raise ValueError("dense Qwen 64K preset requires two 8-GPU trainer nodes and one 8-GPU sampler node")
        if trainer_gpus % (self.recipe.tp * self.recipe.pp) != 0:
            raise ValueError("trainer GPU count must be divisible by TP * PP")
        if self.rollout_num_gpus_per_engine != 1:
            raise ValueError("dense Qwen rollout requires one-GPU SGLang engines")
        if self.context_length != 65504:
            raise ValueError("dense Qwen preset is validated for a 65,504-token service limit")
        model_name = self.base_model.rsplit("/", 1)[-1]
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{model_name}"
        if self.save_dir is None:
            self.save_dir = f"{self.output_dir}/checkpoints"


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs) -> None:
    """Download the selected Hugging Face checkpoint into the persistent cache."""
    backend = args.create_backend()
    backend.exec_command_cpu(f"mkdir -p {args.hf_checkpoint}")
    backend.exec_command_cpu(f"hf download {args.base_model} --local-dir {args.hf_checkpoint}")


def _serve(args: ScriptArgs) -> None:
    recipe = args.recipe
    print(
        f"[run] {args.base_model}: 16 train GPUs (TP{recipe.tp} PP{recipe.pp}) + "
        f"{args.rollout_num_gpus} inference GPUs, 64K Tinker :{args.tinker_port}"
    )

    checkpoint_args = f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "
    lora_args = f'--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout 0.0 --no-gradient-accumulation-fusion --multi-lora-n-adapters {args.n_adapters} --target-modules "{args.target_modules}" '
    tinker_args = f"--tinker-base-model {args.base_model} --tinker-server-port {args.tinker_port} --tinker-checkpoint-root {args.save_dir}/{args.run_id} "
    optimizer_args = "--optimizer adam --lr 1e-4 "
    parallel_args = f"--tensor-model-parallel-size {recipe.tp} --sequence-parallel --pipeline-model-parallel-size {recipe.pp} --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
    batching_args = f"--seq-length {args.context_length} --rollout-max-context-len {args.context_length} --micro-batch-size 1 --use-dynamic-batch-size --max-tokens-per-gpu {args.context_length} --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
    topology_args = f"--actor-num-nodes {args.actor_num_nodes} --actor-num-gpus-per-node {args.actor_num_gpus_per_node} --rollout-num-gpus {args.rollout_num_gpus} --num-gpus-per-node {args.num_gpus_per_node} "
    sglang_args = f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} --sglang-mem-fraction-static {recipe.sglang_mem_fraction_static} --sglang-lora-backend triton "
    model_args = "--qkv-format thd --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash "
    train_args = f"{checkpoint_args}{lora_args}{tinker_args}{optimizer_args}{parallel_args}{batching_args}{topology_args}{sglang_args}{model_args}{args.extra_args}"
    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=recipe.megatron_model_type,
        train_script="serve_tinker.py",
        megatron_path=args.megatron_path,
        job_lifetime="launcher",
    )


@app.command()
@U.dataclass_cli
def serve(args: ScriptArgs) -> None:
    """Start a disaggregated 16-trainer/8-sampler Tinker gateway."""
    _serve(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
