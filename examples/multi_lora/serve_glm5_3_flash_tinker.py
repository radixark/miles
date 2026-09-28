"""Serve GLM-5.3-Flash through Miles' Tinker-compatible multi-LoRA gateway.

The production topology is disaggregated: two 8-GPU trainer nodes and one
8-GPU SGLang node. A three-node Ray cluster must already be running and
``MILES_SCRIPT_EXTERNAL_RAY=1`` must be set on the head before invoking this
launcher.  The first gate deliberately trains MLP/MoE adapters only because
the GLM-5.3 SGLang fork cannot yet apply LoRA to its custom fused KDA q/k/v
operators.
"""

from dataclasses import dataclass, field
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U
from scripts.run_glm5_3_flash import build_model_args, build_parallel_args, build_sglang_args

app = typer.Typer()
_HF_CHECKPOINT_REPOS = {
    "GLM-5.3-Flash": "zai-org/GLM-5.3-Flash-BF16",
    "GLM-5.3-Flash-4layer": "CharyZeng/GLM-5.3-Flash-4layer",
}
_MEGATRON_MODEL_TYPES = {
    "GLM-5.3-Flash": "glm5.3-flash",
    "GLM-5.3-Flash-4layer": "glm5.3-flash-4layer",
}
_ROLLOUT_PORTABLE_TARGETS = ",".join(f"model.language_model.layers.*.{prefix}.{projection}" for prefix in ("mlp", "mlp.shared_experts") for projection in ("gate_proj", "up_proj", "down_proj"))


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    model_name: Literal["GLM-5.3-Flash", "GLM-5.3-Flash-4layer"] = "GLM-5.3-Flash"
    base_model: str = "zai-org/GLM-5.3-Flash"
    # Train from the unquantized release. The default GLM-5.3-Flash repository
    # is FP8 and Multi-LoRA rejects quantized expert weights.
    hf_checkpoint: str | None = None
    save_dir: str = "/checkpoints"
    ref_load: str | None = None
    megatron_path: str = "/root/Megatron-LM"

    num_gpus_per_node: int = 8
    actor_num_nodes: int = 2
    actor_num_gpus_per_node: int = 8
    rollout_num_gpus: int = 8
    tp: int = 8
    pp: int = 2
    ep: int = 8

    lora_rank: int = 32
    lora_alpha: int = 32
    n_adapters: int = 4
    target_modules: str = _ROLLOUT_PORTABLE_TARGETS

    tinker_port: int = 10613
    context_length: int = 65504
    sglang_mem_fraction_static: float = 0.70
    extra_args: str = ""

    def __post_init__(self) -> None:
        if self.hf_checkpoint is None:
            repo_name = _HF_CHECKPOINT_REPOS[self.model_name].rsplit("/", 1)[-1]
            self.hf_checkpoint = f"/data/model-cache/models/{repo_name}"
        trainer_gpus = self.actor_num_nodes * self.actor_num_gpus_per_node
        expected_topologies = {
            ("GLM-5.3-Flash", 16): (16, 8, 8, 2, 8),
            ("GLM-5.3-Flash", 32): (32, 8, 8, 4, 8),
            ("GLM-5.3-Flash-4layer", 4): (4, 4, 2, 2, 2),
        }
        expected = expected_topologies.get((self.model_name, trainer_gpus))
        actual = (trainer_gpus, self.rollout_num_gpus, self.tp, self.pp, self.ep)
        if expected is None or actual != expected:
            raise ValueError(f"{self.model_name} gate requires trainer/rollout/TP/PP/EP={expected}, got {actual}")
        if trainer_gpus % (self.tp * self.pp) != 0:
            raise ValueError("trainer GPU count must be divisible by TP * PP")
        if self.target_modules != _ROLLOUT_PORTABLE_TARGETS:
            raise ValueError("first GLM-5.3-Flash gate supports only rollout-portable MLP/MoE LoRA")


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs) -> None:
    """Populate this node's persistent model cache with the BF16 checkpoint."""
    backend = args.create_backend()
    backend.exec_command_cpu(f"mkdir -p {args.hf_checkpoint}")
    backend.exec_command_cpu(f"hf download {_HF_CHECKPOINT_REPOS[args.model_name]} --local-dir {args.hf_checkpoint}")


def _serve(args: ScriptArgs) -> None:
    if args.model_name == "GLM-5.3-Flash-4layer" and args.ref_load:
        checkpoint_args = f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.ref_load} --megatron-to-hf-mode raw "
    else:
        checkpoint_args = f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "
    lora_args = f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout 0.0 --multi-lora-n-adapters {args.n_adapters} --target-modules {args.target_modules} --no-gradient-accumulation-fusion "
    tinker_args = f"--tinker-base-model {args.base_model} --tinker-server-port {args.tinker_port} --tinker-checkpoint-root {args.save_dir}/{args.run_id} "
    topology_args = f"--actor-num-nodes {args.actor_num_nodes} --actor-num-gpus-per-node {args.actor_num_gpus_per_node} --rollout-num-gpus {args.rollout_num_gpus} --num-gpus-per-node {args.num_gpus_per_node} "
    if args.model_name == "GLM-5.3-Flash":
        parallel_args, canonical_engine_gpus = build_parallel_args(args.model_name, args.actor_num_nodes * args.actor_num_gpus_per_node)
    else:
        parallel_args = "--tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 2 --context-parallel-size 1 --expert-model-parallel-size 2 --expert-tensor-parallel-size 1 --qkv-format thd "
        canonical_engine_gpus = 4
    assert canonical_engine_gpus == args.rollout_num_gpus
    batching_args = f"--seq-length {args.context_length} --rollout-max-context-len {args.context_length} --max-tokens-per-gpu {args.context_length} --micro-batch-size 1 --use-dynamic-batch-size "
    sglang_args = f"{build_sglang_args(canonical_engine_gpus)}--sglang-mem-fraction-static {args.sglang_mem_fraction_static} --sglang-lora-backend triton "
    optimizer_args = "--optimizer adam --lr 1e-4 "
    model_args = build_model_args()
    train_args = f"{checkpoint_args}{lora_args}{tinker_args}{topology_args}{parallel_args}{batching_args}{sglang_args}{optimizer_args}{model_args}{args.extra_args}"
    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=_MEGATRON_MODEL_TYPES[args.model_name],
        train_script="serve_tinker.py",
        megatron_path=args.megatron_path,
        job_lifetime="launcher",
        extra_env_vars={"TORCHINDUCTOR_COMPILE_THREADS": "1"},
    )


@app.command()
@U.dataclass_cli
def serve(args: ScriptArgs) -> None:
    """Start a disaggregated Tinker gateway on an existing Ray cluster."""
    _serve(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
