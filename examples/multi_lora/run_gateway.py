"""Tinker gateway example: concurrent LoRA clients on one shared base model.

Launches ``serve_tinker.py`` with Qwen3-30B-A3B on one node, split into
``actor_num_gpus`` training GPUs and ``rollout_num_gpus`` sampling GPUs
(multi-LoRA forbids ``--colocate``). Adapters cover attention and the
per-expert MoE projections; sequences pack as thd. The gateway speaks the
Tinker protocol on ``--tinker-port``; drive it with ``client.py`` and the
official ``tinker`` SDK.

Usage:
  python examples/multi_lora/run_gateway.py prepare   # download Qwen3-30B-A3B (once per node)
  python examples/multi_lora/run_gateway.py serve     # gateway on :10613
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()

    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    save_dir: str = "/personal/checkpoints"
    megatron_path: str = "/root/Megatron-LM"

    # Disaggregated split on one node.
    num_gpus_per_node: int = 8
    actor_num_gpus: int = 4
    rollout_num_gpus: int = 4
    tp: int = 2
    ep: int = 4

    # LoRA slot pool; per-client rank comes from the SDK, capped by lora_rank.
    lora_rank: int = 32
    lora_alpha: int = 64
    n_adapters: int = 4
    target_modules: str = "linear_qkv,linear_proj,linear_fc1,linear_fc2"

    tinker_port: int = 10613
    rollout_num_gpus_per_engine: int = 2
    sglang_mem_fraction_static: float = 0.7

    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/Qwen3-30B-A3B"


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download the Qwen3-30B-A3B checkpoint. Run once per node before serving."""
    U.exec_command_cpu(f"mkdir -p {args.model_dir}")
    U.exec_command_cpu(f"hf download Qwen/Qwen3-30B-A3B --local-dir {args.model_dir}/Qwen3-30B-A3B")


@app.command()
@U.dataclass_cli
def serve(args: ScriptArgs):
    """Serve the Tinker gateway (idles until clients connect)."""
    print(
        f"[run] tinker gateway: {args.actor_num_gpus} train + {args.rollout_num_gpus} rollout GPUs, "
        f"{args.n_adapters} adapter slots, port {args.tinker_port}"
    )

    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "

    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout 0.0 "
        f'--target-modules "{args.target_modules}" '
        "--no-gradient-accumulation-fusion "
        f"--multi-lora-n-adapters {args.n_adapters} "
    )

    tinker_args = (
        f"--tinker-server-port {args.tinker_port} " f"--tinker-checkpoint-root {args.save_dir}/{args.run_id} "
    )

    # initial config only; AdamParams come per optim_step request
    optimizer_args = "--optimizer adam --lr 1e-4 "

    perf_args = (
        f"--tensor-model-parallel-size {args.tp} --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--expert-model-parallel-size {args.ep} --expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 8192 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-lora-backend triton "
    )

    topology_args = (
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.actor_num_gpus} "
        f"--rollout-num-gpus {args.rollout_num_gpus} "
    )

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash "
    )

    train_args = (
        f"{ckpt_args} {lora_args} {tinker_args} {optimizer_args} "
        f"{perf_args} {sglang_args} {topology_args} {misc_args} {args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3-30B-A3B",
        train_script="serve_tinker.py",
        megatron_path=args.megatron_path,
    )


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
