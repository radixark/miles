"""Tinker-compatible backend example (Qwen3-4B, disaggregated 4 train + 4 rollout GPUs).

Serves the operation API for client-driven LoRA training: no datasets, no
reward functions — clients enqueue forward_backward/optim_step operations and
sample through the shared engines. The driver is ``train_tinker_backend.py``
at the repo root.

Usage:
  python examples/tinker_backend/run_tinker_backend.py prepare   # download Qwen3-4B (once per node)
  python examples/tinker_backend/run_tinker_backend.py serve     # service mode: idles for registrations (API on :8068)
  python examples/tinker_backend/run_tinker_backend.py train     # pre-registers adapters/example.yaml, exits when it retires
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_ADAPTER_DIR = f"{U.repo_base_dir}/examples/tinker_backend/adapters"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()

    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    save_dir: str = "/tmp/tinker_backend"
    megatron_path: str = "/root/Megatron-LM"

    # Disaggregated split (the operation backend forbids colocate).
    num_gpus_per_node: int = 8
    actor_num_gpus: int = 4
    rollout_num_gpus: int = 4
    tp: int = 2

    # LoRA slot pool: clients may register with rank <= lora_rank; alpha is fixed here.
    lora_rank: int = 32
    lora_alpha: int = 64
    target_modules: str = "all-linear"
    n_adapters: int = 4
    adapters: str = "example"

    # Soft coalescing target for one train call (whole client batches only).
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 1
    global_batch_size: int = 32
    use_dynamic_batch_size: bool = True

    api_port: int = 8068
    enable_wandb: bool = False
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/Qwen3-4B"


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download Qwen3-4B. Run once per node before serving."""
    U.exec_command_cpu(f"mkdir -p {args.model_dir}")
    U.exec_command_cpu(f"hf download Qwen/Qwen3-4B --local-dir {args.model_dir}/Qwen3-4B")


def _serve(args: ScriptArgs, service: bool):
    mode = "service" if service else "bounded"
    print(f"[run] tinker backend ({mode}): {args.actor_num_gpus} train + {args.rollout_num_gpus} rollout GPUs")

    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "
    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} "
        f'--lora-dropout 0.0 --target-modules "{args.target_modules}" '
    )
    tinker_args = f"--tinker-backend --multi-lora-n-adapters {args.n_adapters} --multi-lora-idle-poll-s 5 "
    if service:
        tinker_args += f"--multi-lora-api-port {args.api_port} "
    else:
        for name in args.adapters.split(","):
            tinker_args += f'--multi-lora-adapter "{name}" "{_ADAPTER_DIR}/{name}.yaml" '
        tinker_args += "--multi-lora-disable-service-mode "

    # in_place pause + upsert push: adapters publish without unloading.
    sync_args = "--pause-generation-mode in_place "

    rollout_args = (
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--global-batch-size {args.global_batch_size} "
        "--num-rollout 1000000 "
    )

    optimizer_args = "--optimizer adam --lr 1e-4 --lr-decay-style constant "

    dynamic_batch_args = (
        "--use-dynamic-batch-size --max-tokens-per-gpu 9216 " if args.use_dynamic_batch_size else ""
    )
    perf_args = (
        f"--tensor-model-parallel-size {args.tp} --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        f"{dynamic_batch_args}"
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.8 "
    topology_args = (
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.actor_num_gpus} "
        f"--rollout-num-gpus {args.rollout_num_gpus} "
    )
    # Tinker checkpoints move only through save_state operations, but megatron
    # arg validation requires a save interval whenever --save is set.
    save_args = f"--save {args.save_dir} --save-interval 1000000 "
    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash "
    )
    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""

    train_args = (
        f"{ckpt_args} {lora_args} {tinker_args} {sync_args} {rollout_args} "
        f"{optimizer_args} {perf_args} {sglang_args} {topology_args} {save_args} {misc_args} "
        f"{wandb_args} {args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3-4B",
        train_script="train_tinker_backend.py",
        megatron_path=args.megatron_path,
        extra_env_vars={
            # TinkerRolloutFn is class-based: it needs the experimental rollout API.
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        },
    )


@app.command()
@U.dataclass_cli
def serve(args: ScriptArgs):
    """Service mode: no adapters preloaded; register via the HTTP API while it idles."""
    _serve(args, service=True)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Bounded run: pre-register adapters/, exit when every registration retires."""
    _serve(args, service=False)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
