"""Multi-LoRA fully-async GRPO example (Qwen3-4B, disaggregated 4 train + 4 rollout GPUs).

Trains multiple LoRA adapters concurrently on a shared base model. Two example
adapters ship in ``adapters/``: gsm8k (rm_type=math) and dapo_math
(rm_type=deepscaler); each carries its own rank/alpha, batch shape, dataset,
reward, and ``num_step`` stop condition. The driver is
``train_multi_lora_async.py`` at the repo root; fully-async training forbids
``--colocate`` (generation needs continuous GPU).

Usage:
  python examples/multi_lora/run_multi_lora.py prepare      # download Qwen3-4B + both datasets (once per node)
  python examples/multi_lora/run_multi_lora.py train        # bounded run: registers the two adapters, exits when each hits num_step
  python examples/multi_lora/run_multi_lora.py full-train   # prepare + train
  python examples/multi_lora/run_multi_lora.py serve        # service mode: no adapters preloaded, idles for registrations (API on :8068)

Service mode pairs with the smoke client:
  python examples/multi_lora/service_smoke.py --api-url http://127.0.0.1:8068 \\
      --data /root/datasets/gsm8k/train.parquet --input-key messages --label-key label --rm-type math
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_ADAPTER_DIR = f"{U.repo_base_dir}/examples/multi_lora/adapters"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()

    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    save_dir: str = "/tmp/multi_lora"
    megatron_path: str = "/root/Megatron-LM"

    # Disaggregated split (fully-async forbids colocate).
    num_gpus_per_node: int = 8
    actor_num_gpus: int = 4
    rollout_num_gpus: int = 4
    tp: int = 2

    # LoRA slot pool. Per-adapter rank/alpha come from adapter.yaml, capped by lora_rank.
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = "all-linear"
    n_adapters: int = 4
    # Comma-separated adapter names; each resolves to adapters/{name}.yaml (train mode only).
    adapters: str = "dapo_math,gsm8k"

    # Global rollout defaults; the per-adapter batch shapes live in the yamls.
    num_rollout: int = 50
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 4096
    global_batch_size: int = 256
    max_weight_staleness: int = 3

    # Service mode.
    api_port: int = 8068

    save_interval: int = 5
    enable_wandb: bool = False
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/Qwen3-4B"


def _prepare_download(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.data_dir} {args.model_dir}")
    U.exec_command(f"hf download Qwen/Qwen3-4B --local-dir {args.model_dir}/Qwen3-4B")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)


def _train(args: ScriptArgs, service: bool):
    mode = "service" if service else "bounded"
    print(
        f"[run] multi-LoRA ({mode}): {args.actor_num_gpus} train + {args.rollout_num_gpus} rollout GPUs, tp={args.tp}"
    )

    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "

    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} "
        f'--lora-dropout {args.lora_dropout} --target-modules "{args.target_modules}" '
    )

    multi_lora_args = f"--multi-lora-n-adapters {args.n_adapters} --multi-lora-idle-poll-s 5 "
    if service:
        # No adapters preloaded; the control-plane API accepts registrations at runtime.
        multi_lora_args += f"--multi-lora-api-port {args.api_port} "
    else:
        for name in args.adapters.split(","):
            multi_lora_args += f'--multi-lora-adapter "{name}" "{_ADAPTER_DIR}/{name}.yaml" '
        multi_lora_args += "--multi-lora-disable-service-mode "

    # in_place pause + upsert weight push is what lets adapters refresh without
    # unloading (an unload would deadlock behind paused in-flight requests).
    sync_args = f"--pause-generation-mode in_place --max-weight-staleness {args.max_weight_staleness} --use-tis "

    rollout_args = (
        "--apply-chat-template --rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
    )

    grpo_args = (
        "--advantage-estimator grpo --kl-loss-coef 0.00 --kl-coef 0.00 "
        "--entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {args.tp} --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 9216 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.8 "

    topology_args = (
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.actor_num_gpus} "
        f"--rollout-num-gpus {args.rollout_num_gpus} --use-miles-router "
    )

    save_args = f"--save {args.save_dir} --save-interval {args.save_interval} "

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash "
    )

    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""

    train_args = (
        f"{ckpt_args} {lora_args} {multi_lora_args} {sync_args} {rollout_args} {grpo_args} "
        f"{optimizer_args} {perf_args} {sglang_args} {topology_args} {save_args} {misc_args} "
        f"{wandb_args} {args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3-4B",
        train_script="train_multi_lora_async.py",
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download Qwen3-4B and both task datasets. Run once per node before training."""
    _prepare_download(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Bounded run: register the adapters from adapters/, train until each hits num_step, exit."""
    _train(args, service=False)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """Download model + datasets, then run the bounded training."""
    _prepare_download(args)
    _train(args, service=False)


@app.command()
@U.dataclass_cli
def serve(args: ScriptArgs):
    """Service mode: no adapters preloaded; register/deregister via the HTTP API while it idles."""
    _train(args, service=True)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
