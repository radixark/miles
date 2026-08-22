"""Qwen3 dense GRPO LoRA training through the native (raw-mode) LoRA path.

This is the reference / validation recipe for ``miles_plugins.lora``:
LoRA is applied directly to the mcore model built by miles' own model provider
(``--megatron-to-hf-mode raw``) instead of going through Megatron-Bridge. Adapters
are exported under HF/PEFT names and shipped to SGLang with the same adapter sync
the bridge path uses, so a run here exercises the whole loop: frozen base +
adapter grads, TP/EP grad summation, adapter-only weight sync, and LoRA serving.

Qwen3-8B / Qwen3-4B / Qwen3-0.6B are dense GQA models with a SwiGLU MLP, which is
exactly the layout the generic implementation covers. TP2 with sequence parallelism
is the interesting configuration: it exercises both the column-parallel (A
replicated, B row-sharded) and row-parallel (A col-sharded, B replicated)
grad-summation paths. Qwen3-0.6B on 2 GPUs is the cheapest end-to-end check.

Qwen3-30B-A3B is included as a MoE case, but note the coverage: every one of its 48
layers is a pure routed-MoE block (no dense MLP, no shared expert), so only attention
(q/k/v/o) gets adapters there. Routed-expert adapters are out of scope for the generic
provider -- pass ``--target-modules "q_proj,k_proj,v_proj,o_proj"`` so the trainer and
the engine agree on exactly which modules carry an adapter.

Note on Qwen3.5: the gated ``linear_qkv`` and the GDN hybrid layout are covered by
``examples/lora/run_lora_native.py``, which drives the MoE registries. Bridge mode
(``scripts/run_qwen3_5_35b_a3b_lora.py``) remains the recommended path for Qwen3.5 for
now: raw mode's GDN layers produce an unstable backward once the base is frozen.

Usage:
  python examples/lora/run_qwen3_lora_native.py prepare    --model-name Qwen3-8B
  python examples/lora/run_qwen3_lora_native.py full-train --model-name Qwen3-8B --task gsm8k
  python examples/lora/run_qwen3_lora_native.py train      --model-name Qwen3-4B --task dapo-math
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_MEGATRON_MODEL_TYPE = {
    "Qwen3-8B": "qwen3-8B",
    "Qwen3-4B": "qwen3-4B",
    "Qwen3-0.6B": "qwen3-0.6B",
    "Qwen3-30B-A3B": "qwen3-30B-A3B",
}

_NUM_QUERY_GROUPS = {"Qwen3-8B": 8, "Qwen3-4B": 8, "Qwen3-0.6B": 8, "Qwen3-30B-A3B": 4}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: Literal["Qwen3-8B", "Qwen3-4B", "Qwen3-0.6B", "Qwen3-30B-A3B"] = "Qwen3-8B"
    task: Literal["gsm8k", "dapo-math"] = "gsm8k"

    hf_checkpoint: str | None = None
    torch_dist: str | None = None
    model_dir: str = "/root/models"
    save_dir: str = "/personal/checkpoints"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    num_gpus_per_node: int = 8
    tensor_model_parallel_size: int = 2

    lr: float = 1e-5
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = "all-linear"
    lora_adapter_path: str | None = None
    debug_lora_train_only: bool = False
    check_lora_weight_equal: bool = False

    num_rollout: int = 10
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 0
    global_batch_size: int = 64

    rollout_num_gpus_per_engine: int = 2
    sglang_mem_fraction_static: float = 0.6
    sglang_lora_backend: str = "triton"

    enable_wandb: bool = True
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"
        if self.torch_dist is None:
            self.torch_dist = f"{self.model_dir}/{self.model_name}_torch_dist"
        if self.rollout_max_response_len == 0:
            self.rollout_max_response_len = 4096 if self.task == "dapo-math" else 2048

    @property
    def megatron_model_type(self) -> str:
        return _MEGATRON_MODEL_TYPE[self.model_name]


def _get_parallel_config(args: ScriptArgs) -> str:
    """TP with sequence parallelism, DP over the remaining GPUs.

    TP must stay <= num_query_groups (see _NUM_QUERY_GROUPS).
    """
    groups = _NUM_QUERY_GROUPS[args.model_name]
    assert (
        args.tensor_model_parallel_size <= groups
    ), f"{args.model_name} has num_query_groups={groups}; native LoRA needs TP <= {groups}"
    perf = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--micro-batch-size 1 --max-tokens-per-gpu 9216 "
    )
    if args.tensor_model_parallel_size > 1:
        perf += "--sequence-parallel "
    return perf


def _download_dataset(args: ScriptArgs):
    match args.task:
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)
        case "dapo-math":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def _prepare_download(args: ScriptArgs):
    """Download the checkpoint + dataset and build the frozen base as a torch-dist checkpoint.

    The conversion has to go through ``U.convert_checkpoint``: it sources
    ``scripts/models/<megatron_model_type>.sh`` and passes ``MODEL_ARGS``, without which
    ``convert_hf_to_torch_dist.py`` has no ``--num-layers`` and cannot run.
    """
    U.exec_command_cpu(f"mkdir -p {args.data_dir} {args.model_dir}")
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.hf_checkpoint}")
    _download_dataset(args)
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=1,
        dir_dst=args.model_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
        extra_args="--bf16 ",
    )


def _train(args: ScriptArgs):
    print(
        f"[run] Qwen3 native LoRA: model={args.model_name} "
        f"(megatron_model_type={args.megatron_model_type}), {args.num_gpus_per_node} GPUs, "
        f"TP{args.tensor_model_parallel_size}, rollout tp={args.rollout_num_gpus_per_engine}"
    )
    load_save_path = f"{args.save_dir}/{args.run_id}"

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --load {args.torch_dist} "
        "--megatron-to-hf-mode raw --no-load-optim --no-load-rng --finetune "
    )

    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} "
        f'--lora-dropout {args.lora_dropout} --target-modules "{args.target_modules}" '
        "--no-gradient-accumulation-fusion "
    )
    if args.lora_adapter_path is not None:
        lora_args += f"--lora-adapter-path {args.lora_adapter_path} "
    if args.debug_lora_train_only:
        lora_args += "--debug-lora-train-only "
    if args.check_lora_weight_equal:
        lora_args += "--check-lora-weight-equal "

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1.0 "
        f"--global-batch-size {args.global_batch_size} "
    )
    match args.task:
        case "gsm8k":
            rollout_args += f"--prompt-data {args.data_dir}/gsm8k/train.parquet --input-key messages "
        case "dapo-math":
            rollout_args += f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt "

    grpo_args = "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "

    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant "
        "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )

    perf_args = _get_parallel_config(args)

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-dtype bfloat16 --sglang-decode-log-interval 1000 "
        f"--sglang-max-lora-rank {args.lora_rank} "
        f"--sglang-lora-backend {args.sglang_lora_backend} "
    )

    save_args = f"--save-interval 5 --save {load_save_path} "

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--update-weight-buffer-size 536870912 "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} --colocate "
    )

    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""

    train_args = (
        f"{ckpt_args} {lora_args} {rollout_args} {optimizer_args} {grpo_args} "
        f"{wandb_args} {perf_args} {sglang_args} {save_args} {misc_args} {args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download the checkpoint + dataset and convert the base to a torch-dist checkpoint."""
    _prepare_download(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run GRPO LoRA training through the native path (assumes prepare already ran)."""
    _train(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """Prepare, then run GRPO LoRA training through the native path."""
    _prepare_download(args)
    _train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
