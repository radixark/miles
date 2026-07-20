import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_FOUR_LAYER_HF = "/root/models/Kimi-K3-4layer-bf16"
_FOUR_LAYER_DCP = "/root/models/Kimi-K3-4layer-torch-dist"
_FULL_HF = "/root/models/yueming-model-support/native"
_FULL_DCP = "/root/models/yueming-model-support/torch_dist"
_FULL_SGLANG_CONFIG = Path(__file__).with_name("models") / "kimi-k3-lora-full-sglang.yaml"

_LAYERS = "decoder.layers.*"
_DEFAULT_TARGET_MODULES = ",".join(
    [
        f"{_LAYERS}.self_attention.o_proj",
        f"{_LAYERS}.self_attention.q_a_proj",
        f"{_LAYERS}.self_attention.kv_a_proj_with_mqa",
        f"{_LAYERS}.mlp.linear_fc1",
        f"{_LAYERS}.mlp.linear_fc2",
        f"{_LAYERS}.mlp.experts.linear_fc1",
        f"{_LAYERS}.mlp.experts.linear_fc2",
    ]
)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "debug_minimal"
    run_id: str = U.create_run_id()
    model_variant: Literal["4layer", "full"] = "4layer"
    task: Literal["gsm8k", "dapo-math"] = "gsm8k"
    hf_checkpoint: str = _FOUR_LAYER_HF
    ref_load: str = _FOUR_LAYER_DCP
    megatron_model_type: str = "kimi-k3-4layer"
    num_gpus_per_node: int = 8
    data_dir: str = "/root/datasets"
    save_dir: str = "/personal/checkpoints"
    megatron_path: str = "/root/Megatron-LM"
    sglang_path: str = "/root/sglang/python"
    checkpoint_load_mode: Literal["rank_local_cache", "shared"] = "rank_local_cache"
    local_checkpoint_cache_root: str | None = None

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = _DEFAULT_TARGET_MODULES
    experts_shared_outer_loras: bool = True
    lora_base_cpu_backup: bool = False
    check_lora_weight_equal: bool = False
    check_rollout_weight_reload_equal: bool = False

    reward_model: Literal["deterministic_random", "deepscaler", "math"] | None = None
    num_rollout: int | None = None
    rollout_batch_size: int | None = None
    n_samples_per_prompt: int | None = None
    rollout_max_response_len: int | None = None
    sglang_max_total_tokens: int | None = None
    global_batch_size: int | None = None
    distributed_timeout_minutes: int = 10
    save_debug_rollout_data: str | None = None
    enable_wandb: bool = False

    check_weight_update_equal: bool = False
    update_weight_buffer_size: int | None = None
    extra_args: str = ""

    def __post_init__(self):
        if self.lora_rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.lora_rank}")
        if self.sglang_max_total_tokens is not None and self.sglang_max_total_tokens <= 0:
            raise ValueError("SGLang max total tokens must be positive")
        if self.distributed_timeout_minutes <= 0:
            raise ValueError("Distributed timeout must be positive")
        if self.model_variant == "4layer":
            if self.num_nodes != 1 or self.num_gpus_per_node != 8:
                raise NotImplementedError("The verified four-layer Kimi K3 LoRA configuration is one 8-GPU node")
            return

        if self.num_nodes != 16 or self.num_gpus_per_node != 4:
            raise NotImplementedError("Full-model Kimi K3 LoRA requires 16 four-GPU GB300 nodes")
        if self.checkpoint_load_mode == "rank_local_cache" and self.local_checkpoint_cache_root is None:
            raise ValueError("Full-model Kimi K3 LoRA requires a rank-local checkpoint cache")
        if self.checkpoint_load_mode == "shared" and self.local_checkpoint_cache_root is not None:
            raise ValueError("Shared checkpoint load must not set a rank-local checkpoint cache")
        if self.hf_checkpoint == _FOUR_LAYER_HF:
            self.hf_checkpoint = _FULL_HF
        if self.ref_load == _FOUR_LAYER_DCP:
            self.ref_load = _FULL_DCP
        if self.megatron_model_type == "kimi-k3-4layer":
            self.megatron_model_type = "kimi-k3"

    @property
    def tensor_parallel_size(self) -> int:
        return 8 if self.model_variant == "4layer" else 32

    @property
    def expert_parallel_size(self) -> int:
        return 8 if self.model_variant == "4layer" else 64

    @property
    def rollout_num_gpus_per_engine(self) -> int:
        return 8

    @property
    def rollout_expert_parallel_size(self) -> int:
        return 8 if self.model_variant == "4layer" else 1


def _validate_paths(args: ScriptArgs) -> None:
    for name, path in (("hf_checkpoint", args.hf_checkpoint), ("ref_load", args.ref_load)):
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if not Path(args.sglang_path, "sglang").is_dir():
        raise FileNotFoundError(f"sglang package does not exist under sglang_path: {args.sglang_path}")


@app.command()
@U.dataclass_cli
def prepare_data(args: ScriptArgs) -> None:
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    dataset = "zhuzilin/gsm8k" if args.task == "gsm8k" else "zhuzilin/dapo-math-17k"
    U.hf_download_dataset(dataset, data_dir=args.data_dir)


def _execute_train(args: ScriptArgs) -> None:
    _validate_paths(args)
    if args.task == "gsm8k":
        dataset = Path(args.data_dir) / "gsm8k" / "train.parquet"
        input_key = "messages"
    else:
        dataset = Path(args.data_dir) / "dapo-math-17k" / "dapo-math-17k.jsonl"
        input_key = "prompt"
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset}; run prepare-data first")

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        "--megatron-to-hf-mode raw "
        "--model-name kimi_k3 "
    )

    lora_args = (
        f"--lora-rank {args.lora_rank} "
        f"--lora-alpha {args.lora_alpha} "
        f"--lora-dropout {args.lora_dropout} "
        f'--target-modules "{args.target_modules}" '
        "--no-gradient-accumulation-fusion "
    )
    if args.experts_shared_outer_loras:
        lora_args += "--experts-shared-outer-loras "
    if args.lora_base_cpu_backup:
        lora_args += "--lora-base-cpu-backup "
    if args.check_lora_weight_equal:
        lora_args += "--check-lora-weight-equal "
    if args.check_rollout_weight_reload_equal:
        lora_args += "--check-rollout-weight-reload-equal "

    is_debug = args.mode == "debug_minimal"
    reward_model = args.reward_model or (
        "math" if args.task == "gsm8k" else ("deterministic_random" if is_debug else "deepscaler")
    )
    num_rollout = args.num_rollout if args.num_rollout is not None else (2 if is_debug else 3000)
    rollout_batch_size = args.rollout_batch_size if args.rollout_batch_size is not None else (8 if is_debug else 32)
    n_samples_per_prompt = (
        args.n_samples_per_prompt if args.n_samples_per_prompt is not None else (2 if is_debug else 8)
    )
    rollout_max_response_len = (
        args.rollout_max_response_len
        if args.rollout_max_response_len is not None
        else (256 if args.task == "gsm8k" else (32 if is_debug else 16384))
    )
    global_batch_size = args.global_batch_size if args.global_batch_size is not None else (16 if is_debug else 256)

    rollout_args = (
        f"--prompt-data {dataset} "
        f"--input-key {input_key} "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--balance-data "
        f"--rm-type {reward_model} "
        f"--num-rollout {num_rollout} "
        f"--rollout-batch-size {rollout_batch_size} "
        f"--n-samples-per-prompt {n_samples_per_prompt} "
        f"--rollout-max-response-len {rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {global_batch_size} "
        "--use-dynamic-global-batch-size "
    )
    if args.save_debug_rollout_data is not None:
        rollout_args += f"--save-debug-rollout-data {args.save_debug_rollout_data} "

    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_parallel_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {args.expert_parallel_size} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {512 if args.model_variant == '4layer' else 1024} "
        "--log-probs-chunk-size 512 "
        f"--distributed-timeout-minutes {args.distributed_timeout_minutes} "
    )

    optimizer_args = (
        "--optimizer sgd --sgd-momentum 0 --lr 1e-6 --lr-decay-style constant "
        "--weight-decay 0 --use-distributed-optimizer "
        if args.mode == "debug_minimal"
        else (
            "--optimizer adam --lr 1e-5 --lr-decay-style constant "
            "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
            "--optimizer-cpu-offload --optimizer-offload-fraction 0.8 "
            "--overlap-cpu-optimizer-d2h-h2d "
            "--use-precision-aware-optimizer --use-distributed-optimizer "
        )
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.0 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    update_weight_buffer_size = args.update_weight_buffer_size
    if update_weight_buffer_size is None:
        update_weight_buffer_size = 256 * 1024**2 if args.model_variant == "full" else 2 * 1024**3
    # Each full-model TP8 engine serves one request at a time. K3's radix
    # extra-buffer strategy needs five KDA cache slots per running request.
    sglang_request_capacity = 1 if args.model_variant == "full" else 16
    sglang_mamba_capacity = 5 if args.model_variant == "full" else 16

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-tp-size {args.rollout_num_gpus_per_engine} "
        f"--sglang-ep-size {args.rollout_expert_parallel_size} "
        "--sglang-server-concurrency 16 "
        f"--sglang-max-running-requests {sglang_request_capacity} "
        f"--sglang-max-mamba-cache-size {sglang_mamba_capacity} "
        "--sglang-lora-backend triton "
        "--sglang-lora-strict-loading "
        f"--sglang-max-lora-rank {args.lora_rank} "
        "--use-miles-router "
    )
    if args.model_variant == "full":
        sglang_args += (
            f"--sglang-config {_FULL_SGLANG_CONFIG} "
            "--sglang-moe-runner-backend marlin "
            "--sglang-decode-attention-backend trtllm_mla "
            "--sglang-mamba-radix-cache-strategy extra_buffer "
            "--sglang-cuda-graph-bs-decode 1 "
            "--sglang-cuda-graph-backend-prefill disabled "
        )
    else:
        sglang_args += (
            "--sglang-cuda-graph-bs 1 2 4 8 16 "
            "--sglang-mem-fraction-static 0.7 "
            "--sglang-moe-runner-backend triton "
            "--sglang-disable-shared-experts-fusion "
        )
    if args.sglang_max_total_tokens is not None:
        sglang_args += f"--sglang-max-total-tokens {args.sglang_max_total_tokens} "
    if is_debug:
        sglang_args += "--sglang-context-length 8192 "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--colocate "
        "--offload-train "
        "--disable-weights-backuper "
        f"--update-weight-buffer-size {update_weight_buffer_size} "
        f"--train-memory-margin-bytes {(2 if args.mode == 'debug_minimal' else 4) * 1024**3} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )
    if args.model_variant == "4layer":
        misc_args += "--offload-rollout-level kv_cache --no-check-for-nan-in-loss-and-grad "
    else:
        misc_args += (
            "--offload-rollout-level kv_cache weight "
            "--reload-rollout-weights-from-disk "
            "--drop-checkpoint-page-cache-after-load "
        )
    if args.check_weight_update_equal:
        misc_args += "--check-weight-update-equal " "--check-weight-update-skip-list vision_tower. mm_projector. "

    if args.enable_wandb:
        wandb_args = (
            "--use-wandb "
            "--wandb-project miles-run_kimi_k3_lora "
            f"--wandb-group {args.run_id} "
            "--disable-wandb-random-suffix "
        )
    else:
        wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id)

    train_args = (
        f"{ckpt_args} "
        f"{lora_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{wandb_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"--save {args.save_dir}/{args.run_id} --save-interval 1 "
        f"{args.extra_args} "
    )
    extra_env_vars = {
        "NCCL_TIMEOUT": "3600",
        "PYTHONPATH": os.pathsep.join((str(Path(__file__).resolve().parents[1]), args.sglang_path)),
        "SGLANG_JIT_ROUTE_RADIX": "1",
    }
    if args.checkpoint_load_mode == "rank_local_cache" and args.local_checkpoint_cache_root is not None:
        extra_env_vars["MILES_MEGATRON_LOCAL_CHECKPOINT_CACHE"] = args.local_checkpoint_cache_root

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs) -> None:
    _execute_train(args)


if __name__ == "__main__":
    app()
