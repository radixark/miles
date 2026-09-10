"""Kimi K3 RL launcher: LoRA or full-parameter training on the native MXFP4 checkpoint.

python scripts/run_kimi_k3.py prepare-download --model-name Kimi-K3-4layer
python scripts/run_kimi_k3.py prepare-bf16 --model-name Kimi-K3-4layer
python scripts/run_kimi_k3.py prepare-torch-dist --model-name Kimi-K3-4layer
python scripts/run_kimi_k3.py train --model-name Kimi-K3-4layer --train-mode lora
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_DEFAULT_MODEL_ORG = {
    "Kimi-K3": "moonshotai",
    # 4-layer prune of the native MXFP4 checkpoint (1 dense + 3 MoE layers).
    "Kimi-K3-4layer": "Pinaster",
}
_MEGATRON_MODEL_TYPE = {"Kimi-K3": "kimi-k3", "Kimi-K3-4layer": "kimi-k3-4layer"}
_NUM_LAYERS = {"Kimi-K3": 93, "Kimi-K3-4layer": 4}
_NUM_ATTENTION_HEADS = 96
_NUM_EXPERTS = 896
_VALIDATED_FULL_MODEL_GPUS = 64

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
    model_org: str = ""
    model_name: Literal["Kimi-K3", "Kimi-K3-4layer"] = "Kimi-K3-4layer"
    train_mode: Literal["lora", "full"] = "lora"
    task: Literal["gsm8k", "dapo-math"] = "gsm8k"
    hardware: Literal["auto", "H100", "H200", "B200", "B300", "GB200", "GB300"] = "auto"
    num_gpus_per_node: int | None = None

    # native MXFP4 checkpoint (rollout), its BF16 dequantization and the torch_dist conversion (trainer);
    # None derives each from model_dir/model_name
    hf_checkpoint: str | None = None
    bf16_checkpoint: str | None = None
    ref_load: str | None = None
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    save_dir: str = "/personal/checkpoints"
    megatron_path: str = "/root/Megatron-LM"
    sglang_path: str = "/root/sglang/python"

    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    # for layouts the PP-only derivation cannot express, e.g. TP4/CP2/PP4/EP8 (DP=2)
    tp_size_override: int | None = None
    ep_size_override: int | None = None
    rollout_tp_size: int | None = None
    rollout_ep_size: int | None = None
    rollout_max_concurrency: int = 64

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = _DEFAULT_TARGET_MODULES
    experts_shared_outer_loras: bool = True

    reward_model: Literal["deterministic_random", "deepscaler", "math"] | None = None
    num_rollout: int | None = None
    rollout_batch_size: int | None = None
    n_samples_per_prompt: int | None = None
    rollout_max_response_len: int | None = None
    sglang_max_total_tokens: int | None = None
    global_batch_size: int | None = None
    eval_interval: int | None = None
    lr: float | None = None
    max_tokens_per_gpu: int | None = None
    distributed_timeout_minutes: int = 10
    save_debug_rollout_data: str | None = None
    enable_wandb: bool = False
    skip_saving: bool = False

    check_weight_update_equal: bool = False
    check_lora_weight_equal: bool = False
    update_weight_buffer_size: int | None = None
    extra_args: str = ""

    def __post_init__(self):
        self.hardware = U.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]
        if not self.model_org:
            self.model_org = _DEFAULT_MODEL_ORG[self.model_name]
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"
        if self.bf16_checkpoint is None:
            self.bf16_checkpoint = f"{self.model_dir}/{self.bf16_name}"
        if self.ref_load is None:
            self.ref_load = f"{self.model_dir}/{self.bf16_name}_torch_dist"
        if self.lr is None:
            self.lr = 1e-5 if self.train_mode == "lora" else 1e-6
        if self.rollout_tp_size is None:
            self.rollout_tp_size = min(8, self.num_gpus)
        if self.rollout_ep_size is None:
            # the 4-layer recipe serves with triton grouped GEMM, which wants the experts sharded
            self.rollout_ep_size = self.rollout_tp_size if self.is_4layer else 1

        if self.train_mode == "lora" and self.lora_rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.lora_rank}")
        if self.sglang_max_total_tokens is not None and self.sglang_max_total_tokens <= 0:
            raise ValueError("SGLang max total tokens must be positive")
        if self.distributed_timeout_minutes <= 0:
            raise ValueError("Distributed timeout must be positive")
        if self.num_gpus % self.rollout_tp_size != 0 or _NUM_ATTENTION_HEADS % self.rollout_tp_size != 0:
            raise ValueError(
                f"rollout_tp_size must divide {self.num_gpus} GPUs and {_NUM_ATTENTION_HEADS} attention heads, "
                f"got {self.rollout_tp_size}"
            )
        if self.rollout_tp_size % self.rollout_ep_size != 0 or _NUM_EXPERTS % self.rollout_ep_size != 0:
            raise ValueError(
                f"rollout_ep_size must divide rollout_tp_size and {_NUM_EXPERTS} experts, got {self.rollout_ep_size}"
            )

        if self.is_4layer:
            if self.pipeline_parallel_size != 1 or self.context_parallel_size != 1:
                raise NotImplementedError("Pipeline and context parallelism are only wired for the full model")
            return

        if self.num_gpus != _VALIDATED_FULL_MODEL_GPUS and self.tp_size_override is None:
            raise ValueError(
                f"The full-model layout is derived for {_VALIDATED_FULL_MODEL_GPUS} GPUs; pass --tp-size-override "
                f"(and --ep-size-override) for {self.num_gpus} GPUs"
            )
        if self.pipeline_parallel_size > 1:
            first, last = self.pipeline_layer_split
            if not 1 <= last <= first:
                raise ValueError(
                    f"Cannot split {self.num_layers} layers over pipeline_parallel_size={self.pipeline_parallel_size}: "
                    f"first={first}, last={last}"
                )
        model_parallel = self.tensor_parallel_size * self.context_parallel_size * self.pipeline_parallel_size
        if self.num_gpus % model_parallel != 0:
            raise ValueError(
                f"TP{self.tensor_parallel_size}*CP{self.context_parallel_size}*PP{self.pipeline_parallel_size}"
                f"={model_parallel} must divide the {self.num_gpus} training GPUs"
            )
        non_pp_ranks = self.tensor_parallel_size * self.context_parallel_size * (self.num_gpus // model_parallel)
        if non_pp_ranks % self.expert_parallel_size != 0:
            raise ValueError(
                f"expert_parallel_size={self.expert_parallel_size} must divide the DP*CP*TP ranks "
                f"of one pipeline stage ({non_pp_ranks})"
            )

    @property
    def is_4layer(self) -> bool:
        return self.model_name == "Kimi-K3-4layer"

    @property
    def num_layers(self) -> int:
        return _NUM_LAYERS[self.model_name]

    @property
    def num_gpus(self) -> int:
        return self.num_nodes * self.num_gpus_per_node

    @property
    def bf16_name(self) -> str:
        return f"{self.model_name}-bf16"

    @property
    def megatron_model_type(self) -> str:
        return _MEGATRON_MODEL_TYPE[self.model_name]

    @property
    def tensor_parallel_size(self) -> int:
        if self.tp_size_override is not None:
            return self.tp_size_override
        if self.is_4layer:
            return min(8, self.num_gpus)
        if self.pipeline_parallel_size == 1:
            return 32
        return self.num_gpus // (self.pipeline_parallel_size * self.context_parallel_size)

    @property
    def expert_parallel_size(self) -> int:
        if self.ep_size_override is not None:
            return self.ep_size_override
        if self.is_4layer:
            return self.tensor_parallel_size
        # EP is bounded by the DP*CP*TP ranks inside one pipeline stage.
        model_parallel = self.tensor_parallel_size * self.context_parallel_size * self.pipeline_parallel_size
        data_parallel = self.num_gpus // model_parallel
        return self.tensor_parallel_size * self.context_parallel_size * data_parallel

    @property
    def pipeline_layer_split(self) -> tuple[int, int]:
        """(first, last) stage layer counts; middle stages each take `first` layers."""
        pp = self.pipeline_parallel_size
        first = -(-self.num_layers // pp)
        last = self.num_layers - first * (pp - 1)
        return first, last


def _download_dataset(args: ScriptArgs) -> None:
    if args.task == "gsm8k":
        U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)
    else:
        U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
        U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


@app.command()
@U.dataclass_cli
def prepare_data(args: ScriptArgs) -> None:
    _download_dataset(args)


def _prepare_download(args: ScriptArgs) -> None:
    """Native MXFP4 checkpoint + task dataset. Idempotent: hf skips existing blobs."""
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    if args.hf_checkpoint == f"{args.model_dir}/{args.model_name}":
        U.exec_command_cpu(f"hf download {args.model_org}/{args.model_name} --local-dir {args.hf_checkpoint}")
    _download_dataset(args)


@app.command()
@U.dataclass_cli
def prepare_download(args: ScriptArgs) -> None:
    _prepare_download(args)


def _prepare_bf16(args: ScriptArgs) -> None:
    """Dequantize the MXFP4 experts; Megatron loads BF16. One node, GPU."""
    U.exec_command_gpu(
        f"python {U.repo_base_dir}/tools/convert_mxfp4_to_bf16.py "
        f"--model-dir {args.hf_checkpoint} --save-dir {args.bf16_checkpoint} --device cuda"
    )


@app.command()
@U.dataclass_cli
def prepare_bf16(args: ScriptArgs) -> None:
    _prepare_bf16(args)


def _prepare_torch_dist(args: ScriptArgs) -> None:
    """BF16 HF -> torch_dist in the training layout. The output re-shards at load."""
    if not args.is_4layer:
        raise NotImplementedError(
            "The full model converts on 32 ranks; run tools/convert_hf_to_torch_dist.py as documented in "
            "docs/models/kimi/kimi-k3.md"
        )
    U.convert_checkpoint(
        model_name=args.bf16_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        extra_args=(
            f"--bf16 --tensor-model-parallel-size {args.tensor_parallel_size} --sequence-parallel "
            "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
            f"--expert-model-parallel-size {args.expert_parallel_size} --expert-tensor-parallel-size 1 "
            "--megatron-to-hf-mode raw "
        ),
        dir_dst=args.model_dir,
        hf_checkpoint=args.bf16_checkpoint,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare_torch_dist(args: ScriptArgs) -> None:
    _prepare_torch_dist(args)


def _train(args: ScriptArgs) -> None:
    is_debug = args.mode == "debug_minimal"
    is_lora = args.train_mode == "lora"
    if args.task == "gsm8k":
        dataset = Path(args.data_dir) / "gsm8k" / "train.parquet"
        input_key = "messages"
    else:
        dataset = Path(args.data_dir) / "dapo-math-17k" / "dapo-math-17k.jsonl"
        input_key = "prompt"

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        "--megatron-to-hf-mode raw "
        "--model-name kimi_k3 "
    )

    lora_args = ""
    if is_lora:
        lora_args = (
            f"--lora-rank {args.lora_rank} "
            f"--lora-alpha {args.lora_alpha} "
            f"--lora-dropout {args.lora_dropout} "
            f'--target-modules "{args.target_modules}" '
            "--no-gradient-accumulation-fusion "
            # host mirror of the rollout base, so releasing it does not re-ship the base every step
            "--lora-base-cpu-backup "
        )
        if args.experts_shared_outer_loras:
            lora_args += "--experts-shared-outer-loras "
        if args.check_lora_weight_equal:
            lora_args += "--check-lora-weight-equal "

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
    if args.eval_interval is not None:
        # eval on the training task's held-out set, keyed by the eval dataset's own column
        if args.task == "gsm8k":
            eval_name, eval_rel, eval_input_key = "gsm8k", "gsm8k/test.parquet", "messages"
        else:
            eval_name, eval_rel, eval_input_key = "aime", "aime-2024/aime-2024.jsonl", "prompt"
        rollout_args += (
            f"--eval-interval {args.eval_interval} "
            f"--eval-prompt-data {eval_name} {Path(args.data_dir) / eval_rel} "
            f"--eval-input-key {eval_input_key} "
            "--n-samples-per-eval-prompt 1 "
            "--eval-temperature 0 "
            f"--eval-max-response-len {rollout_max_response_len} "
        )
    if args.save_debug_rollout_data is not None:
        rollout_args += f"--save-debug-rollout-data {args.save_debug_rollout_data} "

    pp_split_args = ""
    if args.pipeline_parallel_size > 1:
        first, last = args.pipeline_layer_split
        if first != last:
            pp_split_args = f"--decoder-first-pipeline-num-layers {first} --decoder-last-pipeline-num-layers {last} "
    max_tokens_per_gpu = (
        args.max_tokens_per_gpu if args.max_tokens_per_gpu is not None else (512 if args.is_4layer else 8192)
    )
    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_parallel_size} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {args.pipeline_parallel_size} "
        f"{pp_split_args}"
        f"--context-parallel-size {args.context_parallel_size} "
        f"--expert-model-parallel-size {args.expert_parallel_size} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {max_tokens_per_gpu} "
        "--log-probs-chunk-size 512 "
        f"--distributed-timeout-minutes {args.distributed_timeout_minutes} "
    )

    optimizer_args = (
        "--optimizer sgd --sgd-momentum 0 --lr 1e-6 --lr-decay-style constant "
        "--weight-decay 0 --use-distributed-optimizer "
        if is_debug
        else (
            f"--optimizer adam --lr {args.lr} --lr-decay-style constant "
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
        update_weight_buffer_size = 2 * 1024**3 if args.is_4layer or not is_lora else 256 * 1024**2
    # the radix extra-buffer strategy needs five KDA cache slots per running request
    sglang_request_capacity = 16 if args.is_4layer else args.rollout_max_concurrency
    sglang_mamba_capacity = 16 if args.is_4layer else 5 * args.rollout_max_concurrency
    graph_bs = " ".join(str(b) for b in (1, 2, 4, 8, 16, 32) if b <= max(1, args.rollout_max_concurrency))

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_tp_size} "
        f"--sglang-tp-size {args.rollout_tp_size} "
        f"--sglang-ep-size {args.rollout_ep_size} "
        "--sglang-server-concurrency 16 "
        f"--sglang-max-running-requests {sglang_request_capacity} "
        f"--sglang-max-mamba-cache-size {sglang_mamba_capacity} "
        "--use-miles-router "
    )
    if is_lora:
        sglang_args += (
            "--sglang-lora-backend triton " "--sglang-lora-strict-loading " f"--sglang-max-lora-rank {args.lora_rank} "
        )
        if args.rollout_tp_size > 8:
            # above TP8 the Marlin MoE intermediate is tile-padded, which the virtual-experts LoRA kernel rejects
            sglang_args += "--no-sglang-lora-use-virtual-experts "
        if not args.is_4layer:
            # the adapter is re-streamed every step; a host copy per TP rank (~45 GB) is never read
            sglang_args += "--sglang-lora-no-cpu-backup "
    if args.is_4layer:
        sglang_args += (
            "--sglang-cuda-graph-bs 1 2 4 8 16 "
            "--sglang-mem-fraction-static 0.7 "
            "--sglang-moe-runner-backend triton "
            "--sglang-disable-shared-experts-fusion "
        )
    else:
        sglang_args += (
            "--sglang-moe-runner-backend marlin "
            "--sglang-decode-attention-backend trtllm_mla "
            "--sglang-mamba-radix-cache-strategy extra_buffer "
            f"--sglang-cuda-graph-bs-decode {graph_bs} "
            "--sglang-cuda-graph-backend-prefill disabled "
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
        f"--update-weight-buffer-size {update_weight_buffer_size} "
        f"--train-memory-margin-bytes {(2 if is_debug else 4) * 1024**3} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )
    if args.is_4layer:
        misc_args += "--no-check-for-nan-in-loss-and-grad "
    if args.check_weight_update_equal:
        misc_args += "--check-weight-update-equal --check-weight-update-skip-list vision_tower. mm_projector. "
    if not args.skip_saving:
        misc_args += f"--save {args.save_dir}/{args.run_id} --save-interval 50 "

    if args.enable_wandb:
        wandb_args = (
            "--use-wandb "
            "--wandb-project miles-run_kimi_k3 "
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
        f"{args.extra_args} "
    )
    extra_env_vars = {
        "NCCL_TIMEOUT": "3600",
        "PYTHONPATH": os.pathsep.join((str(Path(__file__).resolve().parents[1]), args.sglang_path)),
        "SGLANG_JIT_ROUTE_RADIX": "1",
        # sglang's membind pins the whole TMS host backup to one NUMA node, which cannot hold it
        "SGLANG_NUMA_BIND_V2": os.environ.get("SGLANG_NUMA_BIND_V2", "0"),
    }
    if is_lora:
        # the LoRA wrapper bypasses the o_proj.forward patch the K3 all-reduce fusion relies on
        extra_env_vars["SGLANG_K3_AR_FUSION"] = "0"
    # Ray's runtime_env replaces the actor env; without the JIT cache dirs the KDA kernels recompile every run
    for cache_var in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        cache_dir = os.environ.get(cache_var)
        if cache_dir:
            extra_env_vars[cache_var] = cache_dir

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
    _train(args)


if __name__ == "__main__":
    app()
