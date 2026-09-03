"""
GLM-5.3-Flash GRPO LoRA training (Megatron-Bridge / bridge mode).

GLM-5.3-Flash (``glm5_next``) is a 45-layer KDA + DSA hybrid MoE with mHC hyper-connections
(see ``scripts/run_glm5_3_flash.py`` and ``docs/models/glm/glm5-3-flash.md`` for the full-FT
recipe). LoRA trains through the bridge path (``--megatron-to-hf-mode bridge``): Megatron-Bridge
builds the model from the HF checkpoint (``Glm5NextBridge``), wraps the adapters, and the same
adapter is served live by SGLang.

Adapter targets (Megatron names, wildcards anchored below ``decoder.layers``):
  * KDA linear attention (34 layers): linear_q / linear_k / linear_v / linear_proj and the gate
    projections linear_b / linear_f_a / linear_f_b / linear_g_a / linear_g_b
    (HF q/k/v/o_proj, b/f_a/f_b/g_a/g_b_proj);
  * DSA sparse MLA (11 layers): linear_q_down_proj / linear_q_up_proj / linear_kv_down_proj /
    linear_kv_up_proj / linear_proj (HF q_a/q_b/kv_a/kv_b/o_proj). The kpool indexer
    (wq_b / wk / weights_proj) gets no gradient on the fused TileLang path, so it is excluded;
  * MLP: dense linear_fc1/linear_fc2, the shared expert and the 288 routed experts (grouped GEMM).
    Drop the ``mlp.experts.*`` entries for attention + shared-expert-only LoRA.

The public checkpoint is FP8 (block-128). The trainer dequantizes it to bf16 on load; SGLang
serves the FP8 checkpoint directly, so no bf16 copy is needed and the base weights are never
re-synced (``--lora-base-cpu-backup`` keeps SGLang's base resident across colocated
offload / onload). Only the adapter is shipped to the engines each step.

Usage (single node, 4-layer slice):
  python scripts/run_glm5_3_flash_lora.py prepare --model-name GLM-5.3-Flash-4layer
  python scripts/run_glm5_3_flash_lora.py train --model-name GLM-5.3-Flash-4layer \\
      --num-nodes 1 --num-gpus-per-node 8 --task gsm8k

Full model on 3 nodes x 8 GPUs (TP8 / EP24 / PP1, one 8-GPU SGLang engine per node):
  export MILES_SCRIPT_EXTERNAL_RAY=1   # after `ray start` on every node
  python scripts/run_glm5_3_flash_lora.py train --model-name GLM-5.3-Flash \\
      --num-nodes 3 --num-gpus-per-node 8 --task gsm8k --rollout-max-response-len 8192
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U
from scripts.run_glm5_3_flash import _MODEL_REGISTRY

app = typer.Typer()

_HF_REPO = {
    "GLM-5.3-Flash": "zai-org/GLM-5.3-Flash",
    "GLM-5.3-Flash-4layer": "CharyZeng/GLM-5.3-Flash-4layer",
}

_LAYERS = "decoder.layers.*"
_ATTENTION_TARGET_MODULES = [
    # KDA linear attention
    f"{_LAYERS}.self_attention.linear_q",
    f"{_LAYERS}.self_attention.linear_k",
    f"{_LAYERS}.self_attention.linear_v",
    f"{_LAYERS}.self_attention.linear_b",
    f"{_LAYERS}.self_attention.linear_f_a",
    f"{_LAYERS}.self_attention.linear_f_b",
    f"{_LAYERS}.self_attention.linear_g_a",
    f"{_LAYERS}.self_attention.linear_g_b",
    # DSA sparse MLA
    f"{_LAYERS}.self_attention.linear_q_down_proj",
    f"{_LAYERS}.self_attention.linear_q_up_proj",
    f"{_LAYERS}.self_attention.linear_kv_down_proj",
    f"{_LAYERS}.self_attention.linear_kv_up_proj",
    # output projection of both layer types
    f"{_LAYERS}.self_attention.linear_proj",
]
_MLP_TARGET_MODULES = [
    f"{_LAYERS}.mlp.linear_fc1",
    f"{_LAYERS}.mlp.linear_fc2",
    f"{_LAYERS}.mlp.shared_experts.linear_fc1",
    f"{_LAYERS}.mlp.shared_experts.linear_fc2",
]
_EXPERT_TARGET_MODULES = [
    f"{_LAYERS}.mlp.experts.linear_fc1",
    f"{_LAYERS}.mlp.experts.linear_fc2",
]
_DEFAULT_TARGET_MODULES = ",".join(_ATTENTION_TARGET_MODULES + _MLP_TARGET_MODULES + _EXPERT_TARGET_MODULES)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: Literal["GLM-5.3-Flash", "GLM-5.3-Flash-8layer", "GLM-5.3-Flash-4layer"] = "GLM-5.3-Flash-4layer"
    task: Literal["gsm8k", "dapo-math"] = "gsm8k"

    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    save_dir: str = "/root/shared_data"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    # parallelism: TP over the GPUs of one node, experts spread over every GPU, no PP
    num_nodes: int = 1
    num_gpus_per_node: int = 8
    tensor_model_parallel_size: int = 0  # 0 => min(8, num_gpus_per_node)

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = _DEFAULT_TARGET_MODULES
    # colocate: keep SGLang's FP8 base resident across offload/onload so it is never re-synced
    lora_base_cpu_backup: bool = True
    # grouped-expert adapter layout; must match SGLang (default per-expert)
    experts_shared_outer_loras: bool = False
    check_lora_weight_equal: bool = True

    # rollout
    num_rollout: int = 10
    rollout_batch_size: int = 16
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    rollout_max_response_len: int = 0  # 0 => per-task default (gsm8k 8192, dapo-math 4096)
    rollout_temperature: float = 1.0
    use_r3: bool = True

    # rollout engine (colocated): one engine per node
    rollout_num_gpus_per_engine: int = 0  # 0 => num_gpus_per_node
    sglang_mem_fraction_static: float = 0.5
    sglang_lora_backend: str = "triton"

    # optimizer
    lr: float = 1e-5

    enable_wandb: bool = False
    skip_saving: bool = True
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"
        if self.rollout_max_response_len == 0:
            self.rollout_max_response_len = 4096 if self.task == "dapo-math" else 8192
        if self.tensor_model_parallel_size == 0:
            self.tensor_model_parallel_size = min(8, self.num_gpus_per_node)
        if self.rollout_num_gpus_per_engine == 0:
            self.rollout_num_gpus_per_engine = self.num_gpus_per_node

    @property
    def megatron_model_type(self) -> str:
        return _MODEL_REGISTRY[self.model_name]


def _download_dataset(args: ScriptArgs):
    match args.task:
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)
        case "dapo-math":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def _prepare_download(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.data_dir} {args.model_dir}")
    repo = _HF_REPO.get(args.model_name)
    if repo is not None and not os.path.isdir(args.hf_checkpoint):
        U.exec_command_cpu(f"hf download {repo} --local-dir {args.hf_checkpoint}")
    _download_dataset(args)


def _parallel_args(args: ScriptArgs) -> str:
    world_size = args.num_nodes * args.num_gpus_per_node
    tp = args.tensor_model_parallel_size
    assert world_size % tp == 0, (world_size, tp)
    # 288 routed experts: spread them over every GPU (EP = world size) with ETP 1; no PP so the
    # adapter sync and the colocated engines stay simple.
    ep = world_size
    assert 288 % ep == 0, f"expert parallel size {ep} must divide 288 experts"
    return (
        f"--tensor-model-parallel-size {tp} --sequence-parallel --pipeline-model-parallel-size 1 "
        f"--context-parallel-size 1 --expert-model-parallel-size {ep} --expert-tensor-parallel-size 1 "
        "--qkv-format thd --micro-batch-size 1 --max-tokens-per-gpu 16384 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
    )


def _rollout_args(args: ScriptArgs) -> str:
    rollout_args = (
        "--label-key label --apply-chat-template --rollout-shuffle --rm-type math "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} --global-batch-size {args.global_batch_size} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--rollout-temperature {args.rollout_temperature} --num-steps-per-rollout 1 --balance-data "
    )
    match args.task:
        case "gsm8k":  # zhuzilin/gsm8k ships {messages, label} parquet
            rollout_args += f"--prompt-data {args.data_dir}/gsm8k/train.parquet --input-key messages "
        case "dapo-math":  # zhuzilin/dapo-math-17k ships {prompt, label} jsonl
            rollout_args += f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl --input-key prompt "
    return rollout_args


def _sglang_args(args: ScriptArgs) -> str:
    engine = args.rollout_num_gpus_per_engine
    return (
        f"--rollout-num-gpus-per-engine {engine} --sglang-tp-size {engine} --sglang-ep-size {engine} "
        f"--sglang-dp-size 1 --sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-chunked-prefill-size 8192 --sglang-disable-radix-cache "
        "--sglang-dsa-prefill-backend tilelang --sglang-dsa-decode-backend tilelang "
        "--sglang-kv-cache-dtype bfloat16 "
        # LoRA serving: triton LoRA + triton MoE runner (virtual experts), no shared-expert fusion
        f"--sglang-lora-backend {args.sglang_lora_backend} --sglang-max-lora-rank {args.lora_rank} "
        "--sglang-moe-runner-backend triton --sglang-disable-shared-experts-fusion "
        "--router-health-success-threshold 1 --router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "
    )


def _lora_args(args: ScriptArgs) -> str:
    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout {args.lora_dropout} "
        f'--target-modules "{args.target_modules}" --megatron-to-hf-mode bridge '
        "--dsa-attention-backend tilelang --no-gradient-accumulation-fusion "
    )
    if args.experts_shared_outer_loras:
        lora_args += "--experts-shared-outer-loras "
    if args.lora_base_cpu_backup:
        lora_args += "--lora-base-cpu-backup "
    if args.check_lora_weight_equal:
        lora_args += "--check-lora-weight-equal "
    return lora_args


def _train(args: ScriptArgs):
    print(
        f"[run] GLM-5.3-Flash LoRA: model={args.model_name} task={args.task} "
        f"{args.num_nodes}x{args.num_gpus_per_node} GPUs tp={args.tensor_model_parallel_size} "
        f"engine={args.rollout_num_gpus_per_engine} r3={args.use_r3}"
    )
    load_save_path = f"{args.save_dir}/{args.run_id}"

    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} "
    save_args = "" if args.skip_saving else f"--save-interval 1 --save {load_save_path} "

    grpo_args = (
        "--advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl --kl-coef 0.00 "
        "--entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
    )
    r3_args = "--use-rollout-routing-replay " if args.use_r3 else ""
    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
    )
    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --calculate-per-token-loss "
        f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} --colocate --offload-train-target cpu "
        f"--update-weight-buffer-size {1 * 1024**3} --model-name glm5_next "
        "--rollout-health-check-interval 300 --rollout-health-check-timeout 300 "
        "--distributed-timeout-minutes 60 "
    )
    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""

    train_args = (
        f"{ckpt_args} {_lora_args(args)} {_rollout_args(args)} {optimizer_args} {grpo_args} {r3_args} "
        f"{wandb_args} {_parallel_args(args)} {_sglang_args(args)} {save_args} {misc_args} {args.extra_args} "
    )

    extra_env_vars = {
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "PYTHONFAULTHANDLER": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TRITON_CACHE_DIR": "/tmp/triton_cache",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor_cache",
    }
    # FLA_CACHE_RESULTS=0 sidesteps a Triton 3.6 autotune-cache hashing failure on fla's KDA
    # kernels (`Unsupported function referenced: next_power_of_2`); newer Triton does not need it.
    for passthrough in ("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", "FLA_CACHE_RESULTS"):
        if os.environ.get(passthrough):
            extra_env_vars[passthrough] = os.environ[passthrough]

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
def prepare(args: ScriptArgs):
    """Download the HF checkpoint (known repos) and the task dataset."""
    _prepare_download(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run GRPO LoRA training (dataset already prepared)."""
    _train(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """prepare + train."""
    _prepare_download(args)
    _train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
