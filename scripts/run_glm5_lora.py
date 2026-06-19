"""
GLM-5.1 GRPO LoRA training script (Megatron-Bridge / bridge mode).

GLM-5.1 is MoE + MLA + DSA (DeepSeek Sparse Attention). LoRA trains through the
Megatron-Bridge path (``--megatron-to-hf-mode bridge``); the GLM-5.1 "dsa"
experimental-attention-variant spec is registered for that path inside miles
(``bridge_lora_helpers.py`` monkey-patch), so NO ``--spec`` is consumed here
(the registry ``.sh`` may still list it; it is provably inert under bridge LoRA).

Modeled on ``scripts/run_deepseek_v4.py`` (typer app + ScriptArgs(ExecuteTrainConfig)
+ model-name -> megatron_model_type registry + ``_get_parallel_config`` + ``U.execute_train``).

Two GLM-5.1/DSA specifics:
  * ``--target-modules`` excludes the 3 DSA indexer modules (wq_b/wk/weights_proj) by default —
    the indexer stays a code capability (miles commits), this run does not train it.
  * ``--qkv-format bshd`` + ``--micro-batch-size 1`` (no ``--use-dynamic-batch-size``):
    megatron-core's DSA core-attention needs a 4D (bshd) query; the default ``thd``
    packing yields a 3D query and raises "not enough values to unpack".

Supported model variants (HF checkpoint must be the native GLM-5.1 config,
model_type=glm_moe_dsa / GlmMoeDsaForCausalLM):
  GLM-5.1            full model
  GLM-5.1-6layer     6-layer prune (jybsuper/GLM-5.1-6layer) — single-node smoke test
  GLM-5.1-4layer / GLM-5.1-20layer  other prunes

Usage (run ON the devbox, miles editable-installed under /personal):
  python scripts/run_glm5_lora.py prepare --task gsm8k          # download gsm8k
  python scripts/run_glm5_lora.py train   --model-name GLM-5.1-6layer --num-gpus-per-node 4
  python scripts/run_glm5_lora.py train   --no-enable-wandb --num-rollout 3
"""

from dataclasses import dataclass, field
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

# 6-layer toy default checkpoint (HF cache snapshot).
_DEFAULT_6LAYER_CKPT = (
    "/cluster-storage/models/models--jybsuper--GLM-5.1-6layer/"
    "snapshots/1ea546e4990647bc651e94953a57dfaa9eedb576"
)

_MEGATRON_MODEL_TYPE = {
    "GLM-5.1": "glm5-744B-A40B",
    "GLM-5.1-6layer": "glm5-744B-A40B_6layer",
    "GLM-5.1-4layer": "glm5-744B-A40B_4layer",
    "GLM-5.1-20layer": "glm5-744B-A40B_20layer",
}

# Explicit LoRA targets: standard attn + MLA + MLP/MoE, EXCLUDING the DSA indexer
# (wq_b/wk/weights_proj). Set --target-modules all-linear to also cover the indexer.
_DEFAULT_TARGET_MODULES = (
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,"
    "q_a_proj,kv_a_proj_with_mqa,q_b_proj,kv_b_proj"
)


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: Literal[
        "GLM-5.1",
        "GLM-5.1-6layer",
        "GLM-5.1-4layer",
        "GLM-5.1-20layer",
    ] = "GLM-5.1-6layer"
    task: Literal["gsm8k"] = "gsm8k"

    hf_checkpoint: str | None = None
    save_dir: str = "/personal/checkpoints"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    # performance
    num_gpus_per_node: int = 4

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = _DEFAULT_TARGET_MODULES

    # rollout
    num_rollout: int = 1
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 4
    rollout_max_response_len: int = 256
    global_batch_size: int = 16

    # rollout engine
    rollout_num_gpus_per_engine: int = 2  # rollout tp=2
    sglang_mem_fraction_static: float = 0.5

    enable_wandb: bool = True
    # pass any extra miles/megatron/sglang args through, e.g. --extra-args '--lora-base-cpu-backup'
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = (
                _DEFAULT_6LAYER_CKPT
                if self.model_name == "GLM-5.1-6layer"
                else f"/root/models/{self.model_name}"
            )

    @property
    def megatron_model_type(self) -> str:
        return _MEGATRON_MODEL_TYPE[self.model_name]


def _get_parallel_config(args: ScriptArgs) -> str:
    """Single-node MoE layout: TP = EP = num_gpus_per_node, DP1 (mirrors run_glm5_744b_a40b).

    bshd (4D query) is REQUIRED for DSA core-attention and forbids --use-dynamic-batch-size,
    hence --micro-batch-size 1.
    """
    ngpu = args.num_gpus_per_node
    return (
        f"--tensor-model-parallel-size {ngpu} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {ngpu} "
        "--expert-tensor-parallel-size 1 "
        "--qkv-format bshd "
        "--micro-batch-size 1 "
    )


def _download_dataset(args: ScriptArgs):
    if args.task == "gsm8k":
        U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)


def _prepare_download(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.data_dir}")
    _download_dataset(args)


def _train(args: ScriptArgs):
    print(
        f"[run] GLM-5.1 LoRA: model={args.model_name} "
        f"(megatron_model_type={args.megatron_model_type}), "
        f"{args.num_gpus_per_node} GPUs, rollout tp={args.rollout_num_gpus_per_engine}"
    )
    load_save_path = f"{args.save_dir}/{args.run_id}"

    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} " "--megatron-to-hf-mode bridge "

    lora_args = (
        f"--lora-rank {args.lora_rank} "
        f"--lora-alpha {args.lora_alpha} "
        f"--lora-dropout {args.lora_dropout} "
        f'--target-modules "{args.target_modules}" '
    )

    # gsm8k + math reward
    rollout_args = (
        f"--prompt-data {args.data_dir}/gsm8k/train.parquet "
        "--input-key messages "
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

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-5 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    perf_args = _get_parallel_config(args)

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-cuda-graph-max-bs 64 "
        "--sglang-moe-runner-backend triton "
        "--sglang-disable-shared-experts-fusion "
        "--sglang-reasoning-parser glm45 "
        "--sglang-tool-call-parser glm47 "
    )

    save_args = f"--save-interval 1 --save {load_save_path} "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--calculate-per-token-loss "
        "--use-miles-router "
        f"--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )

    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""

    train_args = (
        f"{ckpt_args} "
        f"{lora_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{wandb_args} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{save_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"},
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download the task dataset (gsm8k). Run once per node before training."""
    _prepare_download(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run GRPO LoRA training (assumes the dataset is already prepared)."""
    _train(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """Download the dataset, then run GRPO LoRA training."""
    _prepare_download(args)
    _train(args)


if __name__ == "__main__":
    app()
