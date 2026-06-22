"""
GLM-5 / 5.1 / 5.2 GRPO LoRA training script (Megatron-Bridge / bridge mode).

These models are MoE + MLA + DSA (DeepSeek Sparse Attention). LoRA trains through the
Megatron-Bridge path (``--megatron-to-hf-mode bridge``); the "dsa" experimental-attention
spec is provided by the Megatron-Bridge GLM5 provider, so NO ``--spec`` is consumed here
(the registry ``.sh`` may still list it; it is provably inert under bridge LoRA).

GLM-5.2 additionally has DSA *cross-layer index sharing* (only "computing" layers carry the
indexer; "skip" layers reuse the most recent computing layer's top-k). The Megatron-Bridge
GLM5 provider reads that schedule (``index_topk_freq``) from the HF config and builds
``CrossLayerDSAttention`` -- nothing extra is needed here beyond selecting a GLM-5.2 model
name (which maps to the ``glm5.2-744B-A40B*`` registry, identical to GLM-5.1 except
``--rotary-base`` 8e6).

Modeled on ``scripts/run_deepseek_v4.py`` (typer app + ScriptArgs(ExecuteTrainConfig)
+ model-name -> megatron_model_type registry + ``_get_parallel_config`` + ``U.execute_train``).

DSA kernel backend (``--dsa-attention-backend``; bridge path only). Orthogonal to model version
and to LoRA -- BOTH backends run GLM-5.1 *and* GLM-5.2, full or LoRA:
  * ``megatron-bridge`` (default): the portable unfused megatron-core DSA kernels (DSAttention /
    CrossLayerDSAttention). No extra deps. Uses the ``bshd`` query layout.
  * ``slime``: the vendored fused TileLang kernels (SparseMLA + lighting_indexer), matching slime's
    rollout kernels for rollout<->train numerical parity (incl. R3 indexer replay). Needs the
    optional ``tilelang`` dep and the ``thd`` (packed) layout. Training/forward-only (no KV cache):
    it cannot serve generation -- the rollout is always served by sglang.
  This launcher selects the matching ``--qkv-format`` automatically from the backend (see
  ``_get_parallel_config``); just pass ``--dsa-attention-backend slime`` or leave the default. See
  the Megatron-Bridge ``models/glm_moe_dsa/__init__.py`` docstring for the full backend matrix.

Two DSA specifics:
  * ``--target-modules`` excludes the 3 DSA indexer modules (wq_b/wk/weights_proj) by default --
    the indexer stays a frozen base capability; this run does not train it. On the slime backend the
    indexer adapter gets no gradient anyway (a genuine no-op); on the default backend it would get a
    tiny aux-loss gradient (~1e-5), so excluding it there is a deliberate choice.
  * ``--micro-batch-size 1`` (no ``--use-dynamic-batch-size``): both backends pin a static
    micro-batch. The query layout follows the backend -- ``bshd`` (megatron-core's DSA
    core-attention needs a 4D query; the default ``thd`` packing yields a 3D query and raises "not
    enough values to unpack") for ``megatron-bridge``, ``thd`` (packed) for ``slime``.

Supported model variants (HF checkpoint must be the native config,
model_type=glm_moe_dsa / GlmMoeDsaForCausalLM):
  GLM-5.1 / GLM-5.2            full models
  GLM-5.1-6layer              6-layer GLM-5.1 prune (jybsuper/GLM-5.1-6layer)
  GLM-5.2-7layer              7-layer GLM-5.2 prune (jybsuper/GLM-5.2-7layer)
  GLM-5.1-4layer / -20layer   other GLM-5.1 prunes

Usage (run ON the devbox; miles editable-installed under /personal):
  python scripts/run_glm5_lora.py prepare    --model-name GLM-5.1-6layer   # download model + gsm8k
  # default (megatron-bridge / unfused) backend:
  python scripts/run_glm5_lora.py full-train --model-name GLM-5.1-6layer --num-gpus-per-node 4
  # fused slime backend (GLM-5.2 shown; works for GLM-5.1 too):
  python scripts/run_glm5_lora.py full-train --model-name GLM-5.2-7layer \\
      --dsa-attention-backend slime --num-gpus-per-node 4

GLM-5.2 rollout caveat: sglang does not yet serve the GLM-5.2 cross-layer (subset-indexer)
checkpoint, so a full rollout->train loop is blocked on the rollout side. The *training* side
is validated train-only by replaying a dumped rollout (both toys share the GLM tokenizer/vocab):
  # 1) dump a rollout from GLM-5.1 (sglang serves 5.1 fine)
  python scripts/run_glm5_lora.py full-train --model-name GLM-5.1-6layer \\
      --extra-args "--dump-details /personal/dump51"
  # 2) train GLM-5.2 on that dump (no sglang)
  python scripts/run_glm5_lora.py train --model-name GLM-5.2-7layer \\
      --extra-args "--load-debug-rollout-data /personal/dump51/rollout_data/0.pt"
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

# HF repos to download from (full models from zai-org; pruned toys from jybsuper).
# Variants absent here (e.g. -4layer/-20layer) are assumed already present at --hf-checkpoint.
_HF_REPO = {
    "GLM-5.1": "zai-org/GLM-5.1",
    "GLM-5.1-6layer": "jybsuper/GLM-5.1-6layer",
    "GLM-5.2": "zai-org/GLM-5.2",
    "GLM-5.2-7layer": "jybsuper/GLM-5.2-7layer",
}

_MEGATRON_MODEL_TYPE = {
    "GLM-5.1": "glm5-744B-A40B",
    "GLM-5.1-6layer": "glm5.1-744B-A40B_6layer",
    "GLM-5.1-4layer": "glm5-744B-A40B_4layer",
    "GLM-5.1-20layer": "glm5-744B-A40B_20layer",
    "GLM-5.2": "glm5.2-744B-A40B",
    "GLM-5.2-7layer": "glm5.2-744B-A40B_7layer",
}

# Explicit LoRA targets: standard attn + MLA + MLP/MoE, EXCLUDING the DSA indexer
# (wq_b/wk/weights_proj). Set --target-modules all-linear to also cover the indexer.
_DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,q_a_proj,kv_a_proj_with_mqa,q_b_proj,kv_b_proj"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: Literal[
        "GLM-5.1",
        "GLM-5.1-6layer",
        "GLM-5.1-4layer",
        "GLM-5.1-20layer",
        "GLM-5.2",
        "GLM-5.2-7layer",
    ] = "GLM-5.1-6layer"
    task: Literal["gsm8k"] = "gsm8k"

    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    save_dir: str = "/personal/checkpoints"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    # DSA sparse-MLA kernel backend (bridge path only). Orthogonal to model version & LoRA: BOTH
    # backends support GLM-5.1 AND GLM-5.2, full or LoRA.
    #   "megatron-bridge" (default): portable unfused megatron-core DSA kernels; bshd layout; no
    #                                extra deps.
    #   "slime":                     vendored fused TileLang SparseMLA + lighting_indexer; thd
    #                                layout; needs tilelang; matches slime's rollout kernels for
    #                                rollout<->train parity (incl. R3 indexer replay);
    #                                training/forward-only (no KV cache, cannot serve inference).
    # The matching --qkv-format is chosen from this automatically (see _get_parallel_config).
    dsa_attention_backend: Literal["megatron-bridge", "slime"] = "megatron-bridge"

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
            # NB: must be a LOCAL path -- miles sets args.load = hf_checkpoint and asserts it
            # is an existing directory (a HF repo id is not accepted here).
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"

    @property
    def megatron_model_type(self) -> str:
        return _MEGATRON_MODEL_TYPE[self.model_name]


def _get_parallel_config(args: ScriptArgs) -> str:
    """Single-node MoE layout: TP = EP = num_gpus_per_node, DP1 (mirrors run_glm5_744b_a40b).

    The DSA kernel backend dictates the query layout; both forbid --use-dynamic-batch-size, hence
    --micro-batch-size 1:
      * megatron-bridge (unfused megatron-core DSA core-attention): needs bshd (4D query); the
        default thd packing yields a 3D query and raises "not enough values to unpack".
      * slime (fused TileLang SparseMLA + lighting_indexer): needs thd (packed) -- the fused
        kernels index by cu_seqlens and use a [t, heads, dim] layout; bshd has no cu_seqlens.
    """
    ngpu = args.num_gpus_per_node
    qkv_format = "thd" if args.dsa_attention_backend == "slime" else "bshd"
    return (
        f"--tensor-model-parallel-size {ngpu} --sequence-parallel --pipeline-model-parallel-size 1 "
        f"--context-parallel-size 1 --expert-model-parallel-size {ngpu} --expert-tensor-parallel-size 1 "
        f"--qkv-format {qkv_format} --micro-batch-size 1 "
    )


def _download_dataset(args: ScriptArgs):
    if args.task == "gsm8k":
        U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)


def _prepare_download(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.data_dir} {args.model_dir}")
    repo = _HF_REPO.get(args.model_name)
    if repo is not None:
        U.exec_command(f"hf download {repo} --local-dir {args.model_dir}/{args.model_name}")
    _download_dataset(args)


def _train(args: ScriptArgs):
    print(f"[run] GLM-5 LoRA: model={args.model_name} (megatron_model_type={args.megatron_model_type}), dsa-backend={args.dsa_attention_backend}, {args.num_gpus_per_node} GPUs, rollout tp={args.rollout_num_gpus_per_engine}")
    load_save_path = f"{args.save_dir}/{args.run_id}"

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --megatron-to-hf-mode bridge "
        f"--dsa-attention-backend {args.dsa_attention_backend} "
    )

    lora_args = f'--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout {args.lora_dropout} --target-modules "{args.target_modules}" '

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

    grpo_args = "--advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl --kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "

    optimizer_args = "--optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "

    perf_args = _get_parallel_config(args)

    sglang_args = f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} --sglang-mem-fraction-static {args.sglang_mem_fraction_static} --sglang-cuda-graph-max-bs 64 --sglang-moe-runner-backend triton --sglang-disable-shared-experts-fusion --sglang-reasoning-parser glm45 --sglang-tool-call-parser glm47 "

    save_args = f"--save-interval 1 --save {load_save_path} "

    misc_args = f"--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash --calculate-per-token-loss --use-miles-router --actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus_per_node} --colocate "

    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""

    train_args = f"{ckpt_args} {lora_args} {rollout_args} {optimizer_args} {grpo_args} {wandb_args} {perf_args} {sglang_args} {save_args} {misc_args} {args.extra_args} "

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
    """Download the model checkpoint (for a known HF repo) and the task dataset (gsm8k). Run once per node before training."""
    _prepare_download(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run GRPO LoRA training (assumes the dataset is already prepared)."""
    _train(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """Download the model checkpoint + dataset, then run GRPO LoRA training."""
    _prepare_download(args)
    _train(args)


if __name__ == "__main__":
    app()
