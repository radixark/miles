"""Step5 (StepFun Step-5-Preview) GRPO on DAPO-Math or geo3k, colocated SGLang rollout and Megatron training.

The rollout serves the checkpoint as released (routed experts in FP8 except MoE layers 88-90, everything
else BF16); the trainer runs BF16 and syncs each weight back in its checkpoint dtype. Full-attention (DSA)
layers recompute the rollout's region selection with the frozen indexer and attend sparsely, so attention
caps no context length.
A micro-batch holds one prompt beside a full-length response, and text prompts that do not fit are
dropped. The hardware layout picks the recipe. On one 4-GPU node the 5-layer slice
(``pb09204048/Step-5-Preview-5layer``, cut by ``tools/slice_step5_checkpoint.py``) exercises the whole loop at TP1
from a torch_dist conversion.

On 2 x 8 B300 (--num-nodes 2 --num-gpus-per-node 8) every variant runs the full-parameter recipe for the full
model (600B total, 27B active), so the slice can rehearse it: every parameter but the frozen DSA indexer
trains, with 32K responses:

- The trainer runs TP4 + SP and EP16, 22 experts per GPU. Expert data parallelism is then 1, so the
  distributed optimizer cannot shard the expert Adam state, and for the full model it does not fit HBM
  beside the bf16 weights and gradients. It lives in pinned host memory instead, 6 bytes per param: bf16
  moments and the int16 low halves of the fp32 masters, whose high halves are the bf16 params
  themselves. Each step streams it through the GPU in 1 GiB chunks (Megatron's
  --chunked-optimizer-state-offload). Gradients accumulate and reduce in bf16.
- The trainer loads the HF FP8 checkpoint through the Step5 bridge at startup, dequantizing the
  experts on read: the full model's torch_dist copy (1.2 TB) does not fit the node disk, so
  ``prepare`` only fetches the dataset.
- One SGLang engine per node (TP8/EP8), colocated. AIME 2026 is evaluated before the first step, every
  10 steps and after the last, at StepFun's recommended sampling (temperature 1.0, top-p 0.95, top-k off).

On 4 x 8 H200 (--num-nodes 4 --num-gpus-per-node 8), the full model uses rank-16 LoRA on Q/K/V/O,
dense MLPs and shared experts. Routed experts and the attention gate stay frozen. The trainer uses TP4/EP32;
each node serves a TP8/EP8 engine with FA3. Responses default to 8192 tokens. CPU base backups let
subsequent steps synchronize only adapters. Use --no-eval-aime for smoke runs without the eval dataset.

geo3k prompts carry one image each: SGLang expands it into crop and overview tokens, and the trainer runs
the frozen vision tower (``miles_plugins/models/step5/vl_model.py``) over the pixels the rollout sends along.

Needs: the FP8 checkpoint (or slice) under ``--model-dir`` carrying the step5 ``config.json``, and
an SGLang build with Step5 support (sgl-project/sglang#44356). The 2 x 8 recipe also
needs ``aime-2026/aime-2026.jsonl`` (``prompt`` and ``label`` rows) under ``--data-dir``, and a ray
cluster over both nodes: pass --ray-hostfile to join the other node over ssh, or start the cluster
yourself and set MILES_SCRIPT_EXTERNAL_RAY=1.

Args:
    --model-name: Step5 (2 x 8 B300 or 4 x 8 H200) or Step5-5layer (original layers 0, 3, 4, 88, 91: one of each kind).
    --num-nodes / --num-gpus-per-node: (2, 8) selects full training, (4, 8) LoRA; other layouts run the slice at TP1.
    --task: dapo_math (text) or geo3k (one image per prompt, TP1 recipe only).
    --model-dir / --data-dir: checkpoint (and its torch_dist conversion, next to it) and dataset locations.
    --num-rollout: GRPO iterations.
    --rollout-max-response-len: 32768 on 2 x 8, 8192 on 4 x 8, and 1024 otherwise unless set.
    --eval-aime / --no-eval-aime: the 2 x 8 AIME 2026 evals.
    --ray-hostfile: MPI-style hostfile; its other hosts join the ray cluster over ssh.
    --reuse-indexer-on-recompute: cache block IDs to skip the frozen Indexer during activation recomputation
        (disabled by default). Trades memory for less computation.

Example:
    hf download pb09204048/Step-5-Preview-5layer --local-dir /scratch/step5_slices/Step5-5layer
    python scripts/run_step5.py prepare --model-name Step5-5layer --model-dir /scratch/step5_slices
    python scripts/run_step5.py train --model-name Step5-5layer --model-dir /scratch/step5_slices
    python scripts/run_step5.py train --model-name Step5-5layer --model-dir /scratch/step5_slices --task geo3k
    MASTER_ADDR=<head-ip> python scripts/run_step5.py train --model-name Step5-5layer --num-nodes 2 \
        --num-gpus-per-node 8 --model-dir /scratch/step5_slices --data-dir /scratch/datasets \
        --ray-hostfile /root/hostfile --num-rollout 2
    MASTER_ADDR=<head-ip> python scripts/run_step5.py train --model-name Step5 --num-nodes 2 \
        --num-gpus-per-node 8 --model-dir /scratch/step5_models --data-dir /scratch/datasets \
        --ray-hostfile /root/hostfile --num-rollout 20
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U
from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend

app = typer.Typer()

_MODEL_TYPES = {
    "Step5": "step5",
    "Step5-5layer": "step5-5layer",
}


@dataclass(frozen=True)
class _TaskRecipe:
    """A prompt dataset and how the rollout reads it."""

    dataset: str
    prompt_file: str
    prompt_args: str
    has_images: bool = False


_TASKS = {
    "dapo_math": _TaskRecipe(
        dataset="zhuzilin/dapo-math-17k",
        prompt_file="dapo-math-17k/dapo-math-17k.jsonl",
        prompt_args="--input-key prompt --label-key label --apply-chat-template ",
    ),
    "geo3k": _TaskRecipe(
        dataset="chenhegu/geo3k_imgurl",
        prompt_file="geo3k_imgurl/train.parquet",
        prompt_args=(
            "--input-key problem --label-key answer --apply-chat-template "
            """--multimodal-keys '{"image": "images"}' """
            "--custom-generate-function-path examples.geo3k_vlm.rollout.generate "
        ),
        has_images=True,
    ),
}

# The full model's layouts put one of the 4 KV groups on each TP rank and spread the 352 experts over every GPU, so
# expert data parallelism is 1.
_FULL_MODEL_TENSOR_PARALLEL = 4
# AIME 2026 every 10 steps, besides the evals miles runs before the first step and after the last
_FULL_MODEL_EVAL_INTERVAL = 10
# With the chat template DAPO prompts reach 1,461 tokens (p99 382) and AIME ones stay shorter.
_MAX_PROMPT_LEN = 2048
# One node's micro-batch while prompt plus response fit in it.
_ONE_NODE_MAX_TOKENS_PER_GPU = 16384


@dataclass(frozen=True)
class _FullModelLayout:
    """What the full model trains on one cluster shape, and how it fits."""

    cluster: str
    max_response_len: int
    lr: float
    # what the trainer updates and where its state lives
    trainer_args: str
    sglang_attention_backend: str
    sglang_mem_fraction_static: float


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    model_name: Literal["Step5", "Step5-5layer"] = "Step5-5layer"
    task: Literal["dapo_math", "geo3k"] = "dapo_math"
    num_gpus_per_node: int = 4
    run_id: str = U.create_run_id()
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    num_rollout: int = 3
    rollout_max_response_len: int | None = None
    eval_aime: bool = True
    reuse_indexer_on_recompute: bool = False
    ray_hostfile: str | None = None
    extra_args: str = ""

    def __post_init__(self):
        layout = self.full_model_layout
        if self.rollout_max_response_len is None:
            self.rollout_max_response_len = layout.max_response_len if layout is not None else 1024
        assert (
            self.model_name != "Step5" or layout is not None
        ), "the full model is laid out for 2 x 8 B300 and 4 x 8 H200"
        if layout is not None:
            self._check_full_model_recipe(layout)

    def _check_full_model_recipe(self, layout: _FullModelLayout) -> None:
        assert self.task == "dapo_math", f"the {layout.cluster} recipe trains the text decoder only"
        assert self.rollout_max_response_len <= layout.max_response_len, (
            f"a micro-batch holds {self.max_tokens_per_gpu} tokens, one prompt plus a "
            f"{layout.max_response_len}-token response"
        )

    @property
    def task_recipe(self) -> _TaskRecipe:
        return _TASKS[self.task]

    @property
    def full_model_layout(self) -> _FullModelLayout | None:
        """The full model's recipe for this cluster shape; any other shape runs the slice at TP1."""
        return _FULL_MODEL_LAYOUTS.get((self.num_nodes, self.num_gpus_per_node))

    @property
    def megatron_model_type(self) -> str:
        return _MODEL_TYPES[self.model_name]

    @property
    def hf_checkpoint(self) -> str:
        return f"{self.model_dir}/{self.model_name}"

    @property
    def torch_dist_checkpoint(self) -> str:
        return f"{self.model_dir}/{self.model_name}_torch_dist"

    @property
    def trainer_init_checkpoint(self) -> str:
        # With no --load the trainer starts from --ref-load, and an HF directory there is read through the model's
        # bridge: the full model's torch_dist copy would not fit the node disk, and a slice on its layouts rehearses that.
        return self.hf_checkpoint if self.full_model_layout is not None else self.torch_dist_checkpoint

    @property
    def tensor_parallel_size(self) -> int:
        return _FULL_MODEL_TENSOR_PARALLEL if self.full_model_layout is not None else 1

    @property
    def max_tokens_per_gpu(self) -> int:
        """A micro-batch holds one prompt beside a full-length response."""
        if self.full_model_layout is not None:
            return _MAX_PROMPT_LEN + self.full_model_layout.max_response_len
        return max(_ONE_NODE_MAX_TOKENS_PER_GPU, _MAX_PROMPT_LEN + self.rollout_max_response_len)

    @property
    def max_prompt_len(self) -> int:
        """The longest prompt a micro-batch holds beside a full-length response."""
        return self.max_tokens_per_gpu - self.rollout_max_response_len


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download the dataset; for the TP1 recipe, convert the FP8 checkpoint to torch_dist (experts dequantized)."""
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.data_dir}")
    U.hf_download_dataset(args.task_recipe.dataset, data_dir=args.data_dir)
    if args.full_model_layout is not None:
        # its trainer reads the HF checkpoint
        return
    # Shard the experts across every GPU.
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        multinode=args.num_nodes > 1,
        num_nodes=args.num_nodes,
        extra_args=f"--expert-model-parallel-size {args.num_nodes * args.num_gpus_per_node} ",
        dir_dst=args.model_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Submit the GRPO job."""
    layout = args.full_model_layout
    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.trainer_init_checkpoint} "

    task = args.task_recipe
    rollout_args = (
        f"--prompt-data {args.data_dir}/{task.prompt_file} "
        f"{task.prompt_args}"
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {4 if layout is None else 32} "
        "--n-samples-per-prompt 8 "
        "--rollout-temperature 1.0 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
    )
    if not task.has_images:
        # An image prompt counts one token per image here, not the tokens SGLang expands it into.
        rollout_args += f"--rollout-max-prompt-len {args.max_prompt_len} "
    if layout is not None:
        rollout_args += _full_model_sampling_args(args, layout)

    # Step5 attention allows TP 1, 2 or 4 (4 KV groups); experts split 352 ways by EP.
    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_parallel_size} "
        f"{'--sequence-parallel ' if args.tensor_parallel_size > 1 else ''}"
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {args.num_gpus_per_node * args.num_nodes} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        # The rollout runs FP8 experts against a BF16 trainer: replaying its expert choices removes
        # most of the mismatch (0.013 -> 0.005 mean |logprob diff| on an 8-layer slice), TIS the rest.
        "--use-rollout-routing-replay "
        "--use-tis "
    )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {1e-6 if layout is None else layout.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    if layout is not None:
        optimizer_args += layout.trainer_args

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.num_gpus_per_node} "
        f"--sglang-tp-size {args.num_gpus_per_node} "
        f"--sglang-ep-size {args.num_gpus_per_node} "
        f"--sglang-attention-backend {'fa4' if layout is None else layout.sglang_attention_backend} "
        "--sglang-page-size 64 "
        # routing replay needs materialized top-k ids, and triton is the runner validated so far
        "--sglang-moe-runner-backend triton "
        "--sglang-disable-prefill-cuda-graph "
        # fp32 logits: rounding them to bf16 adds ~2.5% to the train-rollout log-prob gap
        "--sglang-enable-fp32-lm-head "
        f"--sglang-cuda-graph-max-bs-decode {32 if layout is None else 128} "
        f"--sglang-mem-fraction-static {0.5 if layout is None else layout.sglang_mem_fraction_static} "
        # a slice has few full-attention layers, and an unbounded pool overflows the DSA kernels' int32 strides;
        # for the full model, 2M tokens (64 32K sequences per engine) keep one layer's K and V under 2 GiB
        f"--sglang-max-total-tokens {1024**2 if layout is None else 2 * 1024**2} "
    )
    if task.has_images:
        sglang_args += "--sglang-enable-multimodal "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        f"{'--accumulate-allreduce-grads-in-fp32 ' if layout is None else ''}"
        f"--update-weight-buffer-size {1 * 1024**3} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--model-name step5 "
        f"{'--reuse-indexer-on-recompute ' if args.reuse_indexer_on_recompute else ''}"
        "--qkv-format thd "
    )
    if task.has_images:
        misc_args += "--custom-model-provider-path miles_plugins.models.step5.vl_model.step5_vl_model_provider "

    train_args = (
        f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} {sglang_args} {misc_args} {args.extra_args} "
    )

    backend = args.create_backend()
    backend.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={"PYTHONPATH": os.environ.get("PYTHONPATH", "")},
        megatron_path=args.megatron_path,
        before_ray_job_submit=_ray_worker_joiner(backend, args),
    )


def _full_model_sampling_args(args: ScriptArgs, layout: _FullModelLayout) -> str:
    # The rollout samples the whole distribution; only the eval truncates it.
    sampling_args = "--rollout-top-p 1.0 --rollout-top-k -1 "
    if not args.eval_aime:
        return sampling_args
    # miles evaluates before the first step unless told to skip it, then every interval and after the last step
    # whatever the interval: steps 0, 10 and 20 of a 20-step run. Eval samples never reach the trainer, so its
    # context caps do not bind them.
    return sampling_args + (
        f"--eval-interval {_FULL_MODEL_EVAL_INTERVAL} "
        f"--eval-prompt-data aime26 {args.data_dir}/aime-2026/aime-2026.jsonl "
        "--n-samples-per-eval-prompt 16 "
        f"--eval-max-response-len {layout.max_response_len} "
        f"--eval-max-context-len {_MAX_PROMPT_LEN + layout.max_response_len} "
        "--eval-temperature 1.0 "
        "--eval-top-p 0.95 "
        # StepFun's top_k 0 (off)
        "--eval-top-k -1 "
    )


# On 2 x 8 expert DP is 1, so each GPU owns the Adam state of its experts plus a quarter of its other params: for the
# full model, 36.5B and 0.84B (13.4B non-expert params, split TP4 then sharded over DP4).
_OPTIMIZER_STATE_OFFLOAD_ARGS = (
    # TE FusedAdam keeps the fp32 masters as int16 remainders of the bf16 params and the moments in bf16.
    "--use-precision-aware-optimizer "
    "--main-params-dtype fp32 "
    "--exp-avg-dtype bf16 "
    "--exp-avg-sq-dtype bf16 "
    # All of it waits in pinned host memory (224 GB per GPU for the full model); a step restores the masters (75 GB)
    # and walks the moments through two 1 GiB windows.
    "--chunked-optimizer-state-offload "
    "--optimizer-state-offload-fraction 1.0 "
    "--optimizer-state-offload-chunk-size-mb 1024 "
    # the full model's fp32 gradients would take 160 GB of HBM per GPU
    "--grad-reduce-in-bf16 "
    # PyTorch rounds each cached pinned block up to a power of two, a third more host memory for the 12 and 24 MiB
    # expert tensors. Blocks above this size are neither rounded nor cached. (pinned_max_round_threshold_mb alone would
    # cache exact-size blocks in the power-of-two free lists, and torch 2.13 hands those to larger requests.)
    """--train-env-vars '{"PYTORCH_ALLOC_CONF": "pinned_max_cached_size_mb:4"}' """
)

# LoRA on the frozen base: Q/K/V/O, the dense MLPs and the shared expert (all-linear less the routed experts; the
# head-wise attention gate is never a target). The adapters' optimizer state is small enough to stay on the GPU, and
# each engine keeps a host copy of the base so a step syncs only the adapters.
_LORA_ARGS = (
    "--lora-rank 16 "
    "--lora-alpha 32 "
    "--lora-dropout 0.0 "
    "--target-modules all-linear "
    "--exclude-modules 'model.layers.*.moe.experts.*' "
    "--lora-base-cpu-backup "
    "--accumulate-allreduce-grads-in-fp32 "
)

# Keyed by (nodes, GPUs per node).
_FULL_MODEL_LAYOUTS = {
    # 2 x 8 B300, EP16: full-parameter training with 32K responses.
    (2, 8): _FullModelLayout(
        cluster="2 x 8 B300",
        max_response_len=32768,
        lr=1e-6,
        trainer_args=_OPTIMIZER_STATE_OFFLOAD_ARGS,
        sglang_attention_backend="fa4",
        sglang_mem_fraction_static=0.75,
    ),
    # 4 x 8 H200, EP32: the FP8 checkpoint takes 81 GB of each GPU's 141 in the TP8 engine, and Hopper serves
    # Step5 with FA3.
    (4, 8): _FullModelLayout(
        cluster="4 x 8 H200",
        max_response_len=8192,
        lr=1e-5,
        trainer_args=_LORA_ARGS,
        sglang_attention_backend="fa3",
        sglang_mem_fraction_static=0.8,
    ),
}


def _ray_worker_joiner(backend: BaseCommandBackend, args: ScriptArgs) -> Callable[[], None] | None:
    if args.ray_hostfile is None:
        return None
    # The head listens on MASTER_ADDR and the join skips the hostfile entry equal to it; with the 127.0.0.1 default
    # every host, the head included, would restart ray against itself.
    master_addr = os.environ.get("MASTER_ADDR")
    assert master_addr, "--ray-hostfile needs MASTER_ADDR, the ray head's address as the hostfile lists it"
    return partial(
        backend.ssh_start_ray_workers,
        master_addr=master_addr,
        num_gpus_per_node=args.num_gpus_per_node,
        hostfile=args.ray_hostfile,
    )


if __name__ == "__main__":
    app()
