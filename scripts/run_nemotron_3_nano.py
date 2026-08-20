"""NVIDIA Nemotron-3-Nano GRPO training script (single node, 8 GPUs).

=====================

One recipe covers both Nano variants: the dense 4B and the 128-expert 30B-A3B MoE. They
differ in the parallelism cell, the dynamic-batch token budget, activation recompute,
log-prob chunking, the MoE-only rollout routing replay, and the rollout shape (the 4B fits
8 samples x 4096 tokens per prompt, the 30B-A3B runs 4 x 1024). Everything else -- rollout
dataset, GRPO constants, optimizer schedule, SGLang engine layout -- is shared.

nemotron_h is a hybrid Mamba2 + Attention architecture loaded through NVIDIA
``megatron.bridge`` (``--megatron-to-hf-mode bridge``): ``--ref-load`` points at the HF
checkpoint itself, so no offline torch_dist conversion step is needed.

Both variants are 10-step smoke tests, hence no eval. The parallelism fields of `_Recipe`
are the knobs a sweep swaps per cell (PP=2, CP=2, TP=4, TP=2 x PP=2 for the dense 4B;
EP for the MoE).

=====================

Args:
  --model-name: Model variant, NVIDIA-Nemotron-3-Nano-4B-BF16 or NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.
  --num-gpus-per-node: GPUs per node (default: 8).
  --num-rollout: Rollout steps (default: 10, i.e. the smoke test).
  --model-dir / --data-dir: Checkpoint / dataset directories.

=====================

  python scripts/run_nemotron_3_nano.py --model-name NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

_MODEL_NAMES = Literal["NVIDIA-Nemotron-3-Nano-4B-BF16", "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"]


@dataclass(frozen=True)
class _Recipe:
    megatron_model_type: str
    tp: int
    pp: int
    cp: int
    ep: int
    etp: int
    max_tokens_per_gpu: int
    recompute: bool
    log_probs_chunk_size: int | None
    use_rollout_routing_replay: bool
    global_batch_size: int
    n_samples_per_prompt: int
    rollout_max_response_len: int


_RECIPES: dict[str, _Recipe] = {
    "NVIDIA-Nemotron-3-Nano-4B-BF16": _Recipe(
        megatron_model_type="nemotron-3-nano-4b",
        tp=2,
        pp=2,
        cp=1,
        # Dense architecture, so EP/ETP are N/A.
        ep=1,
        etp=1,
        max_tokens_per_gpu=9216,
        recompute=False,
        log_probs_chunk_size=None,
        use_rollout_routing_replay=False,
        global_batch_size=256,
        n_samples_per_prompt=8,
        rollout_max_response_len=4096,
    ),
    "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": _Recipe(
        megatron_model_type="nemotron-3-nano-30b-a3b",
        tp=2,
        pp=2,
        cp=1,
        # 64 of the 128 experts per rank.
        ep=2,
        etp=1,
        max_tokens_per_gpu=1024,
        recompute=True,
        log_probs_chunk_size=128,
        use_rollout_routing_replay=True,
        global_batch_size=128,
        n_samples_per_prompt=4,
        rollout_max_response_len=1024,
    ),
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    model_name: _MODEL_NAMES = "NVIDIA-Nemotron-3-Nano-4B-BF16"
    num_gpus_per_node: int = 8
    num_rollout: int = 10
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    @property
    def recipe(self) -> _Recipe:
        return _RECIPES[self.model_name]


def execute(args: ScriptArgs):
    recipe = args.recipe
    hf_checkpoint = f"{args.model_dir}/{args.model_name}"

    ckpt_args = (
        # --ref-load at an HF directory routes load_checkpoint through megatron.bridge,
        # so the run needs no torch_dist copy of the weights.
        f"--hf-checkpoint {hf_checkpoint} "
        f"--ref-load {hf_checkpoint} "
        f"--save {args.output_dir}/checkpoints "
        "--save-interval 20 "
        "--megatron-to-hf-mode bridge "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        f"--n-samples-per-prompt {recipe.n_samples_per_prompt} "
        f"--rollout-max-response-len {recipe.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {recipe.global_batch_size} "
        "--balance-data "
    )

    perf_args = (
        f"--tensor-model-parallel-size {recipe.tp} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {recipe.pp} "
        f"--context-parallel-size {recipe.cp} "
        f"--expert-model-parallel-size {recipe.ep} "
        f"--expert-tensor-parallel-size {recipe.etp} "
    )
    if recipe.recompute:
        perf_args += "--recompute-granularity full " "--recompute-method uniform " "--recompute-num-layers 1 "
    perf_args += "--use-dynamic-batch-size " f"--max-tokens-per-gpu {recipe.max_tokens_per_gpu} "
    if recipe.log_probs_chunk_size is not None:
        perf_args += f"--log-probs-chunk-size {recipe.log_probs_chunk_size} "

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.7 "
    if recipe.use_rollout_routing_replay:
        # Replay the exact rollout routing during the training forward so train logprobs
        # match rollout logprobs (needed for MoE).
        sglang_args += "--use-rollout-routing-replay "

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend auto "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.num_nodes * args.num_gpus_per_node} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=recipe.megatron_model_type,
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
