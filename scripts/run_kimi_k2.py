"""Kimi-K2 full-parameter GRPO training script.

=====================

One recipe covers both released Kimi-K2 checkpoints. Instruct and Thinking share the
32-node TP8/PP8/CP4/EP32 topology, the dp-attention rollout layout and the whole GRPO /
optimizer schedule; they differ only in the checkpoint they load, the response-length
budget, the eval dataset path, whether truncated importance sampling is enabled, and
whether DeepEP drives the Megatron MoE dispatch.

The checkpoint must already be converted to Megatron `torch_dist`; this script only
submits the training job. It also assumes a Ray cluster that is already joined
across all 32 nodes, so run it with `MILES_SCRIPT_EXTERNAL_RAY=1`: the launcher then
skips `ray start` and submits straight to the running cluster.

=====================

Args:
  --model-name: Model variant, one of Kimi-K2-Instruct / Kimi-K2-Thinking.
  --num-nodes: Number of training nodes (default: 32, the only tested topology).
  --num-gpus-per-node: GPUs per node (default: 8).
  --enable-eval: Run AIME evaluation every 20 steps (default: on).
  --save: Write checkpoints to --output-dir (default: on).
  --model-dir / --data-dir: Checkpoint / dataset directories.

=====================

  MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_kimi_k2.py --model-name Kimi-K2-Thinking
"""

from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

_MODEL_NAMES = Literal["Kimi-K2-Instruct", "Kimi-K2-Thinking"]


@dataclass(frozen=True)
class _Recipe:
    megatron_model_type: str
    hf_checkpoint: str
    torch_dist_name: str
    max_response_len: int
    n_samples_per_eval_prompt: int
    use_tis: bool
    megatron_deepep: bool


_RECIPES: dict[str, _Recipe] = {
    "Kimi-K2-Instruct": _Recipe(
        megatron_model_type="kimi-k2",
        hf_checkpoint="Kimi-K2-Instruct",
        torch_dist_name="Kimi-K2",
        max_response_len=32768,
        n_samples_per_eval_prompt=8,
        use_tis=False,
        megatron_deepep=True,
    ),
    "Kimi-K2-Thinking": _Recipe(
        megatron_model_type="kimi-k2-thinking",
        hf_checkpoint="Kimi-K2-Thinking-fp8",
        torch_dist_name="Kimi-K2-Thinking",
        max_response_len=16384,
        n_samples_per_eval_prompt=16,
        use_tis=True,
        megatron_deepep=False,
    ),
}


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    model_name: _MODEL_NAMES = "Kimi-K2-Instruct"
    num_nodes: int = 32
    num_gpus_per_node: int = 8
    enable_eval: bool = True
    save: bool = True
    num_rollout: int = 100
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    @property
    def recipe(self) -> _Recipe:
        return _RECIPES[self.model_name]


def execute(args: ScriptArgs):
    U = args.create_backend()
    recipe = args.recipe

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{recipe.hf_checkpoint} "
        f"--ref-load {args.model_dir}/{recipe.torch_dist_name}_torch_dist "
        f"--load {args.output_dir}/checkpoints "
    )
    if args.save:
        ckpt_args += f"--save {args.output_dir}/checkpoints " "--save-interval 20 "

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 128 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {recipe.max_response_len} "
        "--rollout-temperature 1 "
        # over-sample, then drop the prompts whose rewards carry no gradient signal
        "--over-sampling-batch-size 256 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--num-steps-per-rollout 4 "
        "--balance-data "
    )

    eval_args = ""
    if args.enable_eval:
        eval_args = (
            "--eval-interval 20 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            f"--n-samples-per-eval-prompt {recipe.n_samples_per_eval_prompt} "
            f"--eval-max-response-len {recipe.max_response_len} "
            "--eval-top-p 1 "
        )

    perf_args = (
        "--tensor-model-parallel-size 8 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 8 "
        "--context-parallel-size 4 "
        "--expert-model-parallel-size 32 "
        "--expert-tensor-parallel-size 1 "
        "--decoder-last-pipeline-num-layers 5 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    if recipe.use_tis:
        grpo_args += "--use-tis "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 16 "
        "--sglang-mem-fraction-static 0.7 "
        # dp attention
        "--sglang-enable-dp-attention "
        "--sglang-dp-size 8 "
        "--sglang-moe-dense-tp-size 1 "
        "--sglang-enable-dp-lm-head "
        "--sglang-ep-size 16 "
        # make every dp rank has 128 concurrency
        "--sglang-server-concurrency 1024 "
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
    )
    if recipe.megatron_deepep:
        # use deepep for megatron; the dispatcher overrides the alltoall from the model args
        misc_args += "--moe-enable-deepep " "--moe-token-dispatcher-type flex "
    misc_args += (
        "--colocate "
        f"--update-weight-buffer-size {4 * 512 * 1024 * 1024} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=recipe.megatron_model_type,
        megatron_path=args.megatron_path,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
