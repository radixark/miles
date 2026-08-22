"""Qwen3 / Qwen3.5 / Qwen3.6 dense GRPO training script.

=====================

One recipe covers the whole dense line. The variants differ only in tensor parallelism,
the dynamic-batch token budget, the SGLang engine size and memory fraction, whether the
optimizer state is offloaded to host RAM, and the default rollout count. Everything else
-- rollout dataset, GRPO constants, optimizer schedule, eval -- is shared. Qwen3.6-27B is
architecturally identical to Qwen3.5-27B and takes the same knobs.

The checkpoint must already be converted to Megatron `torch_dist`; this script only
submits the training job.

=====================

Args:
  --model-name: Model variant, see the keys of _RECIPES below.
  --num-gpus-per-node: GPUs per node (default: 8). Set it when running on a slice of a node.
  --cuda-visible-devices: Restrict the launcher and its ray workers to these GPUs.
  --num-rollout: Override the variant's default rollout count.
  --enable-eval: Run AIME evaluation every 20 steps (default: on).
  --save: Write checkpoints to --output-dir (default: on).
  --model-dir / --data-dir: Checkpoint / dataset directories.

=====================

  python scripts/run_qwen3_dense.py --model-name Qwen3.5-27B

Four-GPU slice of an eight-GPU node:

  python scripts/run_qwen3_dense.py --model-name Qwen3-4B \\
      --num-gpus-per-node 4 --cuda-visible-devices 4,5,6,7
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

_MODEL_NAMES = Literal[
    "Qwen3-4B",
    "Qwen3-32B",
    "Qwen3.5-4B",
    "Qwen3.5-9B",
    "Qwen3.5-27B",
    "Qwen3.6-27B",
]


@dataclass(frozen=True)
class _Recipe:
    megatron_model_type: str
    tensor_model_parallel_size: int
    max_tokens_per_gpu: int
    rollout_num_gpus_per_engine: int
    sglang_mem_fraction_static: float
    optimizer_cpu_offload: bool
    num_rollout: int = 3000
    extra_sglang_args: str = ""
    # the quick-start recipe serves the dashboard; the rest stay quiet by default
    use_dashboard: bool = False


# Qwen3-32B decodes a wide batch sweep, so it pins the cuda graph batch sizes.
_QWEN3_32B_CUDA_GRAPH_BS = " ".join(str(bs) for bs in [1, 2, 4, 8, *range(16, 257, 8)])

_RECIPES: dict[str, _Recipe] = {
    "Qwen3-4B": _Recipe("qwen3-4B", 2, 9216, 2, 0.7, False, use_dashboard=True),
    "Qwen3-32B": _Recipe(
        "qwen3-32B",
        8,
        20480,
        8,
        0.7,
        True,
        num_rollout=5,
        extra_sglang_args=f"--sglang-cuda-graph-bs {_QWEN3_32B_CUDA_GRAPH_BS} ",
    ),
    # SGLang TP>1 produces garbage output for Qwen3.5 on 0.5.9, which miles still pins
    # (https://github.com/sgl-project/sglang/issues/21039), hence one GPU per engine.
    "Qwen3.5-4B": _Recipe("qwen3.5-4B", 2, 9216, 1, 0.7, False),
    "Qwen3.5-9B": _Recipe("qwen3.5-9B", 2, 9216, 1, 0.6, False),
    "Qwen3.5-27B": _Recipe("qwen3.5-27B", 4, 8192, 1, 0.5, True),
    "Qwen3.6-27B": _Recipe("qwen3.6-27B", 4, 8192, 1, 0.5, True),
}


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    model_name: _MODEL_NAMES = "Qwen3-4B"
    num_gpus_per_node: int = 8
    cuda_visible_devices: str = ""
    enable_eval: bool = True
    save: bool = True
    num_rollout: int = 0
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    @property
    def recipe(self) -> _Recipe:
        return _RECIPES[self.model_name]


def execute(args: ScriptArgs):
    U = args.create_backend()
    if args.cuda_visible_devices:
        # exported rather than passed along: ray reads it when it starts the head
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
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
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout or args.recipe.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    eval_args = ""
    if args.enable_eval:
        eval_args = (
            "--eval-interval 20 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    perf_args = (
        f"--tensor-model-parallel-size {args.recipe.tensor_model_parallel_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.recipe.max_tokens_per_gpu} "
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

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    if args.recipe.optimizer_cpu_offload:
        optimizer_args += (
            "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
        )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.recipe.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.recipe.sglang_mem_fraction_static} "
        f"{args.recipe.extra_sglang_args}"
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )
    if args.recipe.use_dashboard:
        misc_args += "--use-miles-dashboard " f"--dump-details {args.output_dir}/dump_details "

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
        megatron_model_type=args.recipe.megatron_model_type,
        megatron_path=args.megatron_path,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
