"""Single-turn geo3k image RL for Qwen3.8-27B on the plugin text model, with the frozen vision tower, in token space.

Qwen3.8-27B ships the Qwen3.5-27B architecture, so it runs on the Qwen3.5 multimodal provider: SGLang expands
each image's placeholders in token space and every trainer rank rebuilds the same tensors from the shared image
files (``--multimodal-token-space-processing``), taking the vision placement and mRoPE positions from SGLang.

Args:
    --num-gpus-per-node: GPUs on the node, shared by the trainer and the engines.
    --num-rollout: Training steps (default 100).

Example:
    python examples/geo3k_vlm/run_geo3k_qwen3_8_27b.py
"""

from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils

MODEL_NAME = "Qwen3.8-27B"
MEGATRON_MODEL_TYPE = "qwen3.8-27B"
DATASET = "chenhegu/geo3k_imgurl"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    num_gpus_per_node: int = 8
    num_rollout: int = 100
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir {args.model_dir}/{MODEL_NAME}")
    U.hf_download_dataset(DATASET, data_dir=args.data_dir)
    # The multimodal model keeps the text model's parameter names, so the text conversion applies.
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{MODEL_NAME}",
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    U = args.create_backend()
    data_dir = f"{args.data_dir}/geo3k_imgurl"

    # A fresh --load directory starts the run from the --ref-load weights at rollout 0.
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{MODEL_NAME} "
        f"--ref-load {args.model_dir}/{MODEL_NAME}_torch_dist "
        f"--load {args.output_dir}/{args.run_id}/checkpoints "
        "--megatron-to-hf-mode raw "
        "--custom-model-provider-path miles_plugins.models.qwen3_5.qwen3_5_mm_model_provider "
    )

    rollout_args = (
        f"--prompt-data {data_dir}/train.parquet "
        "--input-key problem "
        "--label-key answer "
        "--apply-chat-template "
        """--apply-chat-template-kwargs '{"enable_thinking": false}' """
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 64 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 512 "
    )

    # SGLang expands the images; the trainer rebuilds the same tensors from the shared files.
    multimodal_args = (
        """--multimodal-keys '{"image": "images"}' """
        f"--shared-media-dir {data_dir}/media "
        "--multimodal-token-space-processing "
        "--sglang-enable-token-space-processor "
    )

    eval_args = (
        "--eval-interval 20 "
        f"--eval-prompt-data geo3k {data_dir}/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 4096 "
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

    # CPU Adam, as scripts/run_qwen3_dense.py uses for the 27B text recipe.
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

    # One GPU per engine keeps the vision tower unsharded, as the trainer runs it.
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.8 " "--sglang-mm-processor-worker-num 16 "
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{multimodal_args} "
        f"{eval_args} "
        f"{grpo_args} "
        f"{optimizer_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        megatron_path=args.megatron_path,
        extra_env_vars={"PYTHONPATH": args.megatron_path},
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
