"""Multi-turn geo3k image RL for Qwen3.5-35B-A3B on the plugin text model, with the frozen vision tower, in token space.

SGLang expands each turn's image placeholders in token space, runs the vision tower unsharded (DP encoder) and caches
the per-image features on disk; every trainer rank rebuilds the same tensors from the shared image files in the
background (``--multimodal-token-space-processing``). Routing replay records each turn's own rows, so the trainer
replays the experts the engine chose for every turn. The rollout, environment and reward are those of
``run_geo3k_vlm_multi_turn.py``.

Args:
    --num-gpus-per-node: GPUs on the node, shared by the trainer and the engines.
    --num-rollout: Training rollouts (default 100).

Example:
    python examples/geo3k_vlm/multi_turn/run_geo3k_vlm_multi_turn_token_space.py
"""

from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils

MODEL_NAME = "Qwen3.5-35B-A3B"
MEGATRON_MODEL_TYPE = "qwen3.5-35B-A3B"
DATASET = "VeraIsHere/geo3k_imgurl_processed"


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
    data_dir = f"{args.data_dir}/geo3k_imgurl_processed"

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
        """--multimodal-keys '{"image": "images"}' """
        f"--shared-media-dir {data_dir}/media "
        "--rm-type math "
        "--apply-chat-template "
        """--apply-chat-template-kwargs '{"enable_thinking": false}' """
        "--custom-generate-function-path examples.geo3k_vlm.multi_turn.rollout.generate "
        "--max-turns 3 "
        "--rollout-interaction-env-path examples.geo3k_vlm.multi_turn.env_geo3k "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
    )

    # SGLang expands the images and runs the vision tower unsharded like the trainer; the per-item features are
    # cached on disk once and read back by the trainer.
    multimodal_args = (
        "--multimodal-token-space-processing "
        "--trainer-mm-processor-worker-num 4 --trainer-mm-io-worker-num 4 "
        "--sglang-enable-token-space-processor "
        "--sglang-mm-processor-worker-num 8 "
        "--sglang-mm-enable-dp-encoder "
        "--sglang-enable-token-space-preprocess-cache "
        f"--sglang-token-space-preprocess-cache-dir {data_dir}/feature_cache "
        "--sglang-mm-preprocess-cache-size-mb 16384 "
        "--sglang-token-space-preprocess-cache-trust-url "
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
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = "--rollout-num-gpus-per-engine 2 --sglang-mem-fraction-static 0.6 "

    # The MoE trains TP2 x EP4 and replays the engines' routing (R3) over the media-expanded sequence.
    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 4 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
        "--use-rollout-routing-replay "
        "--log-probs-chunk-size 4096 "
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
