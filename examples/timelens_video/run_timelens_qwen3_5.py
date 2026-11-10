"""TimeLens temporal-grounding video RL for Qwen3.5 (plugin text model, frozen vision tower, token space).

SGLang samples each video and expands its placeholders in token space; the trainer rebuilds the same tensors
from the shared video files (``--multimodal-token-space-processing``) and takes the vision placement and mRoPE
positions from SGLang. The recipe follows TimeLens GRPO: tIoU reward, no KL, no std normalization, constant
lr 1e-6. Our addition to it: responses cut at the length limit score 0 in training (``zero_truncated_rewards``),
because the reward reads only the first predicted span and the policy otherwise grows answers to the limit.

Needs the data from ``prepare_data.py`` under ``<data-dir>/timelens``.

Args:
    --model-name: Qwen3.5-9B (default) or Qwen3.5-35B-A3B.
    --num-gpus-per-node: GPUs on the node, shared by the trainer and the engines.
    --num-rollout: Training steps (default 300).

Example:
    python examples/timelens_video/prepare_data.py /root/datasets/timelens
    python examples/timelens_video/run_timelens_qwen3_5.py
"""

import json
from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils

# TimeLens video sampling: 2 fps, at most 448 frames, 64 to 14336 visual tokens.
# Videos decode on CPU: on the GPU, concurrent float32 frame resizes run the colocated engines out of memory.
_MM_PROCESS_CONFIG = {
    "video": {"fps": 2.0, "max_frames": 448, "min_pixels": 65536, "total_pixels": 14680064},
    "video_preprocessing_device": "cpu",
}

# What differs between the two models; everything else in execute() is shared. The 35B-A3B MoE trains TP2 x EP4
# with the Adam state on the host and replays the engines' expert routing (R3); its engines take 2 GPUs each, run
# the vision tower unsharded like the trainer (DP encoder) and cache the per-video features on disk.
_MODEL_PROFILES = {
    "Qwen3.5-9B": dict(
        rollout_batch_size=16,
        global_batch_size=128,
        weight_decay=0.0,
        eps_clip_high=0.2,
        adam_beta2=0.999,
        optimizer_offload_args="",
        parallel_args="--tensor-model-parallel-size 4 ",
        max_tokens_per_gpu=20480,
        routing_replay_args="",
        gpus_per_engine=1,
        sglang_mm_workers=16,
        trainer_mm_workers=16,
        engine_vision_args="",
    ),
    "Qwen3.5-35B-A3B": dict(
        rollout_batch_size=32,
        global_batch_size=256,
        weight_decay=0.1,
        eps_clip_high=0.28,
        adam_beta2=0.98,
        optimizer_offload_args="--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer ",
        parallel_args=(
            "--tensor-model-parallel-size 2 --expert-model-parallel-size 4 --expert-tensor-parallel-size 1 "
        ),
        max_tokens_per_gpu=8192,
        routing_replay_args="--use-rollout-routing-replay --log-probs-chunk-size 4096 ",
        gpus_per_engine=2,
        sglang_mm_workers=8,
        trainer_mm_workers=4,
        engine_vision_args=(
            "--sglang-mm-enable-dp-encoder "
            "--sglang-enable-token-space-preprocess-cache "
            "--sglang-token-space-preprocess-cache-dir {data_dir}/feature_cache "
            "--sglang-mm-preprocess-cache-size-mb 16384 "
            "--sglang-token-space-preprocess-cache-trust-url "
        ),
    ),
}


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    model_name: str = "Qwen3.5-9B"
    num_gpus_per_node: int = 8
    num_rollout: int = 300
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""

    @property
    def megatron_model_type(self) -> str:
        # Qwen3.5-9B -> qwen3.5-9B
        return self.model_name.replace("Qwen3.5-", "qwen3.5-")


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir}")
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    # The multimodal model keeps the text model's parameter names, so the text conversion applies.
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    U = args.create_backend()
    data_dir = f"{args.data_dir}/timelens"
    profile = _MODEL_PROFILES[args.model_name]

    # A fresh --load directory starts the run from the --ref-load weights at rollout 0.
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {args.output_dir}/{args.run_id}/checkpoints "
        "--megatron-to-hf-mode raw "
        "--custom-model-provider-path miles_plugins.models.qwen3_5.qwen3_5_mm_model_provider "
    )

    rollout_args = (
        f"--prompt-data {data_dir}/timelens_grpo_train.parquet "
        "--input-key problem "
        "--label-key label "
        "--apply-chat-template "
        """--apply-chat-template-kwargs '{"enable_thinking": false}' """
        "--rollout-shuffle "
        "--custom-rm-path examples.timelens_video.reward.timelens_tiou "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {profile['rollout_batch_size']} "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 1.0 "
        f"--global-batch-size {profile['global_batch_size']} "
    )

    # SGLang samples and expands the videos; the trainer rebuilds the same tensors from the shared files.
    multimodal_args = (
        """--multimodal-keys '{"video": "videos"}' """
        "--multimodal-token-space-processing "
        f"--trainer-mm-processor-worker-num {profile['trainer_mm_workers']} "
        f"--trainer-mm-io-worker-num {profile['trainer_mm_workers']} "
        "--sglang-enable-token-space-processor "
        f"--sglang-mm-process-config '{json.dumps(_MM_PROCESS_CONFIG)}' "
        f"{profile['engine_vision_args'].format(data_dir=data_dir)}"
    )

    eval_args = (
        "--eval-interval 10 "
        f"--eval-prompt-data charades {data_dir}/timelens_bench_charades_600.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 512 "
        "--eval-temperature 0 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--disable-grpo-std-normalization "
        "--kl-loss-coef 0.00 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        f"--eps-clip-high {profile['eps_clip_high']} "
        "--custom-reward-post-process-path examples.timelens_video.reward.zero_truncated_rewards "
        "--use-tis "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        f"--weight-decay {profile['weight_decay']} "
        "--adam-beta1 0.9 "
        f"--adam-beta2 {profile['adam_beta2']} "
        f"{profile['optimizer_offload_args']}"
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {profile['gpus_per_engine']} --sglang-mem-fraction-static 0.6 "
        f"--sglang-mm-processor-worker-num {profile['sglang_mm_workers']} "
    )

    perf_args = (
        f"{profile['parallel_args']}"
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {profile['max_tokens_per_gpu']} "
        f"{profile['routing_replay_args']}"
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
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars={"PYTHONPATH": args.megatron_path},
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
