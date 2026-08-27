"""GLM-5.3-Flash DAPO training.

Assumes an already-running ray cluster (MILES_SCRIPT_EXTERNAL_RAY=1) and a
converted torch_dist reference checkpoint at
{ckpt_dir}/{megatron_model_type}_torch_dist.

Args:
    model-name: "GLM-5.3-Flash" (45-layer, 8 nodes x 4 GPUs, TP8 PP4 EP16
        placeholder mirroring glm5.2's GB300 parallelism at 32 GPUs) or
        "GLM-5.3-Flash-8layer" (smoke slice, 2 nodes x 4 GPUs, TP2 PP2 EP2).
    enable-r3: add --use-rollout-routing-replay (indexer replay lands once the
        sglang-side kpool top-k capture is ported).

Usage (inside the head-node container):
    python scripts/run_glm5_3_flash.py train --model-name GLM-5.3-Flash-8layer --num-rollout 5
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_MODEL_REGISTRY = {
    "GLM-5.3-Flash": "glm5.3-flash",
    "GLM-5.3-Flash-8layer": "glm5.3-flash-8layer",
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    model_name: Literal["GLM-5.3-Flash", "GLM-5.3-Flash-8layer"] = "GLM-5.3-Flash-8layer"
    num_nodes: int = 2
    num_gpus_per_node: int = 4
    run_id: str = "glm53flash-dapo"
    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    ckpt_dir: str = "/root/ckpt"
    data_dir: str = "/root/datasets"
    save_dir: str = "/root/shared_data"
    megatron_path: str = "/root/Megatron-LM"
    num_rollout: int = 5
    rollout_max_response_len: int = 4096
    check_weight_update: bool = True
    enable_r3: bool = False
    skip_saving: bool = True
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"


def _train(args: ScriptArgs):
    shape = (args.num_nodes, args.num_gpus_per_node)
    assert shape in (
        (8, 4),
        (2, 4),
    ), "the parallel configs below are shaped for 8x4 (full) or 2x4 (8layer)"

    megatron_model_type = _MODEL_REGISTRY[args.model_name]

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} " f"--ref-load {args.ckpt_dir}/{megatron_model_type}_torch_dist "
    )
    if not args.skip_saving:
        load_save_path = f"{args.save_dir}/{args.run_id}/checkpoints"
        ckpt_args += (
            f"--load {load_save_path} --save {load_save_path} --save-interval 10 "
            "--no-save-optim --no-save-rng --no-load-optim --no-load-rng "
        )

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 8 "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
    )

    if shape == (8, 4):
        parallel_args = (
            "--tensor-model-parallel-size 8 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 4 "
            "--decoder-first-pipeline-num-layers 11 "
            "--decoder-last-pipeline-num-layers 12 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 16 "
            "--expert-tensor-parallel-size 1 "
        )
        engine_args = "--rollout-num-gpus-per-engine 8 " "--sglang-tp-size 8 " "--sglang-ep-size 8 "
    else:
        parallel_args = (
            "--tensor-model-parallel-size 2 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 2 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 2 "
            "--expert-tensor-parallel-size 1 "
        )
        engine_args = "--rollout-num-gpus-per-engine 4 " "--sglang-tp-size 4 " "--sglang-ep-size 4 "

    perf_args = (
        f"{parallel_args}"
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--max-tokens-per-gpu 8192 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
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

    sglang_args = (
        f"{engine_args}"
        "--sglang-dp-size 1 "
        "--sglang-chunked-prefill-size 8192 "
        "--sglang-disable-radix-cache "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--accumulate-allreduce-grads-in-fp32 "
        f"--update-weight-buffer-size {1 * 1024**3} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--train-memory-margin-bytes 3221225472 "
        "--offload-train-target disk "
        "--offload-train-disk-dir /tmp/train_offload "
        "--sglang-mem-fraction-static 0.7 "
        "--colocate "
        "--model-name glm5_next "
        "--qkv-format thd "
        "--rollout-health-check-interval 300 "
        "--distributed-timeout-minutes 60 "
        "--rollout-health-check-timeout 300 "
    )
    if args.check_weight_update:
        misc_args += "--check-weight-update-equal " "--check-weight-update-skip-list visual. "
    if args.enable_r3:
        misc_args += "--use-rollout-routing-replay "

    train_args = (
        f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} {sglang_args} {misc_args} {args.extra_args} "
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

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    app()
