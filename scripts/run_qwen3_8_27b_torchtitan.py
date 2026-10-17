"""Qwen3.8-27B GRPO on the torchtitan backend, one node.

=====================

Qwen3.8-27B ships the same config.json as Qwen3.5-27B, so torchtitan trains it as its
own `qwen3_5` / `27B` flavor and loads the HF directory as-is: no torch_dist
conversion. The rollout recipe matches `run_qwen3_dense.py --model-name Qwen3.8-27B`
-- dapo-math-17k @ 8k response len, AIME-2024 eval @ 16k every 20 rollouts -- so the
torchtitan and Megatron curves can be read against each other on one model.

torchtitan has no CPU optimizer, so the fp32 master weights and Adam state stay on
the GPUs, FSDP-sharded over whatever the TP and PP degrees leave over: about 54 GB
per GPU on eight. Colocated, the trainer sleeps to host RAM while SGLang generates.
Context parallelism is rejected by torchtitan for qwen3_5.

=====================

Args:
  --num-gpus-per-node: GPUs per node (default: 8).
  --num-rollout: Rollouts to train (default: 3000).
  --tp-size / --pp-size: torchtitan tensor / pipeline parallel degrees (default: 1).
  --enable-eval: Run AIME evaluation every 20 rollouts (default: on).
  --save: Write checkpoints to --output-dir (default: on).
  --model-dir / --data-dir: Checkpoint / dataset directories.

=====================

  python scripts/run_qwen3_8_27b_torchtitan.py

Tensor-parallel within pairs of GPUs, FSDP across the four pairs:

  python scripts/run_qwen3_8_27b_torchtitan.py --tp-size 2
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

HF_REPO = "Qwen/Qwen3.8-27B"
MODEL_NAME = "Qwen3.8-27B"

RESPONSE_LEN = 8192
# Rotary tables and the buffers pipeline stages exchange are sized from this, so
# it has to cover the longest prompt plus the whole response.
SEQ_LEN = RESPONSE_LEN + 2048


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    num_gpus_per_node: int = 8
    num_rollout: int = 3000
    tp_size: int = 1
    pp_size: int = 1
    enable_eval: bool = True
    save: bool = True
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    extra_args: str = ""


def prepare(args: ScriptArgs):
    backend = args.create_backend()
    backend.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    backend.exec_command_cpu(f"hf download {HF_REPO} --local-dir {args.model_dir}/{MODEL_NAME}")
    backend.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    backend.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


def execute(args: ScriptArgs):
    # No --ref-load: at --kl-loss-coef 0 a reference model only costs a second
    # 27B trainer in host RAM and a forward pass per step.
    ckpt_args = f"--hf-checkpoint {args.model_dir}/{MODEL_NAME} "
    if args.save:
        ckpt_args += (
            f"--load {args.output_dir}/checkpoints " f"--save {args.output_dir}/checkpoints " "--save-interval 20 "
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
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {RESPONSE_LEN} "
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
            # the Megatron recipe's eval length; eval runs in SGLang only, so --seq-length does not bound it
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    # The FSDP shard degree is deliberately absent: torchtitan infers it from
    # what the other degrees leave over.
    perf_args = (
        "--train-backend torchtitan "
        "--titan-model-name qwen3_5 "
        "--titan-model-flavor 27B "
        f"--seq-length {SEQ_LEN} "
        f"--tensor-model-parallel-size {args.tp_size} "
        f"--pipeline-model-parallel-size {args.pp_size} "
        "--context-parallel-size 1 "
        "--micro-batch-size 1 "
        "--gradient-checkpointing "
        "--update-weight-buffer-size 1073741824 "
        """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
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
    )

    # One engine per GPU at 0.8, the Megatron recipe's setting: the linear-attention
    # state is resident per sequence, so the pool size is what bounds the batch. See
    # docs/models/qwen/qwen3-8.md.
    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.8 "

    misc_args = (
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )

    args.create_backend().execute_train(
        train_args=(
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{eval_args} "
            f"{perf_args} "
            f"{grpo_args} "
            f"{optimizer_args} "
            f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
            f"{sglang_args} "
            f"{misc_args} "
            f"{args.extra_args} "
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=None,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
