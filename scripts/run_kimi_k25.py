"""
Kimi-K2.5 full-parameter GRPO training script.

=====================

Kimi-K2.5 is a MoE + MLA model (61 layers, 384 experts) shipped as an INT4
(compressed-tensors, group-quantized) checkpoint. Training keeps the INT4
weights for the SGLang rollout while Megatron loads a BF16 reference via the
HF<->Megatron bridge (`--megatron-to-hf-mode bridge`), so there is no offline
`torch_dist` conversion step. The architecture is shared with Kimi-K2-Thinking,
whose Megatron MODEL_ARGS we reuse (`scripts/models/kimi-k2-thinking.sh`).

=====================

Args:
  --model-name: Model variant.
      Kimi-K2.5         Full 61-layer model (requires many nodes; default 32)
      Kimi-K2.5-4layer  4-layer pruned model (single-node CI / smoke test)
  --num-nodes: Number of training nodes. 1 -> single-node 4-layer minimal test.
  --num-gpus-per-node: GPUs per node (default: 8).
  --mode: "normal" or "debug_minimal" (short responses for quick testing).
  --enable-eval: Run AIME evaluation every 20 steps.
  --data-dir / --model-dir: Dataset / model directories (default shared NFS).

=====================

Single-node minimal test (4-layer):
  python scripts/run_kimi_k25.py full-train --model-name Kimi-K2.5-4layer --num-nodes 1

Full model (e.g. 32 nodes):
  1. Start a Ray cluster on all nodes.
  2. On the head node: python scripts/run_kimi_k25.py prepare --model-name Kimi-K2.5 --num-nodes 32
  3. On the head node: python scripts/run_kimi_k25.py train   --model-name Kimi-K2.5 --num-nodes 32
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

# INT4 dequant group size of the published Kimi-K2.5 checkpoints (compressed-tensors).
INT4_GROUP_SIZE = 32

# Megatron bridge identifier consumed by megatron_to_hf dispatch ("kimi_k25" in model_name).
BRIDGE_MODEL_NAME = "kimi_k25"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_org: str = "moonshotai"
    model_name: str = "Kimi-K2.5"
    megatron_model_type: str = "kimi-k2-thinking"
    num_gpus_per_node: int = 8
    enable_eval: bool = False
    num_rollout: int = 3000
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    hardware: Literal["H200", "H100", "B200"] = "H200"

    def __post_init__(self):
        if self.model_name == "Kimi-K2.5":
            self.model_org = "moonshotai"
            self.megatron_model_type = "kimi-k2-thinking"
        elif self.model_name == "Kimi-K2.5-4layer":
            self.model_org = "CharyZeng"
            self.megatron_model_type = "kimi-k25_4layer"
        else:
            raise NotImplementedError(f"{self.model_name} is not supported")

        if self.num_nodes == 1:
            self.mode = "debug_minimal"


def _bf16_ref_dir(args: ScriptArgs) -> str:
    return f"{args.model_dir}/{args.model_name}-bf16"


def _prepare_download(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command(f"hf download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    if args.enable_eval:
        U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


def _convert_to_bf16(args: ScriptArgs):
    """Dequantize the INT4 checkpoint to a BF16 reference for the Megatron bridge."""
    U.exec_command(
        f"python {U.repo_base_dir}/tools/convert_kimi_int4_to_bf16.py "
        f"--model-dir {args.model_dir}/{args.model_name} "
        f"--output-dir {_bf16_ref_dir(args)} "
    )


def _execute_train(args: ScriptArgs):
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {_bf16_ref_dir(args)} "
        "--megatron-to-hf-mode bridge "
        f"--model-name {BRIDGE_MODEL_NAME} "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 20} "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--balance-data "
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 16384} "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
        "--use-dynamic-global-batch-size "
    )

    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += (
            "--eval-interval 20 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    if args.num_nodes == 1:  # single-node 4-layer minimal test
        perf_args = (
            "--tensor-model-parallel-size 8 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 8 "
            "--expert-tensor-parallel-size 1 "
        )
        max_tokens_per_gpu = 2048
    else:  # full model, slime's 32-node setting
        perf_args = (
            "--tensor-model-parallel-size 8 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 8 "
            "--context-parallel-size 4 "
            "--expert-model-parallel-size 32 "
            "--expert-tensor-parallel-size 1 "
            "--decoder-last-pipeline-num-layers 5 "
        )
        max_tokens_per_gpu = 4096

    perf_args += (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {max_tokens_per_gpu} "
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
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
        "--use-distributed-optimizer "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.num_gpus_per_node} "
        "--sglang-mem-fraction-static 0.7 "
        f"--sglang-ep-size {args.num_gpus_per_node} "
        "--sglang-server-concurrency 1024 "
        "--use-rollout-routing-replay "
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--no-check-for-nan-in-loss-and-grad "
        "--colocate "
        "--use-miles-router "
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
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={
            "NCCL_TIMEOUT": "3600",
            "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
            "OPEN_TRAINING_INT4_GROUP_SIZE": str(INT4_GROUP_SIZE),
        },
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    """Full pipeline: download, convert INT4->BF16, train."""
    _prepare_download(args)
    _convert_to_bf16(args)
    _execute_train(args)


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download model/data and dequantize the BF16 reference (run on head node)."""
    _prepare_download(args)
    _convert_to_bf16(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run training only (assumes data is prepared)."""
    _execute_train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
