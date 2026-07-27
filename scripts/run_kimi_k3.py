from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "debug_minimal"
    run_id: str = U.create_run_id()
    hf_checkpoint: str = "/root/models/Kimi-K3-4layer-bf16"
    ref_load: str = "/root/models/Kimi-K3-4layer-torch-dist"
    megatron_model_type: str = "kimi-k3-4layer"
    num_gpus_per_node: int = 8
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    sglang_path: str = "/root/sglang/python"
    check_weight_update_equal: bool = False
    extra_args: str = ""

    def __post_init__(self):
        if self.num_nodes != 1 or self.num_gpus_per_node != 8:
            raise NotImplementedError("The verified Kimi K3 training configuration is one 8-GPU node")


def _validate_paths(args: ScriptArgs) -> None:
    for name, path in (("hf_checkpoint", args.hf_checkpoint), ("ref_load", args.ref_load)):
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if not Path(args.sglang_path, "sglang").is_dir():
        raise FileNotFoundError(f"sglang package does not exist under sglang_path: {args.sglang_path}")


@app.command()
@U.dataclass_cli
def prepare_data(args: ScriptArgs) -> None:
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def _execute_train(args: ScriptArgs) -> None:
    _validate_paths(args)
    dataset = Path(args.data_dir) / "dapo-math-17k" / "dapo-math-17k.jsonl"
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset}; run prepare-data first")

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        "--megatron-to-hf-mode raw "
        "--model-name kimi_k3 "
    )

    rollout_args = (
        f"--prompt-data {dataset} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--balance-data "
        f"--rm-type {'deterministic_random' if args.mode == 'debug_minimal' else 'deepscaler'} "
        f"--num-rollout {2 if args.mode == 'debug_minimal' else 3000} "
        f"--rollout-batch-size {8 if args.mode == 'debug_minimal' else 32} "
        f"--n-samples-per-prompt {2 if args.mode == 'debug_minimal' else 8} "
        f"--rollout-max-response-len {32 if args.mode == 'debug_minimal' else 16384} "
        "--rollout-temperature 1 "
        f"--global-batch-size {16 if args.mode == 'debug_minimal' else 256} "
        "--use-dynamic-global-batch-size "
    )

    perf_args = (
        "--tensor-model-parallel-size 8 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 512 "
        "--log-probs-chunk-size 512 "
    )

    optimizer_args = (
        "--optimizer sgd --sgd-momentum 0 --lr 1e-6 --lr-decay-style constant "
        "--weight-decay 0 --use-distributed-optimizer "
        if args.mode == "debug_minimal"
        else (
            "--optimizer adam --lr 1e-6 --lr-decay-style constant "
            "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
            "--optimizer-cpu-offload --optimizer-offload-fraction 0.8 "
            "--overlap-cpu-optimizer-d2h-h2d "
            "--use-precision-aware-optimizer --use-distributed-optimizer "
        )
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.0 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 8 "
        "--sglang-tp-size 8 "
        "--sglang-ep-size 8 "
        "--sglang-cuda-graph-bs 1 2 4 8 16 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-server-concurrency 16 "
        "--use-miles-router "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--no-check-for-nan-in-loss-and-grad "
        "--colocate "
        "--offload-train "
        f"--update-weight-buffer-size {2 * 1024**3} "
        f"--train-memory-margin-bytes {(2 if args.mode == 'debug_minimal' else 4) * 1024**3} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )
    if args.check_weight_update_equal:
        misc_args += "--check-weight-update-equal " "--check-weight-update-skip-list vision_tower. mm_projector. "

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
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={"NCCL_TIMEOUT": "3600", "PYTHONPATH": args.sglang_path},
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs) -> None:
    _execute_train(args)


if __name__ == "__main__":
    app()
