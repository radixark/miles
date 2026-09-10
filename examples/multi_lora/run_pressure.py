"""Probe Qwen3-30B-A3B capacity, or serve fixed N slots using unchanged Miles.

Requires a shared full BF16 checkpoint, matching Miles/SGLang sources and an
existing four-node Ray cluster. Default placement: 16 trainer + 16 rollout GPUs.
Uses the repository's standard execute_train launcher and process preamble.

Args:
    mode: ``probe`` measures one trainer slot and exits; ``serve`` launches the gateway.
    capacity_report: In serve mode, read N and check model/topology against this report.
    n_adapters: Explicit fixed count in serve mode when not using a capacity report.
    output_dir: Shared path for measurements or the serving job's checkpoints.

Examples:
    MILES_SCRIPT_EXTERNAL_RAY=1 python examples/multi_lora/run_pressure.py \
        --mode probe --hf-checkpoint /models/Qwen3-30B-A3B --output-dir /shared/probe
    MILES_SCRIPT_EXTERNAL_RAY=1 python examples/multi_lora/run_pressure.py \
        --mode serve --capacity-report /shared/probe/slot-capacity.json \
        --hf-checkpoint /models/Qwen3-30B-A3B --output-dir /shared/server
"""

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    mode: Literal["probe", "serve"] = "probe"
    model_dir: str = "/root/models"
    hf_checkpoint: str | None = None
    megatron_path: str = "/root/Megatron-LM"
    sglang_pythonpath: str = ""
    capacity_report: str | None = None
    n_adapters: int = 1
    num_gpus_per_node: int = 8
    actor_num_nodes: int = 2
    training_tp: int = 2
    training_ep: int = 8
    rollout_num_gpus: int = 16
    rollout_tp: int = 2
    lora_rank: int = 16
    lora_alpha: int = 32
    context_length: int = 8192
    capacity_margin_bytes: int = 2 * 1024**3
    engine_host_lora_budget_bytes: int = 0
    tinker_port: int = 10639
    sglang_mem_fraction_static: float = 0.95
    max_running_requests: int = 128
    extra_args: str = ""

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/Qwen3-30B-A3B"
        if (
            min(
                self.num_gpus_per_node,
                self.actor_num_nodes,
                self.training_tp,
                self.training_ep,
                self.rollout_num_gpus,
                self.rollout_tp,
                self.lora_rank,
                self.n_adapters,
            )
            < 1
        ):
            raise ValueError("GPU, parallelism, rank and slot counts must be positive")
        if self.training_gpus % self.training_tp or self.training_gpus % self.training_ep:
            raise ValueError("training GPU count must be divisible by TP and EP")
        if self.rollout_num_gpus % self.rollout_tp:
            raise ValueError("rollout GPU count must be divisible by rollout TP")
        if self.mode == "probe" and (self.capacity_report or self.n_adapters != 1):
            raise ValueError("probe uses exactly one slot; no capacity report is consumed")

    @property
    def training_gpus(self):
        return self.actor_num_nodes * self.num_gpus_per_node


def _slot_count(args):
    if args.capacity_report is None:
        return args.n_adapters
    report = json.loads(Path(args.capacity_report).read_text())
    expected = dict(
        hf_checkpoint=args.hf_checkpoint,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_tokens_per_gpu=args.context_length,
        trainer_gpus=args.training_gpus,
        training_tp=args.training_tp,
        training_ep=args.training_ep,
        keep_k=2,
    )
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"capacity report mismatch for {key}: {report.get(key)!r} != {value!r}")
    n = report["n_slots"]
    if type(n) is not int or n < 1:
        raise ValueError("capacity report does not contain a positive slot estimate")
    return n


def _train_args(args, n):
    checkpoint = (
        f"--hf-checkpoint {shlex.quote(args.hf_checkpoint)} --megatron-to-hf-mode bridge "
        f"--save {shlex.quote(args.output_dir + '/training')} --save-interval 100000 "
        f"--tinker-checkpoint-root {shlex.quote(args.output_dir + '/checkpoints/' + args.run_id)} "
    )
    lora = (
        f"--multi-lora-n-adapters {n} --lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} "
        "--lora-dropout 0 --no-gradient-accumulation-fusion "
        "--target-modules linear_qkv,linear_proj,linear_fc1,linear_fc2 "
    )
    perf = (
        "--train-backend megatron --qkv-format thd "
        f"--actor-num-nodes {args.actor_num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--tensor-model-parallel-size {args.training_tp} --expert-model-parallel-size {args.training_ep} "
        "--expert-tensor-parallel-size 1 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--seq-length {args.context_length} --rollout-max-context-len {args.context_length} "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.context_length} "
        f"--micro-batch-size 1 --global-batch-size {args.training_gpus // args.training_tp} "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash "
    )
    optimizer = "--optimizer adam --lr 1e-5 "
    sglang = (
        f"--rollout-num-gpus {args.rollout_num_gpus} --rollout-num-gpus-per-engine {args.rollout_tp} "
        f"--sglang-ep-size {args.rollout_tp} --sglang-lora-backend triton --sglang-max-lora-rank {args.lora_rank} "
        f"--sglang-max-loaded-loras {2 * n} --sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-context-length {args.context_length} --sglang-max-running-requests {args.max_running_requests} "
        f"--sglang-chunked-prefill-size {args.context_length} --sglang-cuda-graph-max-bs-decode 16 "
        "--sglang-watchdog-timeout 3600 --sglang-moe-runner-backend triton "
    )
    probe = ""
    if args.mode == "probe":
        probe = (
            f"--capacity-output {shlex.quote(args.output_dir + '/slot-capacity.json')} "
            f"--capacity-margin-bytes {args.capacity_margin_bytes} "
            f"--engine-host-lora-budget-bytes {args.engine_host_lora_budget_bytes} --engine-keep-k 2 "
        )
    return f"{checkpoint} {lora} {perf} {optimizer} {sglang} {probe} --tinker-server-port {args.tinker_port} {args.extra_args}"


@U.dataclass_cli
def main(args: ScriptArgs):
    n = _slot_count(args)
    env = {"RAY_DEDUP_LOGS": "0"}
    if args.sglang_pythonpath:
        env["PYTHONPATH"] = args.sglang_pythonpath
    # W&B is controlled only by the SDK scripts; no credentials enter this argv.
    U.execute_train(
        train_args=_train_args(args, n),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3-30B-A3B",
        config=args,
        megatron_path=args.megatron_path,
        extra_env_vars=env,
        train_script="examples/multi_lora/slot_capacity.py" if args.mode == "probe" else "serve_tinker.py",
    )


if __name__ == "__main__":
    typer.run(main)
