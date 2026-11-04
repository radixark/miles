"""Kimi-K3.1 W4A16 single-LoRA RL on 16 B300 (2 nodes x 8 GPUs).

The 32xH200 recipe's per-stage layout on half the GPUs: training runs PP2 x TP4 x CP2 with EP8 inside each stage,
and the rollout is one TP16 EP16 engine with attention DP 2 x TP 8, one node per DP group. Workload, loss and checks
are shared with the other hardware recipes in `scripts/kimi_k31_w4a16_lora_common.py`.

Each card holds twice the H200 recipe's share of the model, so the host cannot also keep SGLang's copy of the
base next to the trainer's offloaded weights (about 1 TB a node on its own); the base is re-sent every step.

Needs the native MXFP4 checkpoint at --hf-checkpoint (default <model-dir>/Kimi-K3.1) on every node, the dapo-math
dataset under --data-dir, and a Ray cluster over the 2 nodes joined beforehand (MILES_SCRIPT_EXTERNAL_RAY=1).

Args:
    --hf-checkpoint: the MXFP4 checkpoint both the trainer and the engine read.
    --num-rollout: training steps.
    --rollout-batch-size / --n-samples-per-prompt / --global-batch-size: 16 prompts x 16 samples, one optimizer
        step of 256 samples per rollout.
    --rollout-max-response-len: response token cap (3072); each DP group's KV budget (--sglang-max-total-tokens)
        follows it.
    --dynamic-sampling-filter / --no-dynamic-sampling-filter: drop all-equal-reward groups and resample (on);
        rollout/raw_reward_unfiltered keeps the mean reward before the filter.
    --over-sampling-batch-size: groups submitted per refill (4).
    --save-dir / --run-id: adapter checkpoints go to <save-dir>/<run-id>; --skip-saving turns them off.
    --extra-args / --extra-env-vars: appended train.py flags and space-separated K=V env vars, e.g. the NIC names.

python scripts/run_kimi_k31_w4a16_lora_16xb300.py prepare-download --model-dir /scratch/models
python scripts/run_kimi_k31_w4a16_lora_16xb300.py train --hf-checkpoint /scratch/models/Kimi-K3.1 --num-rollout 3
"""

from dataclasses import dataclass
from typing import Literal

import typer
from scripts.kimi_k31_w4a16_lora_common import W4A16LoraArgs, train_w4a16_lora
from scripts.run_kimi_k3 import _prepare_download

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(W4A16LoraArgs):
    hardware: Literal["B300"] = "B300"
    num_nodes: int = 2

    pipeline_parallel_size: int = 2
    context_parallel_size: int = 2
    tp_size_override: int | None = 4
    ep_size_override: int | None = 8

    rollout_tp_size: int | None = 16
    rollout_ep_size: int | None = 16
    rollout_attn_dp_size: int = 2
    lora_base_cpu_backup: bool = False

    rollout_batch_size: int | None = 16
    global_batch_size: int | None = 256
    dynamic_sampling_filter: bool = True
    # kept plus in-flight groups never fall below the 16-group batch (256 requests, the engine's concurrency), so the
    # refill size only bounds how far a step overshoots its last kept group: at most 3 groups
    over_sampling_batch_size: int | None = 4


@app.command()
@U.dataclass_cli
def prepare_download(args: ScriptArgs) -> None:
    _prepare_download(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs) -> None:
    train_w4a16_lora(args)


if __name__ == "__main__":
    app()
