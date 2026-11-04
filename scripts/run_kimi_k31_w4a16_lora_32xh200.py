"""Kimi-K3.1 W4A16 single-LoRA RL on 32 H200 (4 nodes x 8 GPUs).

Training runs PP4 x TP4 x CP2 with EP8 inside each stage; the rollout is one TP32 EP32 engine with attention
DP 4 x TP 8, one node per DP group. Workload, loss and checks are shared with the other hardware recipes in
`scripts/kimi_k31_w4a16_lora_common.py`.

Needs the native MXFP4 checkpoint at --hf-checkpoint (default <model-dir>/Kimi-K3.1), the dapo-math dataset under
--data-dir, and a Ray cluster over the 4 nodes joined beforehand (MILES_SCRIPT_EXTERNAL_RAY=1).

Args:
    --hf-checkpoint: the MXFP4 checkpoint both the trainer and the engine read.
    --num-rollout: training steps.
    --rollout-max-response-len: response token cap (3072); each DP group's KV budget (--sglang-max-total-tokens)
        follows it.
    --save-dir / --run-id: adapter checkpoints go to <save-dir>/<run-id>; --skip-saving turns them off.
    --extra-args / --extra-env-vars: appended train.py flags and space-separated K=V env vars, e.g. the NIC names.

python scripts/run_kimi_k31_w4a16_lora_32xh200.py prepare-download --model-dir /root/models
python scripts/run_kimi_k31_w4a16_lora_32xh200.py train --hf-checkpoint /root/models/Kimi-K3.1 --num-rollout 100
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
    hardware: Literal["H200"] = "H200"
    num_nodes: int = 4

    pipeline_parallel_size: int = 4
    context_parallel_size: int = 2
    tp_size_override: int | None = 4
    ep_size_override: int | None = 8

    rollout_tp_size: int | None = 32
    rollout_ep_size: int | None = 32
    rollout_attn_dp_size: int = 4


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
