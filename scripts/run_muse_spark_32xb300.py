"""Train full Muse Spark with GRPO on 32 B300 GPUs (4 nodes of 8). This file sets the cluster size, parallelism and
engine sizing; the recipe and its other options are in scripts/muse_spark/common.py.

Example:
    MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_muse_spark_32xb300.py
    python scripts/run_muse_spark_32xb300.py --partial --num-gpus-per-node 4  # smoke test on 4 GPUs
"""

from dataclasses import dataclass

import typer
from muse_spark import common

import miles.utils.external_utils.command_utils.legacy as U


@dataclass(kw_only=True)
class ScriptArgs(common.ScriptArgs):
    num_nodes: int = 4
    num_gpus_per_node: int = 8
    # TP8/PP4 with 15/16/16/15 layers per stage.
    tensor_model_parallel_size: int = 8
    pipeline_model_parallel_size: int = 4
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 8
    decoder_first_last_pipeline_num_layers: int = 15
    # One engine per node: 288 GB per GPU holds the model on 8 GPUs.
    rollout_num_gpus_per_engine: int = 8
    max_tokens_per_gpu: int = 8192
    # The KV pool at 0.83 covers 128 running requests per engine, all 512 samples of a rollout across 4 engines.
    sglang_mem_fraction_static: float = 0.83
    sglang_max_running_requests: int = 128
    sglang_cuda_graph_bs_decode: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128)
    # As in the validated 30-step B300 run.
    pin_rollout_manager_to_head: bool = True


@U.dataclass_cli
def main(args: ScriptArgs):
    common.run(args, __file__)


if __name__ == "__main__":
    typer.run(main)
