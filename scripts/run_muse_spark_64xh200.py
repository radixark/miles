"""Train full Muse Spark with GRPO on 64 H200 GPUs (8 nodes of 8). This file sets the cluster size, parallelism and
engine sizing; the recipe and its other options are in scripts/muse_spark/common.py.

Example:
    MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_muse_spark_64xh200.py
    python scripts/run_muse_spark_64xh200.py --partial  # one-node smoke test with the 4-layer checkpoint
"""

from dataclasses import dataclass

import typer
from muse_spark import common

import miles.utils.external_utils.command_utils.legacy as U


@dataclass(kw_only=True)
class ScriptArgs(common.ScriptArgs):
    num_nodes: int = 8
    num_gpus_per_node: int = 8
    # TP8/PP8 with 7 layers on each edge stage. TP4/PP8/CP2 (7) and TP4/PP4/CP4/EP16 (15) also fit.
    tensor_model_parallel_size: int = 8
    pipeline_model_parallel_size: int = 8
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 8
    decoder_first_last_pipeline_num_layers: int = 7
    # Four engines of two nodes each.
    rollout_num_gpus_per_engine: int = 16
    max_tokens_per_gpu: int = 4096
    sglang_mem_fraction_static: float = 0.8
    sglang_max_running_requests: int = 64
    sglang_cuda_graph_bs_decode: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    pin_rollout_manager_to_head: bool = False


@U.dataclass_cli
def main(args: ScriptArgs):
    common.run(args, __file__)


if __name__ == "__main__":
    typer.run(main)
