"""WorkerScriptArgs: our custom arguments passed from CLI to torchrun worker.

All flags use the ``--script-`` prefix to avoid collision with Megatron's own
argparse namespace.  Adding a new field here is the *only* change needed —
serialization (CLI → worker) and deserialization (worker argparse) are automatic.
"""

from __future__ import annotations

import dataclasses
from argparse import ArgumentParser
from pathlib import Path

from miles.utils.argparse_utils import DataclassArgparseBridge
from miles_plugins.models.deepseek_v4.arguments import add_dsv4_arguments
from miles_plugins.models.glm5.arguments import add_dsa_arguments


@dataclasses.dataclass(frozen=True)
class WorkerScriptArgs:
    hf_checkpoint: Path
    token_ids_file: Path
    role: str = "actor"
    ref_load: Path | None = None
    run_backward: bool = False
    source_patcher_config: Path | None = None
    routing_replay_dump_path: Path | None = None
    routing_replay_load_path: Path | None = None
    top_k: int = 0
    logprob_output: Path | None = None


WORKER_SCRIPT_ARGS_BRIDGE: DataclassArgparseBridge[WorkerScriptArgs] = DataclassArgparseBridge(
    WorkerScriptArgs,
    prefix="script",
    group_title="run_megatron script args",
)


def register_worker_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Register script and model-plugin arguments before Megatron normalization."""
    WORKER_SCRIPT_ARGS_BRIDGE.register_on_parser(parser)
    add_dsv4_arguments(parser)
    add_dsa_arguments(parser)
    return parser
