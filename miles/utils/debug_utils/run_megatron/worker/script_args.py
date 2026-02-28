"""WorkerScriptArgs: our custom arguments passed from CLI to torchrun worker.

All flags use the ``--script-`` prefix to avoid collision with Megatron's own
argparse namespace.  Adding a new field here is the *only* change needed —
serialization (CLI → worker) and deserialization (worker argparse) are automatic.
"""

from __future__ import annotations

import dataclasses

from miles.utils.argparse_utils import DataclassArgparseBridge


@dataclasses.dataclass(frozen=True)
class WorkerScriptArgs:
    hf_checkpoint: str
    token_ids_file: str
    role: str = "actor"
    ref_load: str | None = None
    run_backward: bool = False
    source_patcher_config: str | None = None
    routing_replay_dump_path: str | None = None
    routing_replay_load_path: str | None = None


WORKER_SCRIPT_ARGS_BRIDGE: DataclassArgparseBridge[WorkerScriptArgs] = DataclassArgparseBridge(
    WorkerScriptArgs,
    prefix="script",
    group_title="run_megatron script args",
)
