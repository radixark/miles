"""WorkerScriptArgs: our custom arguments passed from CLI to torchrun worker.

All flags use the ``--script-`` prefix to avoid collision with Megatron's own
argparse namespace.  Adding a new field here is the *only* change needed —
serialization (CLI → worker) and deserialization (worker argparse) are automatic.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

_PREFIX: str = "script"


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
    indexer_replay_dump_path: str | None = None
    indexer_replay_load_path: str | None = None

    # -- serialization: dataclass → CLI arg string ----------------------------

    def to_cli_args(self) -> str:
        """Serialize to a CLI argument string like ``--script-role actor --script-run-backward``."""
        parts: list[str] = []
        for field in dataclasses.fields(self):
            value: object = getattr(self, field.name)
            flag: str = f"--{_PREFIX}-{field.name.replace('_', '-')}"

            if field.type == "bool":
                if value:
                    parts.append(flag)
            elif value is not None:
                parts.append(f"{flag} {value}")

        return " ".join(parts)

    # -- deserialization: argparse namespace → dataclass ----------------------

    @classmethod
    def register_argparse(cls, parser: argparse.ArgumentParser) -> None:
        """Add ``--script-*`` arguments to an argparse parser."""
        group: argparse._ArgumentGroup = parser.add_argument_group(
            "run_megatron script args"
        )
        for field in dataclasses.fields(cls):
            flag: str = f"--{_PREFIX}-{field.name.replace('_', '-')}"
            dest: str = f"{_PREFIX}_{field.name}"

            if field.type == "bool":
                group.add_argument(flag, dest=dest, action="store_true", default=False)
            elif field.type == "str | None":
                group.add_argument(flag, dest=dest, type=str, default=None)
            elif field.type == "str":
                has_default: bool = field.default is not dataclasses.MISSING
                if has_default:
                    group.add_argument(flag, dest=dest, type=str, default=field.default)
                else:
                    group.add_argument(flag, dest=dest, type=str, required=True)
            else:
                raise TypeError(f"Unsupported field type {field.type!r} for {field.name}")

    @classmethod
    def from_argparse(cls, namespace: argparse.Namespace) -> WorkerScriptArgs:
        """Extract ``script_*`` attributes from a Namespace into a WorkerScriptArgs."""
        kwargs: dict[str, object] = {}
        for field in dataclasses.fields(cls):
            kwargs[field.name] = getattr(namespace, f"{_PREFIX}_{field.name}")
        return cls(**kwargs)  # type: ignore[arg-type]
