import copy
import re
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from miles.utils.audit_utils.process_identity import ProcessIdentity
from miles.utils.env_report.redaction import redact_arg, redact_env_vars, redact_server_info
from miles.utils.test_utils.snapshot import assert_matches_snapshot, dump_snapshot, snapshot_values


@dataclass
class _SnapshotState:
    directory: Path
    run_uuid: str
    source: str
    replacements: dict[str, dict[str, Any]]
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


_snapshot_state: _SnapshotState | None = None


def configure_config_snapshots(*, args: Namespace, source: ProcessIdentity) -> None:
    global _snapshot_state
    _snapshot_state = None
    if args.config_snapshot_dir is None:
        return

    if not args.config_snapshot_name:
        raise ValueError("--config-snapshot-name is required when configuration snapshots are enabled")
    name = Path(args.config_snapshot_name)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("--config-snapshot-name must be a relative path without '..'")
    replacements = (
        {}
        if args.config_snapshot_normalize is None
        else TypeAdapter(dict[str, dict[str, Any]]).validate_json(Path(args.config_snapshot_normalize).read_text())
    )
    _snapshot_state = _SnapshotState(
        directory=Path(args.config_snapshot_dir) / name / args.deploy_component / (args.deploy_instance_id or "default"),
        run_uuid=args.run_uuid,
        source=source.to_name(),
        replacements=replacements,
    )


def check_config_snapshot(*, boundary: str, config: Any) -> None:
    if (state := _snapshot_state) is None:
        return
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", state.source) or not re.fullmatch(r"[a-z_]+", boundary):
        raise ValueError(f"Invalid snapshot identity: {state.source!r}/{boundary!r}")

    sequence = f"{state.source}/{boundary}"
    key = f"{sequence}-{state.counts[sequence]:04d}.yaml"
    state.counts[sequence] += 1
    replacements = state.replacements.get(key, {})
    value = _replace_run_uuid(
        {
            "boundary": boundary,
            "config": _redact(snapshot_values(config)),
        },
        run_uuid=state.run_uuid,
    )
    actual = dump_snapshot({
        "normalization": {"run_uuid": "$RUN_UUID", "replacements": replacements},
        "snapshot": _normalize(value, replacements=replacements),
    })
    assert_matches_snapshot(snapshot=state.directory / key, actual=actual, subject=f"runtime configuration {key}")


def _normalize(value: Any, *, replacements: dict[str, Any]) -> Any:
    result = copy.deepcopy(value)
    for pointer, replacement in replacements.items():
        if not pointer.startswith("/"):
            raise ValueError(f"Expected an absolute JSON pointer, got {pointer!r}")
        parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]
        target = result
        for part in parts[:-1]:
            target = target[int(part)] if isinstance(target, list) else target[part]
        key = int(parts[-1]) if isinstance(target, list) else parts[-1]
        target[key]
        target[key] = replacement
    return result


def _replace_run_uuid(value: Any, *, run_uuid: str) -> Any:
    if isinstance(value, str):
        return value.replace(run_uuid, "$RUN_UUID") if run_uuid else value
    if isinstance(value, dict):
        return {key: _replace_run_uuid(item, run_uuid=run_uuid) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_run_uuid(item, run_uuid=run_uuid) for item in value]
    return value


def _redact(value: Any) -> Any:
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if not isinstance(value, dict):
        return value

    return {
        name: _redact(
            redact_env_vars(item)
            if name in {"env", "env_vars", "train_env_vars"} and isinstance(item, dict)
            else redact_arg(name, item)
        )
        for name, item in redact_server_info(value).items()
    }
