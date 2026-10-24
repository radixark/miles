import os
import re
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from miles.utils.audit_utils.config_snapshot.models import (
    ConfigSnapshotContext,
    ConfigSnapshotPoint,
    ConfigSnapshotRecord,
)
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.audit_utils.process_identity import ProcessIdentity
from miles.utils.env_report.redaction import redact_arg, redact_env_vars, redact_server_info
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR, snapshot_values


@dataclass
class _SnapshotState:
    storage: ConfigSnapshotStorage
    context: ConfigSnapshotContext
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class ConfigSnapshotDumper:
    _state: ClassVar[_SnapshotState | None] = None

    @classmethod
    def configure(cls, *, args: Namespace, source: ProcessIdentity) -> None:
        cls._state = None
        if args.ci_disable_config_snapshot or not args.ci_test:
            return

        if not args.config_snapshot_name:
            raise ValueError("Snapshot name is required")
        if not (record_directory := os.environ.get(SNAPSHOT_RECORD_DIR_ENV_VAR)):
            raise ValueError(f"{SNAPSHOT_RECORD_DIR_ENV_VAR} is required for configuration snapshots")
        cls._state = _SnapshotState(
            storage=ConfigSnapshotStorage(directory=Path(record_directory)),
            context=ConfigSnapshotContext(
                name=args.config_snapshot_name,
                deploy_component=args.deploy_component,
                deploy_instance_id=args.deploy_instance_id or "default",
                source=source,
                run_uuid=args.run_uuid,
            ),
        )

    @classmethod
    def dump(cls, *, stage: str, config: Any) -> None:
        if (state := cls._state) is None:
            return
        if not re.fullmatch(r"[a-z_]+", stage):
            raise ValueError(f"Invalid snapshot stage: {stage!r}")

        index = state.counts[stage]
        state.counts[stage] += 1
        record = ConfigSnapshotRecord(
            context=state.context,
            point=ConfigSnapshotPoint(stage=stage, index=index),
            config=_redact(snapshot_values(config)),
        )
        state.storage.write(record)


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
