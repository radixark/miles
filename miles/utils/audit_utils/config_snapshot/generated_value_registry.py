import json
import os
from typing import Literal

from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotGeneratedValue
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR

GENERATED_VALUES_ENV_VAR = "MILES_SNAPSHOT_GENERATED_VALUES"


class GeneratedValueRegistry:
    def __init__(self) -> None:
        self._values: list[ConfigSnapshotGeneratedValue] = [
            ConfigSnapshotGeneratedValue.model_validate(item)
            for item in json.loads(
                os.environ.get(GENERATED_VALUES_ENV_VAR, "[]") if os.environ.get(SNAPSHOT_RECORD_DIR_ENV_VAR) else "[]"
            )
        ]

    def register(
        self,
        *,
        kind: Literal["run_id", "temporary_directory", "ci_commit_name", "host", "port", "external_host", "wandb_key"],
        value: str,
        name: str | None = None,
    ) -> None:
        if not os.environ.get(SNAPSHOT_RECORD_DIR_ENV_VAR):
            return

        if name is None:
            if any(item.kind == kind and item.value == value for item in self._values):
                return
            name = f"{sum(item.kind == kind for item in self._values):04d}"
        entry = ConfigSnapshotGeneratedValue(kind=kind, name=name, value=value)
        for item in self._values:
            if (item.kind, item.name) == (kind, name):
                if item != entry:
                    raise ValueError(f"Conflicting generated snapshot value: {kind}/{name}")
                return
        self._values.append(entry)

    def read(self) -> list[ConfigSnapshotGeneratedValue]:
        return list(self._values)

    def to_env(self) -> dict[str, str]:
        if not os.environ.get(SNAPSHOT_RECORD_DIR_ENV_VAR) or not self._values:
            return {}
        return {GENERATED_VALUES_ENV_VAR: json.dumps([item.model_dump() for item in self._values])}


generated_value_registry = GeneratedValueRegistry()
