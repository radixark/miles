import argparse
import difflib
import enum
import functools
import inspect
import os
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

SNAPSHOT_UPDATE_ENV_VAR = "MILES_UPDATE_SNAPSHOTS"


def dump_snapshot(value: Any) -> str:
    return yaml.safe_dump(snapshot_values(value), sort_keys=True, allow_unicode=True, width=120)


def snapshot_values(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return {"$enum": _qualified_name(type(value)), "name": value.name}
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, argparse.Namespace):
        return snapshot_values(vars(value))
    if isinstance(value, (argparse._ActionsContainer, argparse.Action)):
        return {"$class": _qualified_name(type(value)), "state": snapshot_values(vars(value))}
    if isinstance(value, BaseModel):
        return snapshot_values({
            **{name: value.__getattribute__(name) for name in type(value).model_fields},
            **(value.model_extra or {}),
        })
    if is_dataclass(value) and not isinstance(value, type):
        return snapshot_values({field.name: value.__getattribute__(field.name) for field in fields(value)})
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Snapshots require string mapping keys")
        return {key: snapshot_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot_values(item) for item in value]
    if isinstance(value, tuple):
        return {"$tuple": [snapshot_values(item) for item in value]}
    if isinstance(value, Path):
        return {"$path": str(value)}
    if isinstance(value, (date, datetime)):
        return {"$date": value.isoformat()}
    if isinstance(value, range):
        return {"$range": [value.start, value.stop, value.step]}
    if isinstance(value, re.Pattern):
        return {"$regex": value.pattern, "flags": value.flags}
    if isinstance(value, functools.partial):
        return {
            "$partial": snapshot_values(value.func),
            "args": snapshot_values(value.args),
            "keywords": snapshot_values(value.keywords),
        }
    if isinstance(value, type) or inspect.isfunction(value) or inspect.isbuiltin(value):
        return {"$callable": _qualified_name(value)}
    if _qualified_name(type(value)) in {"torch.dtype", "torch.device"}:
        return {"$torch": str(value)}
    raise TypeError(f"Unsupported snapshot value type: {_qualified_name(type(value))}")


def assert_scenario_snapshots(*, snapshots: dict[str, str], bases: dict[str, str], directory: Path) -> None:
    failures = []
    for name, snapshot in snapshots.items():
        suffix = ".yaml"
        if base := bases.get(name):
            snapshot = "".join(
                difflib.unified_diff(
                    snapshots[base].splitlines(keepends=True),
                    snapshot.splitlines(keepends=True),
                    fromfile=base,
                    tofile=name,
                    n=0,
                )
            )
            suffix = ".diff"
        try:
            assert_matches_snapshot(
                snapshot=directory / f"{name}{suffix}", actual=snapshot, subject=f"snapshot scenario {name}"
            )
        except AssertionError as error:
            failures.append(str(error))
    if failures:
        raise AssertionError("\n\n".join(failures))


def assert_matches_snapshot(snapshot: Path, actual: str, subject: str, *, update: bool | None = None) -> None:
    if update if update is not None else bool(os.environ.get(SNAPSHOT_UPDATE_ENV_VAR)):
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(actual)
        return

    exists = snapshot.exists()
    expected = snapshot.read_text() if exists else ""
    if not exists or actual != expected:
        diff = difflib.unified_diff(
            expected.splitlines(), actual.splitlines(), fromfile=f"{snapshot}", tofile="actual", lineterm=""
        )
        raise AssertionError(
            f"{subject} does not match its snapshot or baseline is missing: {snapshot}\n"
            + "\n".join(diff)
            + f"\n--- BEGIN ACTUAL {snapshot.name} ---\n{actual}--- END ACTUAL ---\n"
            + f"Copy the content above to {snapshot}, or regenerate with {SNAPSHOT_UPDATE_ENV_VAR}=1."
        )


def _qualified_name(value: Any) -> str:
    return f"{value.__module__}.{value.__qualname__}"
