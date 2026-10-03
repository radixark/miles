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
SNAPSHOT_RECORD_DIR_ENV_VAR = "MILES_SNAPSHOT_RECORD_DIR"


def dump_snapshot(value: Any) -> str:
    return yaml.safe_dump(snapshot_values(value), sort_keys=True, allow_unicode=True, width=120)


def snapshot_values(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return {"$enum": _qualified_name(type(value)), "name": value.name}
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, argparse.Namespace):
        return snapshot_values(vars(value))
    if isinstance(value, argparse._ActionsContainer):
        return {"$class": _qualified_name(type(value)), "state": snapshot_values(_actions_container_state(value))}
    if isinstance(value, argparse.Action):
        return snapshot_values(_action_state(value))
    if isinstance(value, BaseModel):
        return snapshot_values(
            {
                **{name: value.__getattribute__(name) for name in type(value).model_fields},
                **(value.model_extra or {}),
            }
        )
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
    if isinstance(value, (set, frozenset)):
        return {"$set": sorted((snapshot_values(item) for item in value), key=repr)}
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


def _action_state(action: argparse.Action) -> dict[str, Any]:
    state = {name: item for name, item in vars(action).items() if name != "container"}
    if isinstance(choices := state["choices"], (list, tuple)):
        state["choices"] = frozenset(choices)
    if type(action) is not argparse._StoreAction:
        state["action"] = type(action)
    return {
        name: item
        for name, item in state.items()
        if name in _ALWAYS_SNAPSHOTTED_ACTION_FIELDS or item != _ACTION_FIELD_DEFAULTS.get(name, _NO_DEFAULT)
    }


_ALWAYS_SNAPSHOTTED_ACTION_FIELDS = frozenset({"dest", "option_strings", "default", "type", "help"})
_ACTION_FIELD_DEFAULTS = {
    "nargs": None,
    "const": None,
    "choices": None,
    "required": False,
    "metavar": None,
    "deprecated": False,
}
_NO_DEFAULT = object()


def _actions_container_state(container: argparse._ActionsContainer) -> dict[str, Any]:
    if isinstance(container, argparse._MutuallyExclusiveGroup):
        return {"required": container.required, "dests": [action.dest for action in container._group_actions]}
    if isinstance(container, argparse._ArgumentGroup):
        return {
            "title": container.title,
            "description": container.description,
            "dests": [action.dest for action in container._group_actions],
        }
    action_group_titles = {
        id(action): group.title for group in container._action_groups for action in group._group_actions
    }
    return {
        **{name: item for name, item in vars(container).items() if name not in _DERIVED_PARSER_STATE},
        "_actions": [
            {**_action_state(action), "group": action_group_titles[id(action)]} for action in container._actions
        ],
        "_action_groups": [
            {"title": group.title, "description": group.description} for group in container._action_groups
        ],
    }


_DERIVED_PARSER_STATE = frozenset({"_registries", "_option_string_actions", "_optionals", "_positionals"})


def _qualified_name(value: Any) -> str:
    return f"{value.__module__}.{value.__qualname__}"
