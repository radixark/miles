import json
from collections import defaultdict
from copy import deepcopy

from pydantic import Field, JsonValue

from miles.utils.audit_utils.config_snapshot.models import (
    ConfigSnapshotCase,
    ConfigSnapshotDelta,
    ConfigSnapshotProcess,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.snapshot import dump_snapshot


class ConfigSnapshotBases(FrozenStrictBaseModel):
    templates: dict[str, JsonValue]


class _CompactDelta(ConfigSnapshotDelta):
    set: dict[str, JsonValue] = Field(default_factory=dict)
    remove: list[str] = Field(default_factory=list)


class _CompactProcess(FrozenStrictBaseModel):
    ranks: list[int] = Field(default_factory=list)
    template: str
    overrides: _CompactDelta = Field(default_factory=_CompactDelta)
    diffs: dict[str, _CompactDelta] = Field(default_factory=dict)


class CompactConfigSnapshotCase(FrozenStrictBaseModel):
    shared_overrides: dict[str, _CompactDelta] = Field(default_factory=dict)
    processes: dict[str, _CompactProcess]


def compact_config_snapshot(*, case: ConfigSnapshotCase, bases: ConfigSnapshotBases) -> CompactConfigSnapshotCase:
    if not bases.templates:
        raise ValueError("Configuration snapshot templates must not be empty")

    processes: dict[str, _CompactProcess] = {}
    for name, process in case.processes.items():
        candidates = {
            name: make_snapshot_delta(base=base, actual=process.base) for name, base in sorted(bases.templates.items())
        }
        template = min(candidates, key=lambda name: len(dump_snapshot(candidates[name])))
        processes[name] = _CompactProcess(
            ranks=process.ranks,
            template=template,
            overrides=_CompactDelta(**candidates[template].model_dump()),
            diffs={stage: _CompactDelta(**delta.model_dump()) for stage, delta in process.diffs.items()},
        )
    shared = _share_overrides(processes)
    return CompactConfigSnapshotCase(shared_overrides=shared, processes=processes)


def expand_config_snapshot(*, case: CompactConfigSnapshotCase, bases: ConfigSnapshotBases) -> ConfigSnapshotCase:
    common = {
        name: (
            apply_snapshot_delta(base=base, overrides=case.shared_overrides[name])
            if name in case.shared_overrides
            else base
        )
        for name, base in bases.templates.items()
    }
    return ConfigSnapshotCase(
        processes={
            name: ConfigSnapshotProcess(
                ranks=process.ranks,
                base=apply_snapshot_delta(base=common[process.template], overrides=process.overrides),
                diffs={stage: ConfigSnapshotDelta(**delta.model_dump()) for stage, delta in process.diffs.items()},
            )
            for name, process in case.processes.items()
        }
    )


def make_snapshot_delta(*, base: JsonValue, actual: JsonValue) -> ConfigSnapshotDelta:
    changes: dict[str, JsonValue] = {}
    removed: list[str] = []
    _collect_changes(base=base, actual=actual, path="", changes=changes, removed=removed)
    return ConfigSnapshotDelta(set=changes, remove=removed)


def apply_snapshot_delta(*, base: JsonValue, overrides: ConfigSnapshotDelta) -> JsonValue:
    result = deepcopy(base)
    for path in overrides.remove:
        parent, key = _parent(value=result, path=path)
        del parent[key]
    for path, value in overrides.set.items():
        if path == "":
            result = deepcopy(value)
        else:
            parent, key = _parent(value=result, path=path)
            parent[key] = deepcopy(value)
    return result


def _share_overrides(
    processes: dict[str, _CompactProcess],
) -> dict[str, _CompactDelta]:
    groups: dict[str, list[str]] = defaultdict(list)
    for name, process in processes.items():
        groups[process.template].append(name)

    shared: dict[str, _CompactDelta] = {}
    for template, names in sorted(groups.items()):
        if len(names) < 2:
            continue
        first = processes[names[0]].overrides
        assigned = {
            path: value
            for path, value in first.set.items()
            if all(
                path in processes[name].overrides.set
                and json.dumps(value, sort_keys=True)
                == json.dumps(processes[name].overrides.set[path], sort_keys=True)
                for name in names[1:]
            )
        }
        removed = sorted(set(first.remove).intersection(*(processes[name].overrides.remove for name in names[1:])))
        if not assigned and not removed:
            continue
        common = _CompactDelta(set=assigned, remove=removed)
        reduced = {}
        for name in names:
            process = processes[name]
            reduced[name] = process.model_copy(
                update={
                    "overrides": _CompactDelta(
                        set={path: value for path, value in process.overrides.set.items() if path not in assigned},
                        remove=[path for path in process.overrides.remove if path not in removed],
                    )
                }
            )
        original_size = len(dump_snapshot({"processes": {name: processes[name] for name in names}}))
        reduced_size = len(dump_snapshot({"shared_overrides": {template: common}, "processes": reduced}))
        if reduced_size < original_size:
            shared[template] = common
            processes.update(reduced)
    return shared


def _collect_changes(
    *,
    base: JsonValue,
    actual: JsonValue,
    path: str,
    changes: dict[str, JsonValue],
    removed: list[str],
) -> None:
    if isinstance(base, dict) and isinstance(actual, dict):
        for key in sorted(base.keys() | actual.keys()):
            child = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in actual:
                removed.append(child)
            elif key not in base:
                changes[child] = actual[key]
            else:
                _collect_changes(
                    base=base[key],
                    actual=actual[key],
                    path=child,
                    changes=changes,
                    removed=removed,
                )
    elif json.dumps(base, sort_keys=True) != json.dumps(actual, sort_keys=True):
        changes[path] = actual


def _parent(*, value: JsonValue, path: str) -> tuple[dict[str, JsonValue], str]:
    if not path.startswith("/"):
        raise ValueError(f"Expected a non-root JSON Pointer: {path!r}")
    keys = [key.replace("~1", "/").replace("~0", "~") for key in path[1:].split("/")]
    for key in keys[:-1]:
        if not isinstance(value, dict):
            raise ValueError(f"Snapshot override traverses a non-mapping: {path!r}")
        value = value[key]
    if not isinstance(value, dict):
        raise ValueError(f"Snapshot override parent is not a mapping: {path!r}")
    return value, keys[-1]
