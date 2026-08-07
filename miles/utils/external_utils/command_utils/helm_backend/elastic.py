from __future__ import annotations

from typing import Any

import yaml

from miles.utils.pydantic_utils import FrozenStrictBaseModel

SCALABLE_SECTIONS = ("inferenceEngines", "trainers")
SCALABLE_FIELD = "replicas"


class ValuesDiff(FrozenStrictBaseModel):
    changed: list[str] = []
    scaled: list[str] = []

    @property
    def is_allowed(self) -> bool:
        return not self.changed

    def summarize_scaling(self) -> str:
        return "\n".join(f"  {entry}" for entry in self.scaled) or "  (nothing to change)"

    def describe(self) -> str:
        return "\n".join(f"  changed: {entry}" for entry in self.changed) or "  (no difference)"


def diff_values(before: dict[str, Any] | None, after: dict[str, Any] | None) -> ValuesDiff:
    changed: list[str] = []
    scaled: list[str] = []
    _walk(before or {}, after or {}, (), changed=changed, scaled=scaled)
    return ValuesDiff(changed=sorted(changed), scaled=sorted(scaled))


def _walk(old: Any, new: Any, path: tuple[str, ...], *, changed: list[str], scaled: list[str]) -> None:
    if old == new:
        return

    if _is_scalable(path):
        scaled.append(f"{_render(path)}: {old} -> {new}")
        return

    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                changed.append(_render((*path, key)))
            else:
                _walk(old[key], new[key], (*path, key), changed=changed, scaled=scaled)
        return

    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for index, (old_item, new_item) in enumerate(zip(old, new, strict=True)):
            _walk(old_item, new_item, (*path, f"[{index}]"), changed=changed, scaled=scaled)
        return

    changed.append(_render(path) or "(root)")


def _is_scalable(path: tuple[str, ...]) -> bool:
    return (
        len(path) == 4
        and path[0] == "run"
        and path[1] in SCALABLE_SECTIONS
        and path[2].startswith("[")
        and path[3] == SCALABLE_FIELD
    )


def _render(path: tuple[str, ...]) -> str:
    return ".".join(path)


_SCALABLE_KINDS = frozenset({"LeaderWorkerSet"})
_TUNABLE_KINDS = frozenset({"ConfigMap"})


class ManifestDiff(FrozenStrictBaseModel):
    changed: list[str] = []
    added: list[str] = []
    removed: list[str] = []
    scaled: list[str] = []

    @property
    def is_allowed(self) -> bool:
        return not (self.changed or self.added or self.removed)

    def summarize_scaling(self) -> str:
        return "\n".join(f"  {entry}" for entry in self.scaled) or "  (nothing to change)"

    def describe(self) -> str:
        lines = []
        for label, entries in (("changed", self.changed), ("added", self.added), ("removed", self.removed)):
            lines += [f"  {label}: {entry}" for entry in entries]
        return "\n".join(lines) or "  (no difference)"


def diff_manifests(before: str, after: str) -> ManifestDiff:
    old = _objects_by_key(before)
    new = _objects_by_key(after)

    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    shared = sorted(set(old) & set(new))
    changed = [
        f"{key}: {path}" for key in shared for path in _disallowed_manifest_differences(key, old[key], new[key])
    ]
    scaled = [
        f"{key}: replicas {_replicas(old[key])} -> {_replicas(new[key])}"
        for key in shared
        if _replicas(old[key]) != _replicas(new[key])
    ]
    return ManifestDiff(changed=changed, added=added, removed=removed, scaled=scaled)


def manifest_of(dry_run_output: str) -> str:
    _, separator, rest = dry_run_output.partition("MANIFEST:\n")
    if not separator:
        return dry_run_output
    manifest, _, _ = rest.partition("\nNOTES:")
    return manifest


def _objects_by_key(rendered: str) -> dict[str, dict[str, Any]]:
    return {
        f"{document['kind']}/{document['metadata']['name']}": document
        for document in yaml.safe_load_all(rendered)
        if document
    }


def _allowed_for(kind: str) -> tuple[tuple[str, ...], ...]:
    allowed: list[tuple[str, ...]] = []
    if kind in _SCALABLE_KINDS:
        allowed.append(("spec", "replicas"))
    if kind in _TUNABLE_KINDS:
        allowed.append(("data",))
    return tuple(allowed)


def _disallowed_manifest_differences(key: str, old: Any, new: Any) -> list[str]:
    return _walk_manifest(old, new, (), allowed=_allowed_for(key.partition("/")[0]))


def _walk_manifest(old: Any, new: Any, path: tuple[str, ...], *, allowed: tuple[tuple[str, ...], ...]) -> list[str]:
    if path in allowed or old == new:
        return []

    if isinstance(old, dict) and isinstance(new, dict):
        differences = []
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                differences.append(".".join((*path, key)))
            else:
                differences += _walk_manifest(old[key], new[key], (*path, key), allowed=allowed)
        return differences

    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [
            difference
            for index, (old_item, new_item) in enumerate(zip(old, new, strict=True))
            for difference in _walk_manifest(old_item, new_item, (*path, f"[{index}]"), allowed=allowed)
        ]

    return [".".join(path) or "(root)"]


def _replicas(document: dict) -> Any:
    return (document.get("spec") or {}).get("replicas")
