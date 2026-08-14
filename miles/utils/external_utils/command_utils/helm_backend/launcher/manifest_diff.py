from __future__ import annotations

from typing import Any

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest, ManifestObject
from miles.utils.pydantic_utils import FrozenStrictBaseModel

_SCALABLE_KINDS = frozenset({"LeaderWorkerSet"})


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


def diff_manifests(*, before: Manifest, after: Manifest) -> ManifestDiff:
    old = before.by_identity
    new = after.by_identity
    shared = sorted(set(old) & set(new))
    return ManifestDiff(
        changed=[
            f"{identity}: {path}"
            for identity in shared
            for path in _disallowed_differences(old[identity], new[identity])
        ],
        added=sorted(str(identity) for identity in set(new) - set(old)),
        removed=sorted(str(identity) for identity in set(old) - set(new)),
        scaled=[
            f"{identity}: replicas {old[identity].replicas} -> {new[identity].replicas}"
            for identity in shared
            if old[identity].replicas != new[identity].replicas
        ],
    )


def _disallowed_differences(old: ManifestObject, new: ManifestObject) -> list[str]:
    return _differing_paths(old.body, new.body, (), allowed=_allowed_of_kind(old.kind))


def _allowed_of_kind(kind: str) -> tuple[tuple[str, ...], ...]:
    return (("spec", "replicas"),) if kind in _SCALABLE_KINDS else ()


def _differing_paths(old: Any, new: Any, path: tuple[str, ...], *, allowed: tuple[tuple[str, ...], ...]) -> list[str]:
    if path in allowed or old == new:
        return []

    if isinstance(old, dict) and isinstance(new, dict):
        differences = []
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                differences.append(".".join((*path, key)))
            else:
                differences += _differing_paths(old[key], new[key], (*path, key), allowed=allowed)
        return differences

    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [
            difference
            for index, (old_item, new_item) in enumerate(zip(old, new, strict=True))
            for difference in _differing_paths(old_item, new_item, (*path, f"[{index}]"), allowed=allowed)
        ]

    return [".".join(path) or "(root)"]
