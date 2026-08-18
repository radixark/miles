from __future__ import annotations

from typing import Any, Literal

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import (
    Manifest,
    ManifestObject,
    ManifestObjectKey,
    ObjectIdentity,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel

_SCALABLE_KINDS = frozenset({"LeaderWorkerSet"})
_REPLICAS_PATH = ("spec", "replicas")


class ManifestChange(FrozenStrictBaseModel):
    identity: ObjectIdentity
    path: tuple[str, ...]
    allowed_by: Literal["scaling", "whitelist"] | None
    description: str


class ManifestDiffs(FrozenStrictBaseModel):
    changes: list[ManifestChange] = []
    additions: list[ObjectIdentity] = []
    removals: list[ObjectIdentity] = []

    @property
    def disallowed_changed(self) -> list[str]:
        return [change.description for change in self.changes if change.allowed_by is None]

    @property
    def allowed_changed(self) -> list[str]:
        return [change.description for change in self.changes if change.allowed_by is not None]

    @property
    def is_allowed(self) -> bool:
        return not (self.disallowed_changed or self.additions or self.removals)

    def summarize_allowed_changes(self) -> str:
        return "\n".join(f"  {entry}" for entry in self.allowed_changed) or "  (nothing to change)"

    def describe(self) -> str:
        lines = []
        for label, entries in (
            ("changed", self.disallowed_changed),
            ("added", [str(identity) for identity in self.additions]),
            ("removed", [str(identity) for identity in self.removals]),
        ):
            lines += [f"  {label}: {entry}" for entry in entries]
        return "\n".join(lines) or "  (no difference)"


def diff_manifests(
    *, before: Manifest, after: Manifest, allow_diff_object_keys: frozenset[ManifestObjectKey] = frozenset()
) -> ManifestDiffs:
    old = before.by_identity
    new = after.by_identity
    shared = sorted(set(old) & set(new))

    return ManifestDiffs(
        changes=[
            _compute_change(
                old[identity],
                new[identity],
                identity=identity,
                path=path,
                allow_diff_object_keys=allow_diff_object_keys,
            )
            for identity in shared
            for path in _differing_paths(old[identity].body, new[identity].body, ())
        ],
        additions=sorted(set(new) - set(old), key=str),
        removals=sorted(set(old) - set(new), key=str),
    )


def _compute_change(
    old: ManifestObject,
    new: ManifestObject,
    *,
    identity: ObjectIdentity,
    path: tuple[str, ...],
    allow_diff_object_keys: frozenset[ManifestObjectKey],
) -> ManifestChange:
    allowed_by = _compute_allowed_by(
        old, new, identity=identity, path=path, allow_diff_object_keys=allow_diff_object_keys
    )
    if allowed_by is not None and path == _REPLICAS_PATH:
        description = f"{identity}: replicas {old.replicas} -> {new.replicas}"
    else:
        description = f"{identity}: {_describe_path(path)}"
    return ManifestChange(identity=identity, path=path, allowed_by=allowed_by, description=description)


def _compute_allowed_by(
    old: ManifestObject,
    new: ManifestObject,
    *,
    identity: ObjectIdentity,
    path: tuple[str, ...],
    allow_diff_object_keys: frozenset[ManifestObjectKey],
) -> Literal["scaling", "whitelist"] | None:
    if _is_scaling(old, new, path=path):
        return "scaling"
    if identity.key in allow_diff_object_keys:
        return "whitelist"
    return None


def _is_scaling(old: ManifestObject, new: ManifestObject, *, path: tuple[str, ...]) -> bool:
    if old.kind not in _SCALABLE_KINDS or path != _REPLICAS_PATH:
        return False
    return old.replicas is not None and new.replicas is not None


def _describe_path(path: tuple[str, ...]) -> str:
    return ".".join(path) or "(root)"


def _differing_paths(old: Any, new: Any, path: tuple[str, ...]) -> list[tuple[str, ...]]:
    if old == new:
        return []

    if isinstance(old, dict) and isinstance(new, dict):
        differences = []
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                differences.append((*path, key))
            else:
                differences += _differing_paths(old[key], new[key], (*path, key))
        return differences

    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [
            difference
            for index, (old_item, new_item) in enumerate(zip(old, new, strict=True))
            for difference in _differing_paths(old_item, new_item, (*path, f"[{index}]"))
        ]

    return [path]
