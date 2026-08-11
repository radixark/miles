from __future__ import annotations

from pathlib import Path

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class ReleaseArgs(FrozenStrictBaseModel):
    namespace: str
    release: str


class InstallArgs(ReleaseArgs):
    rbac: bool
    lws: bool
    dry_run: bool
    values: tuple[Path, ...]
    overrides: tuple[str, ...]
    skip_preflight: bool
    timeout: int


class ExecArgs(ReleaseArgs):
    command: tuple[str, ...]

