from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from miles.utils.external_utils.command_utils.base_backend import LaunchGuard
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest


class GuardRefusedError(RuntimeError):
    pass


@dataclass
class RecordingLaunchGuard(LaunchGuard):
    installed: Manifest | None = None
    refuse: frozenset[str] = frozenset()
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def before_defuse(
        self, release: str, *, namespace: str, superseded_state_file: Path | None, state_file: Path | None
    ) -> None:
        self._record(
            "before_defuse",
            release=release,
            namespace=namespace,
            superseded_state_file=superseded_state_file,
            state_file=state_file,
        )

    def delete_uninstall_job(self, name: str, *, namespace: str, check: bool = False) -> None:
        self._record("delete_uninstall_job", name=name, namespace=namespace, check=check)

    def upgrade(
        self, *, release: str, namespace: str, chart: str | Path, values_files: list[str | Path], ci_run: bool
    ) -> None:
        self._record(
            "upgrade", release=release, namespace=namespace, chart=chart, values_files=values_files, ci_run=ci_run
        )

    def get_manifest(self, release: str, namespace: str) -> Manifest | None:
        self._record("get_manifest", release=release, namespace=namespace)
        return self.installed

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def kwargs_of(self, name: str) -> dict[str, Any]:
        (kwargs,) = [kwargs for called, kwargs in self.calls if called == name]
        return kwargs

    def _record(self, call: str, /, **kwargs: Any) -> None:
        self.calls.append((call, kwargs))
        if call in self.refuse:
            raise GuardRefusedError(f"guard refused {call}")
