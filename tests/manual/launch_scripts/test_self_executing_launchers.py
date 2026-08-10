from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tests.fast.launch_scripts.py_harness import (
    format_recording,
    freeze_environment,
    import_launch_script,
    install_shell_recorder,
    iter_self_executing_launchers,
)
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot

_SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots" / "launch_scripts" / "self_executing"

_P2P = "examples/infra_features/p2p_weight_transfer/run.py"
_FORMAL_MATH = "examples/experimental/formal_math/single_round/run_minimal.py"


@dataclass(frozen=True)
class LauncherCase:
    rel: str
    name: str
    entrypoint: str | None = None
    kwargs: dict[str, object] = field(default_factory=dict)


_P2P_PROFILES = (
    "GLM-4.5-Air",
    "GLM-4.7-Flash",
    "GLM-5",
    "GLM-5_20layer",
    "GLM-5_4layer",
    "Kimi-K2-Instruct",
    "Qwen3-235B-A22B-Instruct-2507",
    "Qwen3-30B-A3B",
    "Qwen3-4B",
)

_CASES = [
    LauncherCase(
        rel=_P2P,
        name=f"run/{profile}/{mode}",
        entrypoint="cmd_run",
        kwargs={"model_name": profile, "mode": mode, "node_rank": 0, "head_ip": "10.0.0.1"},
    )
    for profile in _P2P_PROFILES
    for mode in ("p2p", "broadcast")
] + [LauncherCase(rel=_FORMAL_MATH, name="import")]

_ENTRYPOINTS_THE_HARNESS_CANNOT_SANDBOX = {(_P2P, "cmd_prepare")}


@pytest.fixture(params=_CASES, ids=[f"{case.rel}::{case.name}" for case in _CASES])
def recorded(request, monkeypatch, tmp_path):
    case = request.param
    freeze_environment(monkeypatch)
    monkeypatch.setenv("SKIP_VALIDATION", "1")
    recording = install_shell_recorder(monkeypatch, sandbox=tmp_path)
    module = import_launch_script(REPO_ROOT / case.rel)
    if case.entrypoint is not None:
        getattr(module, case.entrypoint)(**case.kwargs)
    return case, recording, tmp_path


class TestEverySelfExecutingLauncher:
    def test_commands_match_snapshot(self, recorded):
        """These launchers build their whole command line by hand, so only a snapshot pins it."""
        case, recording, sandbox = recorded
        snapshot = _SNAPSHOT_DIR / case.rel / f"{case.name}.txt"

        assert_matches_snapshot(snapshot, format_recording(recording, sandbox=sandbox), f"{case.rel}::{case.name}")

    def test_reruns_produce_identical_recordings(self, recorded, monkeypatch, tmp_path):
        """These launchers embed their own pid, so a snapshot is only stable if the harness freezes it."""
        case, recording, _ = recorded
        freeze_environment(monkeypatch)
        monkeypatch.setenv("SKIP_VALIDATION", "1")
        again = install_shell_recorder(monkeypatch, sandbox=tmp_path)
        module = import_launch_script(REPO_ROOT / case.rel)
        if case.entrypoint is not None:
            getattr(module, case.entrypoint)(**case.kwargs)

        assert again.commands == recording.commands

    def test_the_launcher_submits_a_ray_job(self, recorded):
        """A launcher that stops reaching `ray job submit` is broken, whatever else it records."""
        _, recording, _ = recorded

        assert [command for command in recording.commands if "ray job submit" in command]


class TestDiscovery:
    def test_every_self_executing_launcher_has_at_least_one_case(self):
        """Discovery is by behaviour, not by path, so a new hand-rolled launcher shows up here."""
        discovered = {path.relative_to(REPO_ROOT).as_posix() for path in iter_self_executing_launchers()}

        assert discovered == {case.rel for case in _CASES}

    def test_the_uncovered_entrypoint_is_named_and_still_uncoverable(self):
        """cmd_prepare rewrites a checkout under a hardcoded /root/models, which no fixture can redirect."""
        module = import_launch_script(REPO_ROOT / _P2P)

        assert {(_P2P, name) for name in ("cmd_run", "cmd_prepare")} - {
            (case.rel, case.entrypoint) for case in _CASES
        } == _ENTRYPOINTS_THE_HARNESS_CANNOT_SANDBOX
        assert '"/root/models"' in Path(module.__file__).read_text()
