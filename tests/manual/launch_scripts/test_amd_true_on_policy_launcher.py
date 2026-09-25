"""Pin the AMD Qwen3 precision contract and its preparation/launch commands."""

import pytest

from tests.fast.launch_scripts.py_harness import (
    format_recording,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot

_LAUNCHER = "examples/infra_features/true_on_policy/run_simple_amd_triton.py"


@pytest.mark.parametrize("entrypoint", ["prepare", "execute"])
def test_amd_true_on_policy_commands(entrypoint, monkeypatch, tmp_path):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / _LAUNCHER)
    getattr(module, entrypoint)()
    snapshot = REPO_ROOT / "tests/snapshots/launch_scripts/py" / _LAUNCHER / f"{entrypoint}.txt"
    assert_matches_snapshot(snapshot, format_recording(recording, sandbox=tmp_path), f"{_LAUNCHER}::{entrypoint}")
