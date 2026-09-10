import pytest

from tests.fast.launch_scripts.py_harness import (
    call_entrypoint,
    format_recording,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot


@pytest.mark.parametrize("model_name", ["qwen3-30B-A3B", "glm5.2"])
def test_pressure_launcher(monkeypatch, tmp_path, model_name):
    freeze_environment(monkeypatch)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    recording = install_command_recorder(monkeypatch)
    relative = "examples/multi_lora/run_pressure.py"
    module = import_launch_script(REPO_ROOT / relative)
    call_entrypoint(module, "main", {"model_name": model_name}, sandbox=tmp_path)
    snapshot = REPO_ROOT / "tests/snapshots/launch_scripts/py" / relative / f"{model_name}.txt"
    assert_matches_snapshot(snapshot, format_recording(recording, sandbox=tmp_path), relative)
