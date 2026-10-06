from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from tests.fast.launch_scripts.py_harness import freeze_environment, import_launch_script, install_command_recorder
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

from miles.utils.external_utils.command_utils import exclusive_path_lock
from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend


@pytest.mark.parametrize("mode", ["rl", "sft"])
@pytest.mark.parametrize("checkpoint_exists", [False, True])
def test_prepare_and_execute_with_the_configured_backend(monkeypatch, tmp_path, mode, checkpoint_exists):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    launcher = import_launch_script(REPO_ROOT / "scripts/run_mimo_v2_6_flash.py")
    args = launcher.ScriptArgs(mode=mode, model_dir=str(tmp_path / "models"), data_dir=str(tmp_path / "data"))
    if checkpoint_exists:
        checkpoint = tmp_path / "models" / args.model_name
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors.index.json").write_text("{}")

    launcher.prepare(args)
    launcher.execute(args)

    commands = recording.commands
    assert any("hf download XiaomiMiMo/MiMo-V2.6-Flash-RL" in cmd for cmd in commands) is not checkpoint_exists
    assert any("convert_mimo_v2_to_bf16.py" in cmd for cmd in commands) is not checkpoint_exists
    assert any("hf download --repo-type dataset zhuzilin/dapo-math-17k" in cmd for cmd in commands) is (mode == "rl")
    train_commands = [cmd for cmd in commands if "ray job submit" in cmd]
    assert len(train_commands) == 1
    train_script = "train.py" if mode == "rl" else "train_async.py"
    assert f"/{train_script} " in train_commands[0]


def test_prepare_waits_for_shared_checkpoint_conversion(monkeypatch, tmp_path):
    freeze_environment(monkeypatch)
    install_command_recorder(monkeypatch)
    launcher = import_launch_script(REPO_ROOT / "scripts/run_mimo_v2_6_flash.py")
    args = launcher.ScriptArgs(mode="sft", model_dir=str(tmp_path / "models"))
    target = tmp_path / "models" / args.model_name
    started, converted = Event(), Event()
    monkeypatch.setattr(BaseCommandBackend, "exec_command_gpu", lambda *args, **kwargs: converted.set())

    def prepare():
        started.set()
        launcher.prepare(args)

    # Preparation takes a real filesystem lock even when its shell commands are
    # recorded, so the launcher snapshots also need a writable model_dir.
    with ThreadPoolExecutor(max_workers=1) as pool:
        with exclusive_path_lock(str(target)):
            future = pool.submit(prepare)
            assert started.wait(timeout=5)
            attempted_conversion = converted.wait(timeout=0.2)
            target.mkdir(parents=True)
            (target / "model.safetensors.index.json").write_text("{}")
        future.result(timeout=5)

    assert not attempted_conversion
    assert not converted.is_set()
