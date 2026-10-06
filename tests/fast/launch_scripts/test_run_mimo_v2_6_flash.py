import pytest

from tests.fast.launch_scripts.py_harness import freeze_environment, import_launch_script, install_command_recorder
from tests.fast.launch_scripts.sh_harness import REPO_ROOT


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
