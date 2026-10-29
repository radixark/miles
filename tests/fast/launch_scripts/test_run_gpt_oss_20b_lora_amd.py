from unittest.mock import ANY, call, create_autospec

import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)

from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend


@pytest.mark.parametrize("entrypoint", ["prepare", "train", "full_train"])
def test_commands_use_backend(monkeypatch, tmp_path, entrypoint):
    freeze_environment(monkeypatch, hardware="MI355X")
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/amd/run_gpt_oss_20b_lora.py")
    cpu = create_autospec(BaseCommandBackend.exec_command_cpu, side_effect=BaseCommandBackend.exec_command_cpu)
    execute_train = create_autospec(BaseCommandBackend.execute_train, side_effect=BaseCommandBackend.execute_train)
    monkeypatch.setattr(BaseCommandBackend, "exec_command_cpu", cpu)
    monkeypatch.setattr(BaseCommandBackend, "execute_train", execute_train)

    model_dir = tmp_path / "models"
    data_dir = tmp_path / "datasets"
    checkpoint = model_dir / "gpt-oss-20b-bf16"
    call_entrypoint(
        module,
        entrypoint,
        {
            "model_dir": str(model_dir),
            "data_dir": str(data_dir),
            "num_gpus_per_node": 4,
        },
        sandbox=tmp_path,
    )

    if entrypoint != "train":
        downloads = [
            f"mkdir -p {model_dir} {data_dir}",
            f"hf download lmsys/gpt-oss-20b-bf16 --local-dir {checkpoint}",
            f"hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir {data_dir}/dapo-math-17k",
        ]
        assert recording.commands[:3] == downloads
        assert cpu.call_args_list[:3] == [call(ANY, command) for command in downloads]
    else:
        assert not any("hf download" in command for command in recording.commands)

    assert not any("tools/convert_" in command for command in recording.commands)
    submissions = [command for command in recording.commands if "ray job submit" in command]
    if entrypoint == "prepare":
        assert recording.commands == downloads
        execute_train.assert_not_called()
        assert not submissions
    else:
        execute_train.assert_called_once_with(
            ANY,
            train_args=ANY,
            num_gpus_per_node=4,
            megatron_model_type="gpt-oss-20b",
            megatron_path="/root/Megatron-LM",
        )
        assert len(submissions) == 1
        command = submissions[0]
        assert f"--hf-checkpoint {checkpoint} --megatron-to-hf-mode bridge " in command
        assert f"--prompt-data {data_dir}/dapo-math-17k/dapo-math-17k.jsonl " in command
        assert "--num-gpus-per-node 4 " in command
        assert "--lora-rank 32 " in command
        assert "--sglang-lora-backend triton " in command
        assert "--sglang-moe-runner-backend triton " in command
        assert "--sglang-attention-backend triton " in command
