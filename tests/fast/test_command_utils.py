from miles.utils.external_utils import command_utils


def test_execute_train_exports_extra_envs_before_model_args_source(monkeypatch):
    commands = []

    def fake_exec_command(cmd, capture_output=False):
        commands.append(cmd)
        return "0" if capture_output else None

    monkeypatch.setattr(command_utils, "exec_command", fake_exec_command)
    monkeypatch.setattr(command_utils, "check_has_nvlink", lambda: True)

    command_utils.execute_train(
        train_args="--dummy-train-arg",
        num_gpus_per_node=8,
        megatron_model_type="qwen3-30B-A3B",
        extra_env_vars={
            "MODEL_ARGS_DISABLE_MOE_PERMUTE_FUSION": "1",
            "SPACE_VALUE": "a b",
        },
        config=command_utils.ExecuteTrainConfig(
            extra_env_vars='{"CONFIG_VALUE": "from config"}',
        ),
    )

    submit_cmd = next(cmd for cmd in commands if "ray job submit" in cmd)
    source_index = submit_cmd.index("source ")
    disable_index = submit_cmd.index("export MODEL_ARGS_DISABLE_MOE_PERMUTE_FUSION=1")
    space_index = submit_cmd.index("export SPACE_VALUE='a b'")
    config_index = submit_cmd.index("export CONFIG_VALUE='from config'")

    assert disable_index < source_index
    assert space_index < source_index
    assert config_index < source_index
    assert '"MODEL_ARGS_DISABLE_MOE_PERMUTE_FUSION": "1"' in submit_cmd
    assert '"CONFIG_VALUE": "from config"' in submit_cmd
