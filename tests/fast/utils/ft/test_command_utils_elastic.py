from __future__ import annotations

from unittest.mock import MagicMock, patch

from miles.utils.external_utils.command_utils import (
    ExecuteTrainConfig,
    _submit_ft_controller_job,
)


class TestExecuteTrainConfig:
    def test_enable_elastic_defaults_false(self) -> None:
        config = ExecuteTrainConfig()
        assert config.enable_elastic is False

    def test_enable_elastic_can_be_set(self) -> None:
        config = ExecuteTrainConfig(enable_elastic=True)
        assert config.enable_elastic is True


class TestSubmitFtControllerJob:
    def test_command_includes_launcher_module(self) -> None:
        mock_exec = MagicMock()
        with patch("miles.utils.external_utils.command_utils.exec_command", mock_exec):
            _submit_ft_controller_job(
                ray_address="http://127.0.0.1:8265",
                train_entrypoint="python3 train.py --lr 0.001",
                megatron_path="/root/Megatron-LM",
            )

        assert mock_exec.call_count == 1
        cmd = mock_exec.call_args[0][0]
        assert "miles.utils.ft.platform.launcher" in cmd
        assert "--as-ray-actor" in cmd
        assert "--no-wait" in cmd
        assert "--platform k8s-ray" in cmd

    def test_command_includes_ray_address(self) -> None:
        mock_exec = MagicMock()
        with patch("miles.utils.external_utils.command_utils.exec_command", mock_exec):
            _submit_ft_controller_job(
                ray_address="http://10.0.0.1:8265",
                train_entrypoint="python3 train.py",
                megatron_path="/root/Megatron-LM",
            )

        cmd = mock_exec.call_args[0][0]
        assert "http://10.0.0.1:8265" in cmd

    def test_command_includes_entrypoint(self) -> None:
        mock_exec = MagicMock()
        with patch("miles.utils.external_utils.command_utils.exec_command", mock_exec):
            _submit_ft_controller_job(
                ray_address="http://127.0.0.1:8265",
                train_entrypoint="python3 /root/miles/train.py --lr 0.001",
                megatron_path="/root/Megatron-LM",
            )

        cmd = mock_exec.call_args[0][0]
        assert "python3 /root/miles/train.py --lr 0.001" in cmd


class TestExecuteTrainElasticIntegration:
    """Test that execute_train() correctly handles enable_elastic.

    These tests mock exec_command and check_has_nvlink to avoid running
    actual shell commands.
    """

    def _run_execute_train(
        self,
        *,
        enable_elastic: bool = False,
        train_args: str = "--train-backend fsdp --lr 0.001",
    ) -> list[str]:
        """Run execute_train with mocked externals, return all exec_command calls."""
        mock_exec = MagicMock()
        config = ExecuteTrainConfig(enable_elastic=enable_elastic)

        with (
            patch("miles.utils.external_utils.command_utils.exec_command", mock_exec),
            patch("miles.utils.external_utils.command_utils.check_has_nvlink", return_value=True),
            patch("miles.utils.external_utils.command_utils.time.sleep"),
            patch("miles.utils.external_utils.command_utils.get_bool_env_var", side_effect=lambda name, default="false": {
                "MILES_SCRIPT_EXTERNAL_RAY": True,
                "MILES_SCRIPT_ENABLE_RAY_SUBMIT": True,
            }.get(name, default == "true")),
        ):
            from miles.utils.external_utils.command_utils import execute_train

            execute_train(
                train_args=train_args,
                num_gpus_per_node=8,
                megatron_model_type=None,
                config=config,
            )

        return [c[0][0] for c in mock_exec.call_args_list]

    def test_elastic_disabled_no_ft_controller_job(self) -> None:
        commands = self._run_execute_train(enable_elastic=False)
        ft_commands = [c for c in commands if "ft.platform.launcher" in c]
        assert len(ft_commands) == 0

    def test_elastic_enabled_submits_ft_controller_job(self) -> None:
        commands = self._run_execute_train(enable_elastic=True)
        ft_commands = [c for c in commands if "ft.platform.launcher" in c]
        assert len(ft_commands) == 1
        assert "--no-wait" in ft_commands[0]

    def test_elastic_enabled_adds_use_fault_tolerance_flag(self) -> None:
        commands = self._run_execute_train(
            enable_elastic=True,
            train_args="--train-backend fsdp --lr 0.001",
        )
        train_commands = [c for c in commands if "ray job submit" in c and "ft.platform.launcher" not in c]
        assert len(train_commands) == 1
        assert "--use-fault-tolerance" in train_commands[0]

    def test_elastic_enabled_does_not_duplicate_flag(self) -> None:
        commands = self._run_execute_train(
            enable_elastic=True,
            train_args="--train-backend fsdp --use-fault-tolerance --lr 0.001",
        )
        train_commands = [c for c in commands if "ray job submit" in c and "ft.platform.launcher" not in c]
        assert len(train_commands) == 1
        cmd = train_commands[0]
        assert cmd.count("--use-fault-tolerance") == 1

    def test_elastic_disabled_does_not_add_flag(self) -> None:
        commands = self._run_execute_train(
            enable_elastic=False,
            train_args="--train-backend fsdp --lr 0.001",
        )
        train_commands = [c for c in commands if "ray job submit" in c]
        assert len(train_commands) == 1
        assert "--use-fault-tolerance" not in train_commands[0]
