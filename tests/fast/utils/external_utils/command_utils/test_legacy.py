import dataclasses

import pytest

import miles.utils.external_utils.command_utils.legacy as legacy
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig as CurrentExecuteTrainConfig


class _RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def execute_train(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


class TestExecuteTrainConfig:
    def test_positional_v1_config_is_converted_before_launch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The v1 positional field order and values must reach the current backend unchanged."""
        config = legacy.ExecuteTrainConfig(True, 4, "MY_VAR=value", "/output")
        backend = _RecordingBackend()
        current_configs: list[CurrentExecuteTrainConfig] = []

        def before_ray_job_submit() -> None:
            pass

        def _create_backend(current_config: CurrentExecuteTrainConfig) -> _RecordingBackend:
            current_configs.append(current_config)
            return backend

        monkeypatch.setattr(legacy, "_create_ray_backend", _create_backend)

        legacy.execute_train(
            train_args="--train-backend fsdp",
            num_gpus_per_node=8,
            megatron_model_type=None,
            config=config,
            before_ray_job_submit=before_ray_job_submit,
        )

        assert [field.name for field in dataclasses.fields(legacy.ExecuteTrainConfig)] == [
            "cuda_core_dump",
            "num_nodes",
            "extra_env_vars",
            "output_dir",
        ]
        current_config = current_configs[0]
        assert current_config.cuda_core_dump is True
        assert current_config.num_nodes == 4
        assert current_config.extra_env_vars == "MY_VAR=value"
        assert current_config.output_dir == "/output"
        assert backend.calls[0]["config"] is current_config
        assert backend.calls[0]["before_ray_job_submit"] is before_ray_job_submit
