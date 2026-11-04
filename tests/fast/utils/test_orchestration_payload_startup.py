import json
import sys

import pytest
from tests.fast.utils.orchestration_startup_fixtures import orchestrator_payload

from miles.utils.orchestration_utils import (
    ArgvOrchestratorStartupInfo,
    PayloadOrchestratorStartupInfo,
    parse_orchestrator_startup_info,
)
from miles.utils.workers.argv_utils import ORCHESTRATOR_CONFIG_FLAG
from miles.utils.workers.serving.utils import parse_orchestrator_argv
from miles.utils.workers.serving.worker_config import OrchestratorWorkerConfig

__all__ = ["orchestrator_payload"]


class TestPayloadStartup:
    @pytest.mark.parametrize("equals_form", [False, True])
    def test_a_payload_dispatches_without_requiring_full_training_command_line_arguments(
        self, monkeypatch: pytest.MonkeyPatch, orchestrator_payload: OrchestratorWorkerConfig, equals_form: bool
    ) -> None:
        """Both supported flag spellings rebuild the driver from the payload alone."""
        value = orchestrator_payload.model_dump_json()
        argv = [f"{ORCHESTRATOR_CONFIG_FLAG}={value}"] if equals_form else [ORCHESTRATOR_CONFIG_FLAG, value]
        monkeypatch.setattr(sys, "argv", ["train.py", *argv])

        startup = parse_orchestrator_startup_info()

        assert isinstance(startup, PayloadOrchestratorStartupInfo)
        assert startup.args.cluster_backend == "kubernetes"
        assert startup.args.num_rollout == 1
        assert startup.static_connections == orchestrator_payload.static_connections
        assert "rollout_num_gpus" not in startup.args.model_dump()

    @pytest.mark.parametrize("backend", ["ray"])
    def test_a_payload_refuses_a_backend_that_would_need_the_full_worker_launch_configuration(
        self, orchestrator_payload: OrchestratorWorkerConfig, backend: str
    ) -> None:
        """Payload startup supports only workers already installed by the Kubernetes launcher."""
        payload = orchestrator_payload.model_copy(
            update={"args": orchestrator_payload.args | {"cluster_backend": backend}}
        )

        with pytest.raises(AssertionError, match="starts an orchestrator whose workers the launcher installed"):
            PayloadOrchestratorStartupInfo.create(payload)


class TestPayloadProtocolRejections:
    @pytest.mark.parametrize("count", [0, 2])
    def test_a_missing_or_repeated_payload_is_refused(
        self, orchestrator_payload: OrchestratorWorkerConfig, count: int
    ) -> None:
        """An ambiguous command must not silently select a payload."""
        argv = [ORCHESTRATOR_CONFIG_FLAG, orchestrator_payload.model_dump_json()] * count

        with pytest.raises(AssertionError, match="must be given exactly once"):
            parse_orchestrator_argv(argv)

    def test_a_flag_without_a_value_is_refused(self) -> None:
        """A trailing payload flag reports its missing value before JSON parsing."""
        with pytest.raises(AssertionError, match="last argument"):
            parse_orchestrator_argv([ORCHESTRATOR_CONFIG_FLAG])

    @pytest.mark.parametrize("field", ["args", "static_connections"])
    def test_an_incomplete_payload_cannot_silently_receive_defaults(
        self, orchestrator_payload: OrchestratorWorkerConfig, field: str
    ) -> None:
        """Both top-level protocol fields must be transported explicitly."""
        payload = orchestrator_payload.model_dump(mode="json")
        del payload[field]

        with pytest.raises(ValueError):
            parse_orchestrator_argv([ORCHESTRATOR_CONFIG_FLAG, json.dumps(payload)])

    def test_unknown_payload_fields_are_refused(self, orchestrator_payload: OrchestratorWorkerConfig) -> None:
        """A misspelled protocol field cannot be dropped by the runtime."""
        payload = orchestrator_payload.model_dump(mode="json") | {"worker_type": "trainer"}

        with pytest.raises(ValueError):
            parse_orchestrator_argv([ORCHESTRATOR_CONFIG_FLAG, json.dumps(payload)])

    def test_invalid_json_is_refused(self) -> None:
        """Broken transported JSON cannot fall back to parsing the full training argv."""
        with pytest.raises(json.JSONDecodeError):
            parse_orchestrator_argv([ORCHESTRATOR_CONFIG_FLAG, "{broken"])

    def test_nested_static_connections_must_be_complete(self, orchestrator_payload: OrchestratorWorkerConfig) -> None:
        """The runtime must reject a roster whose fields were omitted during transport."""
        payload = orchestrator_payload.model_dump(mode="json")
        payload["static_connections"] = {}

        with pytest.raises(ValueError, match="Incomplete configuration"):
            parse_orchestrator_argv([ORCHESTRATOR_CONFIG_FLAG, json.dumps(payload)])

    def test_unknown_driver_fields_are_refused(self, orchestrator_payload: OrchestratorWorkerConfig) -> None:
        """Transported driver settings cannot silently discard unknown keys."""
        payload = orchestrator_payload.model_copy(update={"args": orchestrator_payload.args | {"num_roolout": 4}})

        with pytest.raises(ValueError):
            PayloadOrchestratorStartupInfo.create(payload)


class TestLegacyArgvStartup:
    def test_without_a_payload_the_real_parser_builds_both_startup_configs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Normal CLI launches retain the full worker configuration and a separate driver slice."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train.py",
                "--train-backend",
                "fsdp",
                "--rollout-batch-size",
                "2",
                "--num-rollout",
                "1",
                "--actor-num-gpus-per-node",
                "1",
                "--micro-batch-size",
                "1",
            ],
        )
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("MILES_SCRIPT_ENV_REPORT", "")

        startup = parse_orchestrator_startup_info()

        assert isinstance(startup, ArgvOrchestratorStartupInfo)
        assert startup.all_args.train_backend == "fsdp"
        assert startup.args.num_rollout == 1
        assert startup.args is not startup.all_args
