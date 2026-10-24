import pytest
import torch
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config, parse_megatron_test_config

from miles.utils.args.runtime import OrchestratorConfig, RolloutConfig


class TestRawBackendTraitTransport:
    @pytest.mark.parametrize("component", [OrchestratorConfig, RolloutConfig])
    @pytest.mark.parametrize("backend", ["megatron", "fsdp", "torchtitan"])
    def test_shared_backend_inputs_survive_component_slicing_and_json(
        self, component: type[OrchestratorConfig | RolloutConfig], backend: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both consumers retain the original backend inputs after their independent transport boundary."""
        monkeypatch.setattr(torch, "__version__", "2.13.0")
        config = (
            parse_megatron_test_config("--lr", "0.003")
            if backend == "megatron"
            else parse_fsdp_test_config(
                "--train-backend", backend, "--lr", "0.003", "--rollout-max-response-len", "128"
            )
        )
        sliced = component.slice_from(config)

        restored = component.model_validate_json(sliced.model_dump_json())

        assert restored.raw_megatron.trainers == config.raw_megatron.trainers
        assert restored.raw_fsdp is None if backend != "fsdp" else restored.raw_fsdp.lr == 0.003
        assert restored.raw_torchtitan is None if backend != "torchtitan" else restored.raw_torchtitan.lr == 0.003
        if backend == "megatron":
            assert restored.raw_megatron.base_args["lr"] == 0.003
