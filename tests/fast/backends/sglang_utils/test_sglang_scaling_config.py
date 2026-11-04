import pytest

from miles.backends.sglang_utils.sglang_config import ServerGroupConfig


class TestDeploymentFieldOwnership:
    @pytest.mark.parametrize("field", ["num_gpus", "gpu_offset", "engine_offset"])
    def test_worker_server_groups_reject_deployment_fields(self, field: str) -> None:
        """A worker group rejects deployment layout even after its valid payload survives transport."""
        group = ServerGroupConfig(worker_type="regular", num_gpus_per_engine=2, needs_offload=False)
        restored = ServerGroupConfig.model_validate_json(group.model_dump_json())
        values = restored.model_dump(mode="json")

        assert values["num_gpus_per_engine"] == 2
        assert not {"num_gpus", "gpu_offset", "engine_offset"}.intersection(values)
        with pytest.raises(ValueError, match="extra_forbidden"):
            ServerGroupConfig.model_validate(values | {field: 8})
