import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from pathlib import Path

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig, ExecuteTrainRequest
from miles.utils.external_utils.command_utils.common import train_env_vars
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR, SNAPSHOT_UPDATE_ENV_VAR


class TestSnapshotWorkerEnvironment:
    def test_capture_directory_and_update_mode_reach_workers_without_mutating_backend_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Worker launch environments preserve the exact capture settings selected by the parent process."""
        directory = str(tmp_path / "records with spaces")
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, directory)
        monkeypatch.setenv(SNAPSHOT_UPDATE_ENV_VAR, "1")
        request = ExecuteTrainRequest(
            train_args="",
            num_gpus_per_node=1,
            megatron_model_type=None,
            train_script="/repo/train.py",
            train_backend_fsdp=False,
            extra_env_vars={},
            megatron_path="/repo/megatron",
            before_ray_job_submit=None,
            prepare_cmd={},
            extra_manifests=[],
        )
        backend_environment = {"MASTER_ADDR": "10.0.0.1"}

        environment = train_env_vars(request, backend_environment, config=ExecuteTrainConfig(run_id="explicit"))

        assert environment[SNAPSHOT_RECORD_DIR_ENV_VAR] == directory
        assert environment[SNAPSHOT_UPDATE_ENV_VAR] == "1"
        assert environment["MASTER_ADDR"] == "10.0.0.1"
        assert backend_environment == {"MASTER_ADDR": "10.0.0.1"}
