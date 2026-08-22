import os
from pathlib import Path

import pytest
from tests.fast.utils.command_recorder import patch_helper, record_commands

import miles.utils.external_utils.command_utils as command_utils
from miles.utils.external_utils.command_utils import common
from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend
from miles.utils.typer_utils import SCRIPT_ENV_VAR_PREFIX

_CELLS_THAT_RUN_ON_A_CLUSTER = (
    "tests/e2e/ft/conftest_ft/scenario_realistic_gsm8k.py",
    "tests/e2e/short/test_qwen2.5_0.5B_gsm8k_short.py",
    "tests/e2e/short/test_qwen3_0.6B_fsdp_colocated_2xGPU.py",
)


@pytest.fixture(autouse=True)
def a_bare_environment(monkeypatch):
    """A workbench pod exports variables that would answer for the defaults under test."""
    for name in [key for key in os.environ if key.startswith(SCRIPT_ENV_VAR_PREFIX)]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def commands(monkeypatch) -> list[str]:
    recorded = record_commands(monkeypatch)
    patch_helper(monkeypatch, "_check_has_nvlink", lambda self: False, backend_class=RayCommandBackend)
    return recorded


class TestWhereModelsAndDataAreLookedFor:
    def test_a_dataset_is_downloaded_where_the_configuration_points(self, commands):
        """A pod only mounts the shared directory, so a download onto the launcher's disk never reaches it."""
        command_utils.default_config().create_backend().hf_download_dataset("zhuzilin/gsm8k")

        assert f"--local-dir {common.data_dir()}/gsm8k" in commands[-1]

    def test_a_conversion_reads_the_checkpoint_from_the_configured_model_dir(self, commands, tmp_path):
        """The default source of a conversion must move with the model directory, or it converts nothing."""
        command_utils.default_config().create_backend().convert_checkpoint(
            model_name="Qwen3-4B", megatron_model_type="qwen3-4B", num_gpus_per_node=8, dir_dst=str(tmp_path)
        )

        assert f"--hf-checkpoint {common.model_dir()}/Qwen3-4B " in commands[0]

    @pytest.mark.parametrize("relative_path", _CELLS_THAT_RUN_ON_A_CLUSTER)
    def test_a_cell_that_runs_on_a_cluster_names_no_launcher_local_path(self, relative_path):
        """These cells run where /root is the pod's own disk, so a hardcoded path there resolves to nothing."""
        source = (Path(common.repo_base_dir) / relative_path).read_text()

        assert "/root/models" not in source
        assert "/root/datasets" not in source
