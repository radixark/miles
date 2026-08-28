import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from tests.e2e.common_dirs import get_test_data_dir, get_test_model_dir
from tests.fast.utils.command_recorder import record_commands

from miles.utils.external_utils.command_utils.common import repo_base_dir

_CELLS_THAT_RUN_ON_A_CLUSTER = (
    "tests/e2e/ft/conftest_ft/scenario_realistic_gsm8k.py",
    "tests/e2e/long/test_qwen2.5_0.5B_gsm8k.py",
    "tests/e2e/short/test_qwen2.5_0.5B_gsm8k_short.py",
    "tests/e2e/short/test_qwen3_0.6B_fsdp_colocated_2xGPU.py",
)
_CELLS_WITH_A_STANDALONE_PREPARE = (
    "tests/e2e/short/test_qwen2.5_0.5B_gsm8k_short.py",
    "tests/e2e/short/test_qwen3_0.6B_fsdp_colocated_2xGPU.py",
)


def _load_cell(relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("cell_under_test", Path(repo_base_dir) / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestWhereModelsAndDataAreLookedFor:
    def test_the_directories_fall_back_to_the_image_defaults(self, monkeypatch):
        """A pod that exports nothing still has to find what the image baked in."""
        monkeypatch.delenv("MILES_SCRIPT_MODEL_DIR", raising=False)
        monkeypatch.delenv("MILES_SCRIPT_DATA_DIR", raising=False)

        assert get_test_model_dir() == "/root/models"
        assert get_test_data_dir() == "/root/datasets"

    def test_the_environment_overrides_the_directories(self, monkeypatch):
        """A cluster mounts its shared storage elsewhere, so the launcher must be able to redirect both."""
        monkeypatch.setenv("MILES_SCRIPT_MODEL_DIR", "/shared/models")
        monkeypatch.setenv("MILES_SCRIPT_DATA_DIR", "/shared/datasets")

        assert get_test_model_dir() == "/shared/models"
        assert get_test_data_dir() == "/shared/datasets"

    @pytest.mark.parametrize("relative_path", _CELLS_THAT_RUN_ON_A_CLUSTER)
    def test_a_cell_that_runs_on_a_cluster_names_no_launcher_local_path(self, relative_path):
        """These cells run where /root is the pod's own disk, so a hardcoded path there resolves to nothing."""
        source = (Path(repo_base_dir) / relative_path).read_text()

        assert "/root/models" not in source
        assert "/root/datasets" not in source

    @pytest.mark.parametrize("relative_path", _CELLS_WITH_A_STANDALONE_PREPARE)
    def test_a_cell_prepares_into_the_configured_directories(self, monkeypatch, relative_path):
        """A download left to the backend's own default lands where the pod never looks."""
        monkeypatch.setenv("MILES_SCRIPT_MODEL_DIR", "/shared/models")
        monkeypatch.setenv("MILES_SCRIPT_DATA_DIR", "/shared/datasets")
        commands = record_commands(monkeypatch)

        _load_cell(relative_path).prepare()

        assert [cmd for cmd in commands if "--local-dir /shared/models/" in cmd]
        assert [cmd for cmd in commands if "--local-dir /shared/datasets/gsm8k" in cmd]
        assert not [cmd for cmd in commands if "/root/models" in cmd or "/root/datasets" in cmd]
