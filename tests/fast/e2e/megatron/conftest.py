from pathlib import Path

import pytest
from tests.fast.utils.command_recorder import patch_helper, record_commands

from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend


@pytest.fixture
def eval_launch_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    commands = record_commands(monkeypatch)
    patch_helper(monkeypatch, "convert_checkpoint", lambda self, **kwargs: None)
    patch_helper(monkeypatch, "hf_download_dataset", lambda self, *args, **kwargs: None)
    patch_helper(monkeypatch, "_check_has_nvlink", lambda self: False, backend_class=RayCommandBackend)
    for name in (
        "RAY_ADDRESS",
        "WANDB_API_KEY",
        "NCCL_NVLS_ENABLE",
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MILES_SCRIPT_CLUSTER_BACKEND", "ray")
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    monkeypatch.setenv("MILES_SCRIPT_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    return commands
