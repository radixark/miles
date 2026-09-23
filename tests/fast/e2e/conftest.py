import os
from pathlib import Path

import pytest
from tests.fast.e2e.scenario_harness import SCENARIO_RUN_ID, ScenarioHarness
from tests.utils.soak.deploy import entrypoint as deploy_entrypoint
from tests.utils.soak.ft import entrypoint as ft_entrypoint

from miles.utils.external_utils.command_utils.helm_backend.backend import KubernetesCommandBackend
from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend
from miles.utils.typer_utils import SCRIPT_ENV_VAR_PREFIX

_PROXY_ENV_VARS: tuple[str, ...] = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")


@pytest.fixture
def scenario_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ScenarioHarness:
    for name in [name for name in os.environ if name.startswith(SCRIPT_ENV_VAR_PREFIX)]:
        monkeypatch.delenv(name)
    for name in _PROXY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(f"{SCRIPT_ENV_VAR_PREFIX}RUN_ID", SCENARIO_RUN_ID)
    monkeypatch.setenv("MILES_TEST_DUMPS_ROOT", str(tmp_path / "dumps"))

    harness = ScenarioHarness(dumps_root=tmp_path / "dumps")
    monkeypatch.setattr(ft_entrypoint, "run_soak", harness.run_soak)
    monkeypatch.setattr(deploy_entrypoint, "run_soak", harness.run_soak)
    for backend_cls in (RayCommandBackend, KubernetesCommandBackend):
        monkeypatch.setattr(backend_cls, "_execute_train_inner", harness.execute_train_inner)
    return harness
