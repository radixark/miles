"""Offline tests for the launcher's Harbor-side wiring (no harbor, no keys)."""

import sys
import types
from types import SimpleNamespace

import launch_common
import pytest


def _args(**overrides):
    defaults = dict(
        harbor_env_type="e2b",
        harbor_env_kwargs="",
        harbor_tasks_dir="/tasks",
        harbor_trials_dir="/trials",
        agent_model_name="model",
        agent_timeout=5400,
        router_external_host="trainer.tailnet",
        daytona_api_key_file="",
        e2b_api_key_file="",
        modal_config_file="",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def fake_harbor_and_sdk(monkeypatch):
    """harbor and the provider SDKs only need to be importable for the preflight."""
    for name in ("harbor", "e2b", "daytona", "modal"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))


def test_env_type_is_required(monkeypatch):
    with pytest.raises(ValueError, match="HARBOR_ENV_TYPE"):
        launch_common.harbor_env_vars(_args(harbor_env_type=""))


def test_docker_is_refused_with_a_pointer_to_the_agent_server():
    with pytest.raises(ValueError, match="swe-agent-harbor-docker"):
        launch_common.harbor_env_vars(_args(harbor_env_type="docker"))


def test_missing_harbor_names_the_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "harbor", None)
    with pytest.raises(RuntimeError, match="README install line"):
        launch_common.harbor_env_vars(_args())


def test_known_provider_is_provisioned_by_key_path(monkeypatch, tmp_path):
    key_file = tmp_path / "api_key"
    key_file.write_text("e2b_secret\n")
    monkeypatch.setenv("E2B_API_URL", "http://agentenv.internal:8000")

    env = launch_common.harbor_env_vars(_args(e2b_api_key_file=str(key_file)))

    assert env["HARBOR_ENV_TYPE"] == "e2b"
    assert env["HARBOR_TASKS_DIR"] == "/tasks"
    assert env["MILES_ROUTER_EXTERNAL_HOST"] == "trainer.tailnet"
    assert env["E2B_API_KEY_FILE"] == str(key_file)
    assert env["E2B_API_URL"] == "http://agentenv.internal:8000"
    assert "e2b_secret" not in str(env)


def test_unknown_provider_passes_through_with_a_notice(monkeypatch, capsys):
    """The backend still reaches Harbor untouched; the operator is told no
    credential wiring exists for it."""
    env = launch_common.harbor_env_vars(_args(harbor_env_type="runloop"))
    assert env["HARBOR_ENV_TYPE"] == "runloop"
    assert "no credential wiring known for 'runloop'" in capsys.readouterr().out
    assert not any("KEY" in var for var in env)


def test_env_kwargs_and_server_knobs_are_forwarded_when_set(monkeypatch):
    monkeypatch.setenv("E2B_API_KEY", "e2b_x")  # worker-env key supply
    monkeypatch.setenv("HARBOR_RESPONSE_LENGTH_POLICY", "abort")
    monkeypatch.delenv("HARBOR_MAX_SEQ_LEN", raising=False)

    env = launch_common.harbor_env_vars(_args(harbor_env_kwargs='{"auto_snapshot": true}'))

    assert env["HARBOR_ENV_KWARGS"] == '{"auto_snapshot": true}'
    assert env["HARBOR_RESPONSE_LENGTH_POLICY"] == "abort"
    assert "HARBOR_MAX_SEQ_LEN" not in env
