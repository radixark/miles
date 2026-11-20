"""Offline tests for what the launcher itself owns of the credential wiring.

The contract (path-not-value key supply, address forwarding, SDK preflight)
lives in miles.rollout.agentic.credentials and is tested at
tests/fast/rollout/agentic/test_credentials.py. What stays here is the
launcher's own obligation: every backend the example registers must be wired
to a complete credential spec it can act on, and a backend's own launch
preflight must run before anything else can fail.
"""

import importlib.metadata
import sys
import types

import openenv_launch_common as launch
import openenv_sandbox_common as common
import pytest
import tb2_sandbox_runta


def test_every_backend_has_a_credential_spec():
    """A backend registered without credential wiring would fail at launch with
    a KeyError instead of telling the operator what to provision."""
    assert set(launch.PROVIDER_CREDENTIALS) == set(common.AGENT_MODULES)


@pytest.mark.parametrize("backend", sorted(launch.PROVIDER_CREDENTIALS))
def test_every_spec_names_a_launcher_arg(backend):
    """The arg the launcher reads must be declared on the shared config Protocol,
    else a launcher can never override the key-file path."""
    assert launch.PROVIDER_CREDENTIALS[backend]["arg_attr"] in launch.LaunchArgs.__annotations__, backend


# --- backend launch preflight ---------------------------------------------------


def _runta_launch_args(tmp_path, monkeypatch):
    """Launcher args selecting the runta backend, with everything the credential
    registry preflights in place: a token file and an importable, recent SDK."""
    token_file = tmp_path / "token"
    token_file.write_text("rt_secret\n")
    monkeypatch.setitem(sys.modules, "runta", types.ModuleType("runta"))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.2.6")
    monkeypatch.delenv("RUNTA_ENDPOINT", raising=False)
    monkeypatch.delenv("RUNTA_INGRESS_BASE_URL", raising=False)
    return types.SimpleNamespace(
        miles_host_ip="",
        openenv_sandbox_backend="runta",
        openenv_tb2_tasks_dir=str(tmp_path),
        runta_token_file=str(token_file),
    )


def test_launch_fails_fast_when_runta_cannot_tell_where_ports_are_served(tmp_path, monkeypatch):
    """A Runta endpoint that does not imply an ingress domain is a launch error.
    Left to the episodes, each would build a checkpoint and create a runtime
    before failing, and the rollout loop would refill the dropped groups forever."""
    args = _runta_launch_args(tmp_path, monkeypatch)
    monkeypatch.setattr(tb2_sandbox_runta, "api_endpoint", lambda: "http://127.0.0.1:8080")
    # Raised before the env-package check, which this test does not satisfy.
    monkeypatch.setitem(sys.modules, "tbench2_env", None)

    with pytest.raises(ValueError, match="RUNTA_INGRESS_BASE_URL"):
        launch.apply_optional_env_vars({}, args)


def test_launch_preflight_passes_a_runta_endpoint_that_implies_its_ingress(tmp_path, monkeypatch):
    """The same launch gets past the backend's check — as far as the env-package
    one — once the endpoint says where ports are served, or the operator does."""
    args = _runta_launch_args(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "tbench2_env", None)

    monkeypatch.setattr(tb2_sandbox_runta, "api_endpoint", lambda: "https://api.runta.me")
    with pytest.raises(RuntimeError, match="tbench2_env"):
        launch.apply_optional_env_vars({}, args)

    monkeypatch.setattr(tb2_sandbox_runta, "api_endpoint", lambda: "http://127.0.0.1:8080")
    monkeypatch.setenv("RUNTA_INGRESS_BASE_URL", "http://ingress.internal:9000")
    env = {}
    with pytest.raises(RuntimeError, match="tbench2_env"):
        launch.apply_optional_env_vars(env, args)
    assert env["RUNTA_INGRESS_BASE_URL"] == "http://ingress.internal:9000"  # and the workers are told


def test_only_registered_backends_have_a_launch_preflight(monkeypatch):
    """The hook is per backend: every preflight names a registered backend and a
    materialization module that really has one, and a backend without one is
    left alone (no module of its is imported for the check)."""
    assert set(common.LAUNCH_PREFLIGHTS) <= set(common.AGENT_MODULES)
    for module in common.LAUNCH_PREFLIGHTS.values():
        assert callable(importlib.import_module(module).preflight)

    monkeypatch.setattr(common.importlib, "import_module", lambda name: pytest.fail(f"imported {name}"))
    for backend in set(common.AGENT_MODULES) - set(common.LAUNCH_PREFLIGHTS):
        common.preflight_backend(backend)
