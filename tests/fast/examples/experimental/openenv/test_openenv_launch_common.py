"""Offline unit tests for the launcher's sandbox-credential wiring.

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/openenv -q

The launcher is the only place that can turn a missing credential into a
launch-time error instead of a rollout that aborts every sample and refills
forever, so what it must get right is: never forward a secret VALUE, accept
either supply (file path or worker env), and treat a partially-supplied
credential (Modal's token pair) as missing rather than usable.
"""

import openenv_launch_common as launch
import openenv_sandbox_common as common
import pytest

_SPEC_KEYS = {
    "provider",
    "key_env_vars",
    "file_env_var",
    "arg_attr",
    "default_path",
    "provision_hint",
    "sdk",
    "sdk_hint",
    "forward",
    "target",
}


# --- the credential table ---------------------------------------------------


def test_every_backend_has_a_credential_spec():
    """A backend registered without credential wiring would fail at launch with
    a KeyError instead of telling the operator what to provision."""
    assert set(launch.PROVIDER_CREDENTIALS) == set(common.AGENT_MODULES)


@pytest.mark.parametrize("backend", sorted(launch.PROVIDER_CREDENTIALS))
def test_credential_spec_is_complete(backend):
    spec = launch.PROVIDER_CREDENTIALS[backend]
    assert set(spec) == _SPEC_KEYS, backend
    assert spec["key_env_vars"], backend
    # The arg the launcher reads must be declared on the shared config Protocol,
    # else a launcher can never override the path.
    assert spec["arg_attr"] in launch.LaunchArgs.__annotations__, backend


def test_forwarded_vars_are_addresses_not_secrets():
    """Whatever `forward` names is forwarded BY VALUE through ray's runtime_env,
    which is logged in plaintext — so no credential-shaped var may appear."""
    for backend, spec in launch.PROVIDER_CREDENTIALS.items():
        for var in spec["forward"]:
            assert not any(word in var for word in ("KEY", "TOKEN", "SECRET")), (backend, var)


def test_a_forwarded_url_may_not_smuggle_a_credential():
    """The check above is on the var's NAME, which cannot see a credential
    hidden in an endpoint's userinfo — and that value would be forwarded and
    logged in plaintext just the same."""
    env: dict[str, str] = {}
    launch.forward_address(env, "E2B_API_URL", "https://agentenv.internal")
    assert env == {"E2B_API_URL": "https://agentenv.internal"}

    with pytest.raises(ValueError, match="embeds credentials"):
        launch.forward_address({}, "E2B_API_URL", "https://user:tok@agentenv.internal")


# --- key supply -------------------------------------------------------------


def _supply(env, spec_name, *, arg_path="", **overrides):
    spec = launch.PROVIDER_CREDENTIALS[spec_name]
    kwargs = {
        "provider": spec["provider"],
        "key_env_vars": spec["key_env_vars"],
        "file_env_var": spec["file_env_var"],
        "arg_path": arg_path,
        "default_path": spec["default_path"],
        "provision_hint": spec["provision_hint"],
    }
    launch.sandbox_key_supply(env, **{**kwargs, **overrides})


def test_readable_file_forwards_the_path_never_the_value(tmp_path):
    key_file = tmp_path / "api_key"
    key_file.write_text("dtn_secret_value\n")
    env: dict[str, str] = {}
    _supply(env, "daytona", arg_path=str(key_file))
    assert env == {"DAYTONA_API_KEY_FILE": str(key_file)}
    assert "dtn_secret_value" not in str(env)


def test_configured_path_that_does_not_resolve_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="is missing or empty"):
        _supply({}, "e2b", arg_path=str(tmp_path / "absent"))


def test_empty_file_is_not_a_credential(tmp_path):
    key_file = tmp_path / "api_key"
    key_file.write_text("   \n")
    with pytest.raises(ValueError, match="is missing or empty"):
        _supply({}, "e2b", arg_path=str(key_file))


def test_worker_environment_supply_is_accepted_without_forwarding(monkeypatch, tmp_path):
    """When the launcher itself has the credential in env, workers are assumed
    to have it too (platform-injected / single-host inheritance) and nothing is
    forwarded."""
    monkeypatch.setenv("DAYTONA_API_KEY", "dtn_from_env")
    env: dict[str, str] = {}
    _supply(env, "daytona", default_path=str(tmp_path / "absent"))
    assert env == {}


def test_modal_token_pair_must_be_complete(monkeypatch, tmp_path):
    """Half a token pair is a misconfiguration, not a usable credential: it
    would authenticate nothing and fail every episode."""
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-123")
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    with pytest.raises(ValueError, match="MODAL_TOKEN_ID \\+ MODAL_TOKEN_SECRET"):
        _supply({}, "modal", default_path=str(tmp_path / "absent"))

    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-456")
    env: dict[str, str] = {}
    _supply(env, "modal", default_path=str(tmp_path / "absent"))
    assert env == {}  # the token halves are never forwarded


def test_modal_config_file_is_forwarded_by_path(tmp_path):
    """Modal's file is a config file rather than a bare key, but it rides the
    same path-not-value contract."""
    config = tmp_path / "modal.toml"
    config.write_text('[radixark]\ntoken_id = "ak-123"\ntoken_secret = "as-456"\n')
    env: dict[str, str] = {}
    _supply(env, "modal", arg_path=str(config))
    assert env == {"MODAL_CONFIG_PATH": str(config)}
    assert "as-456" not in str(env)


def test_missing_credential_names_what_to_provision(monkeypatch, tmp_path):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    with pytest.raises(ValueError, match="mkdir -p ~/.config/e2b"):
        _supply({}, "e2b", default_path=str(tmp_path / "absent"))
