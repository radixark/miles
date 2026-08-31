from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import sys
import types

import pytest

from miles.rollout.agentic.credentials import (
    PROVIDER_CREDENTIALS,
    forward_address,
    preflight_sdk,
    resolve_provider_api_key,
    sandbox_key_supply,
)

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


# --- the credential table ----------------------------------------------------


@pytest.mark.parametrize("backend", sorted(PROVIDER_CREDENTIALS))
def test_credential_spec_is_complete(backend):
    spec = PROVIDER_CREDENTIALS[backend]
    assert set(spec) == _SPEC_KEYS, backend
    assert spec["key_env_vars"], backend
    assert spec["default_path"], backend


def test_forwarded_vars_are_addresses_not_secrets():
    """Whatever `forward` names is forwarded BY VALUE through ray's runtime_env,
    which is logged in plaintext — so no credential-shaped var may appear."""
    for backend, spec in PROVIDER_CREDENTIALS.items():
        for var in spec["forward"]:
            assert not any(word in var for word in ("KEY", "TOKEN", "SECRET")), (backend, var)


def test_a_forwarded_url_may_not_smuggle_a_credential():
    """The check above is on the var's NAME, which cannot see a credential
    hidden in an endpoint's userinfo — and that value would be forwarded and
    logged in plaintext just the same."""
    env: dict[str, str] = {}
    forward_address(env, "E2B_API_URL", "https://agentenv.internal")
    assert env == {"E2B_API_URL": "https://agentenv.internal"}

    with pytest.raises(ValueError, match="embeds credentials"):
        forward_address({}, "E2B_API_URL", "https://user:tok@agentenv.internal")


# --- launcher-side key supply -------------------------------------------------


def _supply(env, spec_name, *, arg_path="", **overrides):
    spec = PROVIDER_CREDENTIALS[spec_name]
    kwargs = {
        "provider": spec["provider"],
        "key_env_vars": spec["key_env_vars"],
        "file_env_var": spec["file_env_var"],
        "arg_path": arg_path,
        "default_path": spec["default_path"],
        "provision_hint": spec["provision_hint"],
    }
    sandbox_key_supply(env, **{**kwargs, **overrides})


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


def test_unreadable_path_counts_as_absent(monkeypatch, tmp_path):
    """A path that raises on read (here: a directory) is 'no file', so the env
    supply still satisfies — the launcher cannot probe worker nodes anyway."""
    monkeypatch.setenv("E2B_API_KEY", "e2b_from_env")
    env: dict[str, str] = {}
    _supply(env, "e2b", default_path=str(tmp_path))
    assert env == {}


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


# --- SDK preflight -------------------------------------------------------------


def test_preflight_passes_when_the_sdk_imports(monkeypatch):
    monkeypatch.setitem(sys.modules, "fake_provider_sdk", types.ModuleType("fake_provider_sdk"))
    preflight_sdk("fake_provider_sdk", "pip install fake-provider-sdk")


def test_preflight_names_the_install_hint_when_the_sdk_is_missing():
    with pytest.raises(RuntimeError, match="pip install nonexistent-sdk"):
        preflight_sdk("nonexistent_provider_sdk", "pip install nonexistent-sdk")


# --- worker-side key resolution -------------------------------------------------


def test_resolve_env_value_wins_over_the_file(monkeypatch, tmp_path):
    key_file = tmp_path / "api_key"
    key_file.write_text("from_file\n")
    monkeypatch.setenv("PROV_API_KEY", "from_env")
    monkeypatch.setenv("PROV_API_KEY_FILE", str(key_file))
    assert resolve_provider_api_key("PROV_API_KEY", "PROV_API_KEY_FILE", "~/nope") == "from_env"


def test_resolve_falls_back_to_the_file_and_strips_whitespace(monkeypatch, tmp_path):
    key_file = tmp_path / "api_key"
    key_file.write_text("  from_file \n")
    monkeypatch.delenv("PROV_API_KEY", raising=False)
    monkeypatch.setenv("PROV_API_KEY_FILE", str(key_file))
    assert resolve_provider_api_key("PROV_API_KEY", "PROV_API_KEY_FILE", "~/nope") == "from_file"


def test_resolve_default_path_expands_the_home_dir(monkeypatch, tmp_path):
    (tmp_path / ".config").mkdir()
    (tmp_path / ".config" / "api_key").write_text("from_default\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PROV_API_KEY", raising=False)
    monkeypatch.delenv("PROV_API_KEY_FILE", raising=False)
    assert resolve_provider_api_key("PROV_API_KEY", "PROV_API_KEY_FILE", "~/.config/api_key") == "from_default"


def test_resolve_missing_key_names_both_supplies(monkeypatch, tmp_path):
    monkeypatch.delenv("PROV_API_KEY", raising=False)
    monkeypatch.setenv("PROV_API_KEY_FILE", str(tmp_path / "absent"))
    with pytest.raises(RuntimeError, match="PROV_API_KEY is unset"):
        resolve_provider_api_key("PROV_API_KEY", "PROV_API_KEY_FILE", "~/nope")
