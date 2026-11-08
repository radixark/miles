import pytest

from miles.router.dynamo.config import Address, DynamoConfig, LaunchOptions
from miles.router.dynamo.frontend import frontend_env, frontend_launch, launch_args, runtime_env


@pytest.fixture
def config():
    return DynamoConfig(
        namespace="test-run",
        discovery={"backend": "etcd", "endpoints": [{"host": "localhost", "port": 2379}]},
        pins={"dynamo": "a" * 40, "sglang": "b" * 40},
    )


def test_default_launch_does_not_connect_to_inherited_nats(config):
    assert "NATS_SERVER" not in runtime_env(config, inherited_env={"NATS_SERVER": "nats://wrong:4222"})


def test_nats_is_run_owned(config):
    config = config.model_copy(update={"nats_server": "nats://nats:4222"})
    assert runtime_env(config, inherited_env={"NATS_SERVER": "nats://wrong:4222"})["NATS_SERVER"] == config.nats_server
    with pytest.raises(ValueError, match="NATS_SERVER"):
        runtime_env(config.model_copy(update={"env": {"NATS_SERVER": "wrong"}}), inherited_env={})


def test_nats_cannot_be_enabled_through_unvalidated_env(config):
    with pytest.raises(ValueError, match="NATS_SERVER"):
        runtime_env(config.model_copy(update={"env": {"NATS_SERVER": "nats://nats:4222"}}), inherited_env={})


@pytest.mark.parametrize("option", ["--namespace=wrong", "--namespace", "--names", "--no-namespace", "--"])
def test_owned_flags_cannot_be_overridden(option):
    with pytest.raises(ValueError, match="managed"):
        launch_args(LaunchOptions(extra_args=(option,)), managed={"--namespace": "test-run"})


def test_native_options_are_not_allowlisted():
    assert launch_args(LaunchOptions(extra_args=("--future-option", "value")), managed={}) == [
        "--future-option",
        "value",
    ]


def test_explicit_options_override_inherited_non_owned_env(config):
    options = LaunchOptions(env={"NEW_NATIVE_OPTION": "explicit"})
    env = runtime_env(config, options=options, inherited_env={"NEW_NATIVE_OPTION": "inherited", "OTHER": "kept"})
    assert env["NEW_NATIVE_OPTION"] == "explicit"
    assert env["OTHER"] == "kept"


def test_frontend_listener_is_supplied_only_through_argv(config):
    env = frontend_env(config, inherited_env={"DYN_HTTP_HOST": "old-host", "DYN_HTTP_PORT": "9999", "OTHER": "kept"})
    assert "DYN_HTTP_HOST" not in env
    assert "DYN_HTTP_PORT" not in env
    assert env["OTHER"] == "kept"
    argv = frontend_launch(config, address=Address(host="127.0.0.1", port=18000), interpreter_prefix=["python"])
    assert argv[:3] == ["python", "-m", "dynamo.frontend"]
    assert argv[argv.index("--http-host") + 1] == "127.0.0.1"
    assert argv[argv.index("--http-port") + 1] == "18000"


@pytest.mark.parametrize("key,value", [("DYN_HTTP_HOST", "127.0.0.1"), ("DYN_HTTP_PORT", "18000")])
@pytest.mark.parametrize("scope", ["global", "frontend"])
def test_frontend_rejects_explicit_listener_env(config, key, value, scope):
    if scope == "global":
        config = config.model_copy(update={"env": {key: value}})
    else:
        config = config.model_copy(update={"frontend": config.frontend.model_copy(update={"env": {key: value}})})
    with pytest.raises(ValueError, match=key):
        frontend_env(config, inherited_env={})
