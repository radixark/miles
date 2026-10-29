import pytest

from miles.backends.dynamo_utils.config import DynamoConfig, LaunchOptions, launch_args, runtime_env


@pytest.fixture
def config():
    return DynamoConfig(
        namespace="test-run",
        discovery={"backend": "etcd", "endpoints": [{"host": "localhost", "port": 2379}]},
        pins={"dynamo": "a" * 40, "sglang": "b" * 40},
    )


def test_namespace_and_nats_are_run_owned(config):
    with pytest.raises(ValueError, match="global"):
        DynamoConfig(**(config.model_dump() | {"namespace": "dynamo"}))
    assert runtime_env(config, inherited_env={"NATS_SERVER": "nats://wrong:4222"})["NATS_SERVER"] == config.nats_server
    with pytest.raises(ValueError, match="NATS_SERVER"):
        runtime_env(config.model_copy(update={"env": {"NATS_SERVER": "wrong"}}), inherited_env={})


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
