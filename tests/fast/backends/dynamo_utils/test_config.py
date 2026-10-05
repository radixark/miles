import json

import pytest
from pydantic import ValidationError

from miles.backends.dynamo_utils.config import (
    Address,
    DynamoConfig,
    LaunchOptions,
    launch_args,
    load_dynamo_config,
    runtime_env,
)


def config_dict():
    return {
        "namespace": "run-a",
        "model_path": "/models/policy",
        "discovery": {"backend": "etcd", "endpoints": [{"host": "localhost", "port": 2379}]},
        "engines": [
            {
                "name": f"replica-{index}",
                "http": {"host": f"engine-{index}", "port": 30000},
                "grpc": {"host": f"engine-{index}", "port": 30001},
                "sidecar": {"host": f"engine-{index}", "port": 8081},
                "tensor_parallel_size": 2,
            }
            for index in range(2)
        ],
        "pins": {"dynamo": "a" * 40, "sglang": "b" * 40},
    }


def test_json_roundtrip_and_frozen_fields(tmp_path):
    data = config_dict()
    path = tmp_path / "dynamo.json"
    path.write_text(json.dumps(data))
    config = load_dynamo_config(path)
    assert DynamoConfig.model_validate_json(config.model_dump_json()) == config
    assert sum(engine.tensor_parallel_size for engine in config.engines) == 4
    with pytest.raises(ValidationError, match="frozen"):
        config.namespace = "another-run"
    with pytest.raises(ValidationError, match="frozen"):
        config.engines[0].http.port = 1


@pytest.mark.parametrize(
    "field,value", [("namespace", ""), ("namespace", "a.b"), ("model_path", " "), ("engines", [])]
)
def test_invalid_config(field, value):
    data = config_dict()
    data[field] = value
    with pytest.raises(ValidationError):
        DynamoConfig.model_validate(data)


@pytest.mark.parametrize("field", ["colocate", "router_mode", "policy_version_taints", "controller_managed"])
def test_unimplemented_options_are_not_silent_noops(field):
    data = config_dict()
    data[field] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        DynamoConfig.model_validate(data)


@pytest.mark.parametrize("port", [0, 65536, True, "30000"])
def test_invalid_port(port):
    with pytest.raises(ValidationError):
        Address(host="localhost", port=port)


@pytest.mark.parametrize("host", ["http://localhost", "[::1]", "a/b", "a b", "-bad", "a..b", ""])
def test_invalid_host(host):
    with pytest.raises(ValidationError):
        Address(host=host, port=30000)


def test_address_normalization_and_ipv6():
    assert Address(host="LOCALHOST.", port=30000).url == "http://localhost:30000"
    assert Address(host="::1", port=30000).url == "http://[::1]:30000"
    assert Address(host="0:0:0:0:0:0:0:1", port=30000) == Address(host="::1", port=30000)


@pytest.mark.parametrize("problem", ["host", "wildcard", "ports", "name", "overlap", "tp"])
def test_invalid_binding(problem):
    data = config_dict()
    first, second = data["engines"]
    match problem:
        case "host":
            first["grpc"]["host"] = "wrong-engine"
        case "wildcard":
            first["sidecar"]["host"] = "0.0.0.0"
        case "ports":
            first["grpc"]["port"] = first["http"]["port"]
        case "name":
            second["name"] = first["name"]
        case "overlap":
            second["sidecar"] = first["http"].copy()
        case "tp":
            second["tensor_parallel_size"] = 4
    with pytest.raises(ValidationError):
        DynamoConfig.model_validate(data)


@pytest.mark.parametrize(
    "pins",
    [
        {"dynamo": "main", "sglang": "b" * 40},
        {"dynamo": "a" * 40, "sglang": "main"},
        {"dynamo": "a" * 40, "sglang": "b" * 40, "sglang_branch": "main"},
    ],
)
def test_source_pins(pins):
    data = config_dict()
    data["pins"] = pins
    with pytest.raises(ValidationError):
        DynamoConfig.model_validate(data)


def test_runtime_env_is_explicit_and_does_not_mutate_input():
    config = DynamoConfig.model_validate(config_dict())
    inherited = {
        "PATH": "/bin",
        "DYN_NAMESPACE_PREFIX": "other",
        "DYN_REQUEST_PLANE": "nats",
        "DYN_SYSTEM_PORT": "9000",
        "DYN_LOG": "debug",
        "DYN_NAMESPACE_WORKER_SUFFIX": "old",
        "ETCD_ENDPOINTS": "wrong",
    }
    before = inherited.copy()
    env = runtime_env(config, inherited_env=inherited)
    assert inherited == before
    assert env == {
        "PATH": "/bin",
        "DYN_SYSTEM_PORT": "9000",
        "DYN_LOG": "debug",
        "DYN_NAMESPACE": "run-a",
        "DYN_DISCOVERY_BACKEND": "etcd",
        "DYN_REQUEST_PLANE": "tcp",
        "DYN_RESPONSE_PLANE": "tcp",
        "DYN_EVENT_PLANE": "zmq",
        "ETCD_ENDPOINTS": "http://localhost:2379",
    }


def test_file_discovery(tmp_path):
    data = config_dict()
    data["discovery"] = {"backend": "file", "root": str(tmp_path)}
    config = DynamoConfig.model_validate(data)
    env = runtime_env(config, inherited_env={"ETCD_ENDPOINTS": "old"})
    assert env["DYN_FILE_KV"] == str(tmp_path)
    assert "ETCD_ENDPOINTS" not in env
    data["discovery"]["root"] = "relative"
    with pytest.raises(ValidationError, match="absolute"):
        DynamoConfig.model_validate(data)


def test_shared_planes_and_environment_precedence():
    data = config_dict()
    data.update(request_plane="nats", response_plane="quic", event_plane="nats", env={"DYN_LOG": "info"})
    config = DynamoConfig.model_validate(data)
    options = LaunchOptions(env={"DYN_LOG": "debug", "NATS_SERVER": "nats://cluster:4222"})
    inherited = {"DYN_LOG": "warn", "DYN_TCP_TLS_CA_CERT_PATH": "/tls/ca.pem", "PATH": "/bin"}
    env = runtime_env(config, inherited_env=inherited, options=options)
    assert env["DYN_LOG"] == "debug"
    assert env["DYN_TCP_TLS_CA_CERT_PATH"] == "/tls/ca.pem"
    assert env["NATS_SERVER"] == "nats://cluster:4222"
    assert (env["DYN_REQUEST_PLANE"], env["DYN_RESPONSE_PLANE"], env["DYN_EVENT_PLANE"]) == ("nats", "quic", "nats")
    assert inherited["DYN_LOG"] == "warn"


@pytest.mark.parametrize(
    "key",
    ["DYN_NAMESPACE", "DYN_NAMESPACE_PREFIX", "DYN_NAMESPACE_WORKER_SUFFIX", "DYN_REQUEST_PLANE", "ETCD_ENDPOINTS"],
)
@pytest.mark.parametrize("scope", ["shared", "process"])
def test_explicit_environment_cannot_change_owned_settings(key, scope):
    data = config_dict()
    override = {key: "conflict"}
    if scope == "shared":
        data["env"] = override
    config = DynamoConfig.model_validate(data)
    options = LaunchOptions(env=override if scope == "process" else {})
    with pytest.raises(ValueError, match=key):
        runtime_env(config, inherited_env={}, options=options)


@pytest.mark.parametrize(
    "token", ["--namespace", "--namespace=other", "--names", "--no-namespace", "--", "-h", "--help", "--version"]
)
def test_native_arguments_cannot_override_managed_options(token):
    with pytest.raises(ValueError, match="managed"):
        launch_args(LaunchOptions(extra_args=(token,)), managed={"--namespace": "run-a"})


def test_native_arguments_preserve_tokens_and_repeated_flags():
    options = LaunchOptions(
        extra_args=(
            "--metrics-prefix=training",
            "--frontend-route-extension",
            "a:b",
            "--frontend-route-extension",
            "c:d",
            "--no-tokenizer-fallback",
        )
    )
    assert launch_args(options, managed={"--namespace": "run-a"}) == ["--namespace", "run-a", *options.extra_args]


@pytest.mark.parametrize(
    "field",
    [
        "grpc_connections",
        "grpc_connect_attempt_timeout_secs",
        "grpc_retry_interval_secs",
        "grpc_startup_deadline_secs",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, "8"])
def test_invalid_grpc_transport_settings(field, value):
    data = config_dict()
    data["sidecar"] = {field: value}
    with pytest.raises(ValidationError):
        DynamoConfig.model_validate(data)
