import pytest
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.dynamo_utils.arguments import configure_dynamo
from miles.backends.dynamo_utils.config import DynamoConfig
from miles.utils.args.runtime import InferenceControllerConfig


@pytest.fixture
def configured_args(tmp_path, monkeypatch):
    config = DynamoConfig(
        namespace="test-run",
        discovery={"backend": "etcd", "endpoints": [{"host": "localhost", "port": 2379}]},
        pins={"dynamo": "a" * 40, "sglang": "b" * 40},
    )
    path = tmp_path / "dynamo.json"
    path.write_text(config.model_dump_json())
    monkeypatch.setattr("miles.backends.dynamo_utils.arguments.use_legacy_rollout_v1", lambda: False)
    return make_args(dynamo_config=str(path))


def test_fault_tolerance_and_api_key_are_accepted(configured_args):
    configured_args.ft_components = ["rollout"]
    configured_args.sglang_api_key = "test-key"
    configure_dynamo(configured_args, sglang=configured_args.sglang)
    assert configured_args.dynamo_config.namespace == "test-run"
    assert configured_args.sglang_api_key == "test-key"


def test_dynamo_configuration_survives_controller_config_transport(configured_args):
    parsed = parse_fsdp_test_config("--dynamo-config", configured_args.dynamo_config, "--rollout-num-gpus", "1")
    config = InferenceControllerConfig.slice_from(parsed)

    restored = InferenceControllerConfig.model_validate_json(config.model_dump_json())

    assert restored.dynamo_config.namespace == "test-run"
    assert restored.dynamo_config == parsed.dynamo_config


@pytest.mark.parametrize(
    "name,value",
    [
        ("rollout_external", True),
        ("use_miles_router", True),
        ("sglang_dp_size", 2),
        ("eval_uses_snapshots", True),
        ("sglang_grpc_port", 1234),
        ("sglang_sidecar_args", ["--namespace", "wrong"]),
        ("sglang_router_policy", "manual"),
    ],
)
def test_incompatible_options_fail_at_startup(configured_args, name, value):
    setattr(configured_args, name, value)
    with pytest.raises(ValueError):
        configure_dynamo(configured_args, sglang=configured_args.sglang)
