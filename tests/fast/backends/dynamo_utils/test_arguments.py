from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.dynamo_utils.arguments import configure_dynamo
from miles.backends.dynamo_utils.config import DynamoConfig
from miles.backends.sglang_utils.sglang_api_client import WorkerType


@pytest.fixture
def configured_args(tmp_path, monkeypatch):
    config = DynamoConfig(
        namespace="test-run",
        discovery={"backend": "etcd", "endpoints": [{"host": "localhost", "port": 2379}]},
        pins={"dynamo": "a" * 40, "sglang": "b" * 40},
    )
    path = tmp_path / "dynamo.json"
    path.write_text(config.model_dump_json())
    resolved = SimpleNamespace(
        models=[SimpleNamespace(server_groups=[SimpleNamespace(worker_type=WorkerType.REGULAR, overrides={})])]
    )
    monkeypatch.setattr("miles.backends.dynamo_utils.arguments.resolve_sglang_config", lambda args: resolved)
    monkeypatch.setattr("miles.backends.dynamo_utils.arguments.use_legacy_rollout_v1", lambda: False)
    return make_args(dynamo_config=str(path))


def test_fault_tolerance_and_api_key_are_accepted(configured_args):
    configured_args.ft_components = ["rollout"]
    configured_args.sglang_api_key = "test-key"
    configure_dynamo(configured_args)
    assert configured_args.dynamo_config.namespace == "test-run"
    assert configured_args.sglang_api_key == "test-key"


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
        configure_dynamo(configured_args)
