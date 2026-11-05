import pytest

from miles.router.dynamo.config import DynamoConfig


@pytest.fixture
def config():
    return DynamoConfig(
        namespace="test-run",
        discovery={"backend": "etcd", "endpoints": [{"host": "localhost", "port": 2379}]},
        pins={"dynamo": "a" * 40, "sglang": "b" * 40},
    )


def test_namespace_is_run_specific(config):
    with pytest.raises(ValueError, match="global"):
        DynamoConfig(**(config.model_dump() | {"namespace": "dynamo"}))


def test_default_planes_do_not_require_nats(config):
    assert config.nats_server is None


@pytest.mark.parametrize("plane", ["request_plane", "event_plane"])
def test_nats_plane_requires_explicit_server(config, plane):
    data = config.model_dump() | {plane: "nats"}
    with pytest.raises(ValueError, match="nats_server is required"):
        DynamoConfig(**data)
    assert DynamoConfig(**(data | {"nats_server": "nats://nats:4222"})).nats_server == "nats://nats:4222"
