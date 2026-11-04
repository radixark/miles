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
