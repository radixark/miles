import pytest
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

from miles.utils.args.runtime import OrchestratorConfig
from miles.utils.workers.connection_config import StaticConnConfig
from miles.utils.workers.serving.worker_config import OrchestratorWorkerConfig


@pytest.fixture
def orchestrator_payload() -> OrchestratorWorkerConfig:
    args = parse_fsdp_test_config("--cluster-backend", "kubernetes")
    return OrchestratorWorkerConfig(
        args=OrchestratorConfig.slice_from(args).model_dump(mode="json"), static_connections=StaticConnConfig()
    )
