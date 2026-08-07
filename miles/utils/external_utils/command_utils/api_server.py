from __future__ import annotations

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.workers.types import ClusterBackend


def api_server_url(config: ExecuteTrainConfig, *, port: int) -> str:
    if ClusterBackend(config.cluster_backend) is not ClusterBackend.KUBERNETES:
        return f"http://localhost:{port}"

    assert config.run_id and config.namespace, (
        "the api server of a kubernetes run answers on the orchestrator's pod, which is named after the release; "
        "set ExecuteTrainConfig.run_id and .namespace before asking where that pod is"
    )
    host = naming.orchestrator_host(naming.release_name(config.run_id), config.namespace)
    return f"http://{host}:{port}"
