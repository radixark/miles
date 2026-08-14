from __future__ import annotations


from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend, ExecuteTrainRequest
from miles.utils.external_utils.command_utils.helm_backend.launcher import entrypoint
from miles.utils.external_utils.command_utils.helm_backend.naming import RunNames


class KubernetesCommandBackend(BaseCommandBackend):
    def _execute_train_inner(self, request: ExecuteTrainRequest) -> None:
        entrypoint.execute_train(request=request, config=self.config)

    def exec_command_gpu(
        self, cmd: str, capture_output: bool = False, num_gpus_per_node: int | None = None
    ) -> str | None:
        raise NotImplementedError("A later milestone runs a command as a job in the cluster")

    def exec_command_multi_node(
        self,
        cmd: str,
        capture_output: bool = False,
        num_nodes: int | None = None,
        num_gpus_per_node: int | None = None,
    ) -> list[str | None]:
        raise NotImplementedError("A later milestone runs a command as a job in the cluster")

    def api_server_host(self) -> str:
        assert self.config.run_id and self.config.namespace, (
            "The api server of a kubernetes run answers on the orchestrator's pod, which is named after the "
            "release; set ExecuteTrainConfig.run_id and .namespace before asking where that pod is"
        )
        return RunNames.orchestrator_host(
            release=RunNames.release(run_id=self.config.run_id), namespace=self.config.namespace
        )
