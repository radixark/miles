from __future__ import annotations


from miles.utils.external_utils.command_utils.base_backend import (
    BaseCommandBackend,
    ExecuteTrainConfig,
    ExecuteTrainRequest,
)
from miles.utils.external_utils.command_utils.common import chart_dir, repo_base_dir
from miles.utils.external_utils.command_utils.helm_backend import command_job
from miles.utils.external_utils.command_utils.helm_backend.launcher import entrypoint
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName, RunNames


class KubernetesCommandBackend(BaseCommandBackend):
    def _execute_train_inner(self, *, request: ExecuteTrainRequest, config: ExecuteTrainConfig) -> None:
        entrypoint.execute_train(request=request, config=config)

    def exec_command_gpu(
        self, cmd: str, capture_output: bool = False, num_gpus_per_node: int | None = None
    ) -> str | None:
        return self.exec_command_multi_node(
            cmd, capture_output=capture_output, num_nodes=1, num_gpus_per_node=num_gpus_per_node
        )[0]

    def exec_command_multi_node(
        self,
        cmd: str,
        capture_output: bool = False,
        num_nodes: int | None = None,
        num_gpus_per_node: int | None = None,
    ) -> list[str | None]:
        assert self.config.namespace, "Set CommandUtilConfig.namespace to run a command somewhere"
        return command_job.run_on_nodes(
            command_job.CommandJobContext(
                namespace=self.config.namespace,
                chart_dir=chart_dir(repo_base_dir=repo_base_dir),
                helm_values_files=tuple(self.config.helm_values),
                gpus_per_node=num_gpus_per_node if num_gpus_per_node is not None else 1,
            ),
            cmd,
            capture_output=capture_output,
            completions=num_nodes or 1,
            step="command",
        )

    def api_server_host(self, config: ExecuteTrainConfig) -> str:
        assert config.run_id and config.namespace, (
            "The api server of a kubernetes run answers on the orchestrator's pod, which is named after the "
            "release; set the launch config's run_id and namespace before asking where that pod is"
        )
        assert not config.deploy_component.is_split(), (
            f"The api server, and the mini ft controller polling it, answer for the cells of their own deployment, "
            f"so a split run is refused one (--api-server-port 0) and nothing listens on the "
            f"{config.deploy_component.value} deployment for this host to name"
        )
        return RunNames.orchestrator_host(
            release=ReleaseName(
                run_id=config.run_id,
                deploy_component=config.deploy_component,
                deploy_instance_id=config.deploy_instance_id,
            ).serialize(),
            namespace=config.namespace,
        )
