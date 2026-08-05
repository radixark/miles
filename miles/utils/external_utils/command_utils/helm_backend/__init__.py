from __future__ import annotations

import hashlib
import re
import shlex
from typing import Any

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.arguments import parse_args_from_argv
from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend, ExecuteTrainRequest, use_backend
from miles.utils.external_utils.command_utils.common import repo_base_dir
from miles.utils.external_utils.command_utils.helm_backend import adhoc, helm, launcher
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.run_uuid import RUN_UUID_LENGTH
from miles.utils.workers.types import ClusterBackend

_RUN_ID_PATTERN = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?")

CLUSTER_BACKEND_FLAG = "--cluster-backend"


class KubernetesCommandBackend(BaseCommandBackend):
    def __init__(self) -> None:
        self._adhoc_context: adhoc.AdhocContext | None = None

    def prepare_for(self, config: Any) -> None:
        assert config.namespace, "set ExecuteTrainConfig.namespace to install the run somewhere"
        self._adhoc_context = adhoc.AdhocContext(
            namespace=config.namespace,
            chart_dir=str(helm.chart_dir(repo_base_dir)),
            infra_values_files=tuple(config.infra_values),
        )

    def execute_train(self, request: ExecuteTrainRequest) -> None:
        config = request.config
        self.prepare_for(config)
        self._adhoc_context = self._context().model_copy(update={"gpus_per_node": request.num_gpus_per_node})
        use_backend(self)

        run_id = stable_run_id(config)
        train_argv = with_run_uuid(
            with_cluster_backend(
                [
                    *shlex.split(shell_safe_model_args(request.megatron_model_type)),
                    *shlex.split(request.train_args),
                ],
                cluster_backend=ClusterBackend.KUBERNETES.value,
            ),
            run_id=run_id,
        )
        args = parse_args_from_argv(train_argv)

        run = launcher.launch(
            request,
            specs=compute_specs(args),
            run_id=run_id,
            namespace=config.namespace,
            shared_root=config.shared_root,
            infra_values_files=list(config.infra_values),
            repo_base_dir=repo_base_dir,
            train_argv=train_argv,
            stage_to_local=config.stage_to_local,
            node_local_root=config.node_local_root,
            force=config.force,
            ci_run=config.ci_run,
        )

        exit_code = launcher.follow_until_finished(run)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def exec_command_cpu(self, cmd: str, capture_output: bool = False) -> str | None:
        return adhoc.run_locally(cmd, capture_output=capture_output)

    def exec_command_gpu(self, cmd: str, capture_output: bool = False) -> str | None:
        return adhoc.run_on_one_gpu_node(self._context(), cmd, capture_output=capture_output)

    def exec_command_multi_node(
        self, cmd: str, capture_output: bool = False, num_nodes: int | None = None
    ) -> list[str | None]:
        context = self._context()
        return adhoc.run_on_nodes(
            context,
            cmd,
            capture_output=capture_output,
            completions=num_nodes or 1,
            gpus_per_pod=context.gpus_per_node,
            step="step",
        )

    def _context(self) -> adhoc.AdhocContext:
        assert self._adhoc_context is not None, (
            "the Kubernetes backend runs adhoc commands as Jobs on the namespace of the run it is "
            "launching, so execute_train has to have chosen that namespace first"
        )
        return self._adhoc_context


def stable_run_id(config: Any) -> str:
    assert config.run_id, (
        "set ExecuteTrainConfig.run_id: it names the helm release and the run directory, so a generated one "
        "would open a new release every time the same run is relaunched and break the elastic upgrade"
    )
    assert _RUN_ID_PATTERN.fullmatch(
        config.run_id
    ), f"run_id {config.run_id!r} is not a valid kubernetes object name; it has to match {_RUN_ID_PATTERN.pattern}"
    return config.run_id


def with_cluster_backend(train_argv: list[str], *, cluster_backend: str) -> list[str]:
    declared = declared_cluster_backends(train_argv)
    conflicting = sorted(set(declared) - {cluster_backend})
    assert not conflicting, (
        f"this run is launched onto kubernetes, so its pods have to be told {CLUSTER_BACKEND_FLAG} "
        f"{cluster_backend}, but the train args already say {conflicting}; drop that flag or launch with the "
        f"backend it names"
    )
    if declared:
        return list(train_argv)
    return [*train_argv, CLUSTER_BACKEND_FLAG, cluster_backend]


def declared_cluster_backends(train_argv: list[str]) -> list[str]:
    declared: list[str] = []
    for index, token in enumerate(train_argv):
        if token == CLUSTER_BACKEND_FLAG:
            assert index + 1 < len(train_argv), f"{CLUSTER_BACKEND_FLAG} is the last train arg, so it names no backend"
            declared.append(train_argv[index + 1])
        elif token.startswith(f"{CLUSTER_BACKEND_FLAG}="):
            declared.append(token.split("=", maxsplit=1)[1])
    return declared


def with_run_uuid(train_argv: list[str], *, run_id: str) -> list[str]:
    if "--run-uuid" in train_argv:
        return train_argv
    return [*train_argv, "--run-uuid", run_uuid_of(run_id)]


def run_uuid_of(run_id: str) -> str:
    return hashlib.blake2b(run_id.encode(), digest_size=RUN_UUID_LENGTH).hexdigest()[:RUN_UUID_LENGTH]
