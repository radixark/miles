from __future__ import annotations

import hashlib
import re
import shlex
from typing import Any

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.arguments import parse_args_from_argv
from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend, ExecuteTrainRequest
from miles.utils.external_utils.command_utils.common import repo_base_dir
from miles.utils.external_utils.command_utils.helm_backend import launcher
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.run_uuid import RUN_UUID_LENGTH
from miles.utils.workers.types import ClusterBackend


_RUN_ID_PATTERN = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?")

CLUSTER_BACKEND_FLAG = "--cluster-backend"


class KubernetesCommandBackend(BaseCommandBackend):
    def execute_train(self, request: ExecuteTrainRequest) -> None:
        config = request.config
        assert config.namespace, "set ExecuteTrainConfig.namespace to install the run somewhere"

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
