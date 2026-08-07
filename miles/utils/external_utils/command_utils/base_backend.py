from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import SkipValidation

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.types import ClusterBackend


# This class can be extended by concrete scripts
@dataclass
class ExecuteTrainConfig:
    cuda_core_dump: bool = False
    num_nodes: int = field(default_factory=lambda: int(os.environ.get("SLURM_JOB_NUM_NODES", "1")))
    extra_env_vars: str = ""
    output_dir: str = "/root/shared_data"
    cluster_backend: str = ClusterBackend.RAY.value
    run_id: str = ""
    namespace: str = ""
    shared_root: str = ""
    infra_values: tuple[str, ...] = ()
    stage_to_local: tuple[str, ...] = ()
    node_local_root: str = ""
    force: bool = False
    ci_run: bool = False


class ExecuteTrainRequest(FrozenStrictBaseModel):
    train_args: str
    num_gpus_per_node: int
    megatron_model_type: str | None
    train_script: str
    train_backend_fsdp: bool
    extra_env_vars: dict[str, str]
    config: SkipValidation[ExecuteTrainConfig]
    megatron_path: str
    before_ray_job_submit: Callable[[], None] | None


class BaseCommandBackend(ABC):
    @abstractmethod
    def execute_train(self, request: ExecuteTrainRequest) -> None: ...

    @abstractmethod
    def exec_command_cpu(self, cmd: str, capture_output: bool = False) -> str | None: ...

    @abstractmethod
    def exec_command_gpu(
        self, cmd: str, capture_output: bool = False, num_gpus_per_node: int | None = None
    ) -> str | None: ...

    @abstractmethod
    def exec_command_multi_node(
        self,
        cmd: str,
        capture_output: bool = False,
        num_nodes: int | None = None,
        num_gpus_per_node: int | None = None,
    ) -> list[str | None]: ...


_active_backend: BaseCommandBackend | None = None


def use_backend(backend: BaseCommandBackend) -> None:
    global _active_backend
    _active_backend = backend


def active_backend() -> BaseCommandBackend:
    global _active_backend
    if _active_backend is None:
        from miles.utils.external_utils.command_utils import install_cluster_backend
        from miles.utils.external_utils.command_utils.env_config import config_from_env

        install_cluster_backend(config_from_env())
        assert _active_backend is not None, "installing a backend is what makes one active"
    return _active_backend


def exec_command_cpu(cmd: str, capture_output: bool = False) -> str | None:
    return active_backend().exec_command_cpu(cmd, capture_output=capture_output)


def exec_command_gpu(cmd: str, capture_output: bool = False, num_gpus_per_node: int | None = None) -> str | None:
    return active_backend().exec_command_gpu(cmd, capture_output=capture_output, num_gpus_per_node=num_gpus_per_node)


def exec_command_multi_node(
    cmd: str, capture_output: bool = False, num_nodes: int | None = None, num_gpus_per_node: int | None = None
) -> list[str | None]:
    return active_backend().exec_command_multi_node(
        cmd, capture_output=capture_output, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node
    )
