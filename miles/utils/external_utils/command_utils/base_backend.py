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
