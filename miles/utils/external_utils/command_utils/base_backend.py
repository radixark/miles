from __future__ import annotations

import logging
import os
import shlex
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import get_args

from miles.utils.external_utils.command_utils.common import (
    ArgvManipulator,
    _parse_extra_env_vars,
    _pythonpath_with_sources,
    create_run_id,
    detect_hardware,
    repo_base_dir,
    run_shell_command,
)
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.typer_utils import dataclass_from_env
from miles.utils.workers.types import ClusterBackend

logger = logging.getLogger(__name__)


# This class can be extended by concrete scripts
@dataclass
class ExecuteTrainConfig:
    cuda_core_dump: bool = False
    num_nodes: int = field(default_factory=lambda: int(os.environ.get("SLURM_JOB_NUM_NODES", "1")))
    extra_env_vars: str = ""
    output_dir: str = "/root/shared_data"
    cluster_backend: ClusterBackend = ClusterBackend.RAY
    run_id: str = field(default_factory=create_run_id)
    namespace: str = ""
    helm_values: tuple[str, ...] = ()
    skip_upgrade_check: bool = False
    ci_run: bool = False

    def create_backend(self) -> BaseCommandBackend:
        match self.cluster_backend:
            case ClusterBackend.KUBERNETES:
                from miles.utils.external_utils.command_utils.helm_backend.backend import KubernetesCommandBackend

                return KubernetesCommandBackend(self)
            case ClusterBackend.RAY:
                from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend

                return RayCommandBackend(self)


def default_config(config_class: type = ExecuteTrainConfig) -> ExecuteTrainConfig:
    return dataclass_from_env(config_class)


class ExecuteTrainRequest(FrozenStrictBaseModel):
    train_args: str
    num_gpus_per_node: int
    megatron_model_type: str | None
    train_script: str
    train_backend_fsdp: bool
    extra_env_vars: dict[str, str]
    megatron_path: str
    before_ray_job_submit: Callable[[], None] | None
    prepare_cmd: dict[str, str]


CLUSTER_BACKEND_FLAG = "--cluster-backend"

TRAINER_ROLE = "trainer"
_PREPARE_CMD_ROLES = frozenset({TRAINER_ROLE})


class BaseCommandBackend(ABC):
    def __init__(self, config: ExecuteTrainConfig) -> None:
        from miles.utils.logging_utils import configure_logger_raw

        configure_logger_raw("launcher")
        self.config = config

    def execute_train(
        self,
        train_args: str,
        num_gpus_per_node: int,
        megatron_model_type: str | None,
        train_script: str = "train.py",
        before_ray_job_submit: Callable[[], None] | None = None,
        extra_env_vars: dict[str, str] | None = None,
        megatron_path: str = "/root/Megatron-LM",
        prepare_cmd: dict[str, str] | None = None,
    ) -> None:
        prepare_cmd = prepare_cmd if prepare_cmd is not None else {}
        assert set(prepare_cmd) <= _PREPARE_CMD_ROLES, (
            f"prepare_cmd names the roles {sorted(set(prepare_cmd) - _PREPARE_CMD_ROLES)}, but a backend only "
            f"knows how to run a preparation command for {sorted(_PREPARE_CMD_ROLES)}"
        )

        if not os.path.isabs(train_script):
            train_script = f"{repo_base_dir}/{train_script}"

        train_argv = shlex.split(train_args)
        train_backend_fsdp = "fsdp" in ArgvManipulator.values_of(train_argv, "--train-backend")
        assert train_backend_fsdp == (megatron_model_type is None)
        _assert_train_args_name_no_other_backend(train_argv, cluster_backend=self.config.cluster_backend.value)

        self._execute_train_inner(
            ExecuteTrainRequest(
                train_args=train_args,
                num_gpus_per_node=num_gpus_per_node,
                megatron_model_type=megatron_model_type,
                train_script=train_script,
                train_backend_fsdp=train_backend_fsdp,
                extra_env_vars=extra_env_vars if extra_env_vars is not None else {},
                megatron_path=megatron_path,
                before_ray_job_submit=before_ray_job_submit,
                prepare_cmd=prepare_cmd,
            )
        )

    def convert_checkpoint(
        self,
        model_name,
        megatron_model_type,
        num_gpus_per_node: int,
        multinode: bool = False,
        num_nodes: int | None = None,
        extra_args: str = "",
        dir_dst: str = "/root",
        hf_checkpoint: str | None = None,
        megatron_path: str = "/root/Megatron-LM",
    ):
        hf_checkpoint = hf_checkpoint or f"/root/models/{model_name}"

        # TODO shall we make it in host-mapped folder and thus can cache it to speedup CI
        path_dst = f"{dir_dst}/{model_name}_torch_dist"
        tracker = Path(path_dst) / "latest_checkpointed_iteration.txt"
        if tracker.exists() and tracker.read_text().strip() == "release":
            logger.info(f"convert_checkpoint skip {path_dst} since tracker is 'release'")
            return

        multinode_args = ""
        if multinode:
            multinode_args = (
                "--master-addr {{master_addr}} "
                "--master-port 23456 "
                "--nnodes={{nnodes}} "
                "--node-rank {{node_rank}} "
            )

        if multinode:
            fn = partial(self.exec_command_multi_node, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
        else:
            fn = partial(self.exec_command_gpu, num_gpus_per_node=num_gpus_per_node)
        pythonpath = shlex.quote(_pythonpath_with_sources(megatron_path))
        fn(
            f"PYTHONPATH={pythonpath} "
            f"torchrun "
            f"--nproc-per-node {num_gpus_per_node} "
            f"{multinode_args}"
            f"{repo_base_dir}/tools/convert_hf_to_torch_dist.py "
            f"{shell_safe_model_args(megatron_model_type)} "
            f"--hf-checkpoint {hf_checkpoint} "
            f"--save {path_dst} "
            f"{extra_args}"
        )

    def ssh_start_ray_workers(
        self,
        master_addr: str,
        num_gpus_per_node: int,
        hostfile: str = "/root/mpi_rack_hostfile",
        head_host: str | None = None,
    ):
        """Join every host in an MPI-style hostfile to the ray cluster over ssh, in parallel.

        Ray itself cannot bring up the workers: the head is already running locally and the
        workers have no agent yet. Pass this as `execute_train(before_ray_job_submit=...)` so
        the cluster is complete before the job is submitted.
        """
        head_host = head_host or master_addr
        self.exec_command_cpu(
            f"for worker_ip in $(awk '{{print $1}}' {hostfile}); do "
            f'if [ "$worker_ip" = {shlex.quote(head_host)} ]; then continue; fi; '
            'echo "Starting Ray worker on $worker_ip"; '
            'ssh root@"$worker_ip" '
            '"pkill -9 sglang ; ray stop --force ; pkill -9 miles ; '
            f"ray start --address={master_addr}:6379 --num-gpus {num_gpus_per_node} "
            '--node-ip-address $worker_ip --disable-usage-stats" & '
            "done; wait"
        )

    def hf_download_dataset(self, full_name: str, data_dir: str = "/root/datasets"):
        _, partial_name = full_name.split("/")
        self.exec_command_cpu(f"hf download --repo-type dataset {full_name} --local-dir {data_dir}/{partial_name}")

    def fp8_cast_bf16(self, path_src, path_dst):
        sentinel = Path(path_dst) / "model.safetensors.index.json"
        if sentinel.exists():
            logger.info(f"fp8_cast_bf16 skip {path_dst} since {sentinel} exists")
            return

        self.exec_command_gpu(
            f"python {repo_base_dir}/tools/fp8_cast_bf16.py "
            f"--input-fp8-hf-path {path_src} "
            f"--output-bf16-hf-path {path_dst} "
        )

    def api_server_host(self) -> str:
        return "localhost"

    @abstractmethod
    def _execute_train_inner(self, request: ExecuteTrainRequest) -> None: ...

    def exec_command_cpu(self, cmd: str, capture_output: bool = False) -> str | None:
        return run_shell_command(cmd, capture_output=capture_output)

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


def resolve_extra_env_vars(extra_env_vars: dict[str, str], config: ExecuteTrainConfig) -> dict[str, str]:
    return {
        **extra_env_vars,
        **_parse_extra_env_vars(config.extra_env_vars),
    }


def resolve_hardware(config: ExecuteTrainConfig) -> str:
    """`auto` asks the node the launcher runs on; anything explicit overrides it."""
    if config.hardware == "auto":
        hardware = detect_hardware()
        logger.info(f"detected --hardware {hardware}")
    else:
        hardware = config.hardware
    supported = get_args(config.__dataclass_fields__["hardware"].type)
    assert hardware in supported, f"{type(config).__name__} has no verified profile for {hardware}"
    return hardware


def _declared_cluster_backends(train_argv: list[str]) -> list[str]:
    return ArgvManipulator.values_of(train_argv, CLUSTER_BACKEND_FLAG)


def _assert_train_args_name_no_other_backend(train_argv: list[str], *, cluster_backend: str) -> None:
    conflicting = sorted(set(_declared_cluster_backends(train_argv)) - {cluster_backend})
    assert not conflicting, (
        f"This run is launched onto {cluster_backend}, so its pods have to be told {CLUSTER_BACKEND_FLAG} "
        f"{cluster_backend}, but the train args already say {conflicting}; drop that flag or launch with the "
        f"backend it names"
    )
