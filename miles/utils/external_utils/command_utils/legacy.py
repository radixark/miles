from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig as _CurrentExecuteTrainConfig
from miles.utils.external_utils.command_utils.common import (
    GENERATION_HARDWARE,
    MOONCAKE_MASTER_LOG_PATH,
    MOONCAKE_MASTER_METRICS_PORT,
    MOONCAKE_MASTER_PORT,
    NUM_GPUS_OF_HARDWARE,
    create_run_id,
    encode_pseudo_file,
    get_bool_env_var,
    get_default_wandb_args,
    get_env_enable_infinite_run,
    get_mooncake_object_store_args,
    repo_base_dir,
    rsync_cmd,
)
from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend
from miles.utils.external_utils.command_utils.ray_backend.command import start_mooncake_master
from miles.utils.typer_utils import dataclass_cli
from miles.utils.workers.types import ClusterBackend

__all__ = [
    "ExecuteTrainConfig",
    "GENERATION_HARDWARE",
    "MOONCAKE_MASTER_LOG_PATH",
    "MOONCAKE_MASTER_METRICS_PORT",
    "MOONCAKE_MASTER_PORT",
    "NUM_GPUS_OF_HARDWARE",
    "check_has_nvlink",
    "convert_checkpoint",
    "create_run_id",
    "dataclass_cli",
    "encode_pseudo_file",
    "exec_command_cpu",
    "exec_command_gpu",
    "exec_command_multi_node",
    "execute_train",
    "fp8_cast_bf16",
    "get_bool_env_var",
    "get_default_wandb_args",
    "get_env_enable_infinite_run",
    "get_mooncake_object_store_args",
    "hf_download_dataset",
    "repo_base_dir",
    "rsync_simple",
    "ssh_start_ray_workers",
    "start_mooncake_master",
]


@dataclass
class ExecuteTrainConfig:
    cuda_core_dump: bool = False
    num_nodes: int = field(default_factory=lambda: int(os.environ.get("SLURM_JOB_NUM_NODES", "1")))
    extra_env_vars: str = ""
    output_dir: str = "/root/shared_data"


def execute_train(
    train_args: str,
    num_gpus_per_node: int,
    megatron_model_type: str | None,
    train_script: str = "train.py",
    before_ray_job_submit: Callable[[], None] | None = None,
    extra_env_vars: dict[str, str] | None = None,
    config: ExecuteTrainConfig | None = None,
    megatron_path: str = "/root/Megatron-LM",
) -> None:
    if config is None:
        config = ExecuteTrainConfig()
    current_config = _to_current_config(config)
    _create_ray_backend(current_config).execute_train(
        config=current_config,
        train_args=train_args,
        num_gpus_per_node=num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        train_script=train_script,
        before_ray_job_submit=before_ray_job_submit,
        extra_env_vars=extra_env_vars,
        megatron_path=megatron_path,
    )


def exec_command_cpu(cmd: str, capture_output: bool = False) -> str | None:
    return _create_ray_backend().exec_command_cpu(cmd, capture_output=capture_output)


def exec_command_gpu(cmd: str, capture_output: bool = False) -> str | None:
    return _create_ray_backend().exec_command_gpu(cmd, capture_output=capture_output)


def exec_command_multi_node(cmd: str, capture_output: bool = False, num_nodes: int | None = None) -> list[str | None]:
    return _create_ray_backend().exec_command_multi_node(cmd, capture_output=capture_output, num_nodes=num_nodes)


def convert_checkpoint(
    model_name: str,
    megatron_model_type: str | None,
    num_gpus_per_node: int,
    multinode: bool = False,
    num_nodes: int | None = None,
    extra_args: str = "",
    dir_dst: str = "/root",
    hf_checkpoint: str | None = None,
    megatron_path: str = "/root/Megatron-LM",
) -> None:
    _create_ray_backend().convert_checkpoint(
        model_name=model_name,
        megatron_model_type=megatron_model_type,
        num_gpus_per_node=num_gpus_per_node,
        multinode=multinode,
        num_nodes=num_nodes,
        extra_args=extra_args,
        dir_dst=dir_dst,
        hf_checkpoint=hf_checkpoint,
        megatron_path=megatron_path,
    )


def rsync_simple(path_src: str, path_dst: str, num_nodes: int | None = None) -> None:
    _create_ray_backend().exec_command_multi_node(rsync_cmd(path_src=path_src, path_dst=path_dst), num_nodes=num_nodes)


def ssh_start_ray_workers(
    master_addr: str,
    num_gpus_per_node: int,
    hostfile: str = "/root/mpi_rack_hostfile",
    head_host: str | None = None,
) -> None:
    _create_ray_backend().ssh_start_ray_workers(
        master_addr=master_addr,
        num_gpus_per_node=num_gpus_per_node,
        hostfile=hostfile,
        head_host=head_host,
    )


def hf_download_dataset(full_name: str, data_dir: str = "/root/datasets") -> None:
    _create_ray_backend().hf_download_dataset(full_name=full_name, data_dir=data_dir)


def fp8_cast_bf16(path_src: str, path_dst: str) -> None:
    _create_ray_backend().fp8_cast_bf16(path_src=path_src, path_dst=path_dst)


def check_has_nvlink() -> bool:
    return _create_ray_backend()._check_has_nvlink()


def _to_current_config(config: ExecuteTrainConfig) -> _CurrentExecuteTrainConfig:
    return _CurrentExecuteTrainConfig(
        cuda_core_dump=config.cuda_core_dump,
        num_nodes=config.num_nodes,
        extra_env_vars=config.extra_env_vars,
        output_dir=config.output_dir,
    )


def _create_ray_backend(config: _CurrentExecuteTrainConfig | None = None) -> RayCommandBackend:
    if config is None:
        config = _CurrentExecuteTrainConfig()
    assert config.cluster_backend is ClusterBackend.RAY, (
        f"This is the v1 command_utils API, which was written before miles could launch onto anything but a ray "
        f"cluster and therefore has no way to express the namespace, release and manifests a "
        f"{config.cluster_backend.value} launch is made of, yet the config it was handed asks for "
        f"{config.cluster_backend.value}; drop this module and drive the launch through the v2 backend "
        f"API instead, i.e. build the backend with config.create_backend() and call execute_train on it"
    )
    backend = config.create_backend()
    assert isinstance(backend, RayCommandBackend)
    return backend
