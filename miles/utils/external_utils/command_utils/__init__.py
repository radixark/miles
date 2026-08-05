import importlib
import os
from collections.abc import Callable
from typing import Any

from miles.utils.external_utils.command_utils.base_backend import (
    BaseCommandBackend,
    ExecuteTrainConfig,
    ExecuteTrainRequest,
    exec_command_cpu,
    exec_command_gpu,
    exec_command_multi_node,
    use_backend,
)
from miles.utils.external_utils.command_utils.common import (
    GENERATION_HARDWARE,
    MOONCAKE_MASTER_LOG_PATH,
    MOONCAKE_MASTER_METRICS_PORT,
    MOONCAKE_MASTER_PORT,
    NUM_GPUS_OF_HARDWARE,
    _is_tcp_server_ready,
    _parse_extra_env_vars,
    _pythonpath_with_sources,
    check_has_nvlink,
    convert_checkpoint,
    create_run_id,
    encode_pseudo_file,
    fp8_cast_bf16,
    get_bool_env_var,
    get_default_wandb_args,
    get_env_enable_infinite_run,
    get_mooncake_object_store_args,
    hf_download_dataset,
    repo_base_dir,
    rsync_simple,
    start_mooncake_master,
)
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.file_arg_utils import PSEUDO_FILE_PREFIX
from miles.utils.http_utils import wait_for_server_ready
from miles.utils.typer_utils import dataclass_cli, register_post_init_hook
from miles.utils.workers.types import ClusterBackend

__all__ = [
    "BaseCommandBackend",
    "PSEUDO_FILE_PREFIX",
    "ExecuteTrainConfig",
    "ExecuteTrainRequest",
    "GENERATION_HARDWARE",
    "KubernetesCommandBackend",
    "MOONCAKE_MASTER_LOG_PATH",
    "MOONCAKE_MASTER_METRICS_PORT",
    "MOONCAKE_MASTER_PORT",
    "NUM_GPUS_OF_HARDWARE",
    "RayCommandBackend",
    "_parse_extra_env_vars",
    "_pythonpath_with_sources",
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
    "shell_safe_model_args",
    "start_mooncake_master",
    "install_cluster_backend",
    "use_backend",
    "wait_for_server_ready",
    "_is_tcp_server_ready",
]


_BACKEND_CLASS_NAMES = {
    ClusterBackend.RAY.value: "RayCommandBackend",
    ClusterBackend.KUBERNETES.value: "KubernetesCommandBackend",
}

_LAZY_BACKENDS = {
    "KubernetesCommandBackend": "miles.utils.external_utils.command_utils.helm_backend",
    "RayCommandBackend": "miles.utils.external_utils.command_utils.ray_backend",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_BACKENDS:
        return getattr(importlib.import_module(_LAZY_BACKENDS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _resolve_backend(cluster_backend: str, *, train_args: str) -> BaseCommandBackend:
    flag = f"--cluster-backend {cluster_backend}"
    other_backends = [backend.value for backend in ClusterBackend if backend.value != cluster_backend]
    conflicting = [name for name in other_backends if f"--cluster-backend {name}" in train_args]
    assert not conflicting, (
        f"ExecuteTrainConfig asks for {cluster_backend} but the train args say {conflicting}; "
        f"pass {flag} to agree, or change the config"
    )

    assert cluster_backend in _BACKEND_CLASS_NAMES, f"unknown cluster backend {cluster_backend!r}"
    return __getattr__(_BACKEND_CLASS_NAMES[cluster_backend])()


def install_cluster_backend(config: object) -> None:
    if not isinstance(config, ExecuteTrainConfig):
        return
    backend = __getattr__(_BACKEND_CLASS_NAMES[config.cluster_backend])()
    if hasattr(backend, "prepare_for"):
        backend.prepare_for(config)
    use_backend(backend)


register_post_init_hook(install_cluster_backend)


def execute_train(
    train_args: str,
    num_gpus_per_node: int,
    megatron_model_type: str | None,
    train_script: str = "train.py",
    before_ray_job_submit: Callable[[], None] | None = None,
    extra_env_vars: dict[str, str] | None = None,
    config: ExecuteTrainConfig | None = None,
    megatron_path: str = "/root/Megatron-LM",
):
    if config is None:
        config = ExecuteTrainConfig()
    if not os.path.isabs(train_script):
        train_script = f"{repo_base_dir}/{train_script}"

    train_backend_fsdp = "--train-backend fsdp" in train_args
    assert train_backend_fsdp == (megatron_model_type is None)

    backend = _resolve_backend(config.cluster_backend, train_args=train_args)

    request = ExecuteTrainRequest(
        train_args=train_args,
        num_gpus_per_node=num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        train_script=train_script,
        train_backend_fsdp=train_backend_fsdp,
        extra_env_vars=extra_env_vars if extra_env_vars is not None else {},
        config=config,
        megatron_path=megatron_path,
        before_ray_job_submit=before_ray_job_submit,
    )
    backend.execute_train(request)
