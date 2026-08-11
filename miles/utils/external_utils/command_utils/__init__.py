"""
This package is not for miles framework itself, but as an optional utility to easily launch miles jobs and tests.
"""

from miles.utils.external_utils.command_utils.base_backend import (
    ExecuteTrainConfig,
    check_has_nvlink,
    convert_checkpoint,
    execute_train,
    fp8_cast_bf16,
    hf_download_dataset,
    resolve_hardware,
    rsync_simple,
    ssh_start_ray_workers,
    start_mooncake_master,
)
from miles.utils.external_utils.command_utils.common import (
    GENERATION_HARDWARE,
    NUM_GPUS_OF_HARDWARE,
    create_run_id,
    detect_hardware,
    encode_pseudo_file,
    get_bool_env_var,
    get_default_wandb_args,
    get_env_enable_infinite_run,
    get_mooncake_object_store_args,
    repo_base_dir,
)
from miles.utils.external_utils.exec_command import exec_command_cpu, exec_command_gpu, exec_command_multi_node
from miles.utils.typer_utils import dataclass_cli

__all__ = [
    "ExecuteTrainConfig",
    "GENERATION_HARDWARE",
    "NUM_GPUS_OF_HARDWARE",
    "check_has_nvlink",
    "convert_checkpoint",
    "create_run_id",
    "dataclass_cli",
    "detect_hardware",
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
    "resolve_hardware",
    "rsync_simple",
    "ssh_start_ray_workers",
    "start_mooncake_master",
]
