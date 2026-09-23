from pathlib import Path

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.workers.types import HOT_RESTART_SEPARATOR, HotRestartComponent

# ============================== run directories ===============================


HOT_RESTART_ARG: str = HOT_RESTART_SEPARATOR.join(one.value for one in HotRestartComponent)

CHECKPOINT_DIRNAME: str = "checkpoints"


def compute_checkpoint_dir(dump_dir: str) -> Path:
    raise NotImplementedError


def compute_hot_restart_config(config: ExecuteTrainConfig, *, installed_release: str) -> ExecuteTrainConfig:
    raise NotImplementedError
