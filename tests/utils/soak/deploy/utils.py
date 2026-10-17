import dataclasses
from pathlib import Path

from tests.utils.soak.core.utils import compute_release_of_config

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.workers.types import HOT_RESTART_SEPARATOR, HotRestartComponent

# ============================== run directories ===============================


HOT_RESTART_ARG: str = HOT_RESTART_SEPARATOR.join(one.value for one in HotRestartComponent)

CHECKPOINT_DIRNAME: str = "checkpoints"


def compute_checkpoint_dir(dump_dir: str) -> Path:
    return Path(dump_dir) / CHECKPOINT_DIRNAME


def compute_hot_restart_config(config: ExecuteTrainConfig, *, installed_release: str) -> ExecuteTrainConfig:
    assert (relaunched := compute_release_of_config(config)) == installed_release, (
        f"a hot restart upgrades the release that is already up: this relaunch would install {relaunched}, not the "
        f"watched {installed_release}, so it built a config with a run id of its own and would leave the trainers behind"
    )
    return dataclasses.replace(config, hot_restart=HOT_RESTART_ARG)
